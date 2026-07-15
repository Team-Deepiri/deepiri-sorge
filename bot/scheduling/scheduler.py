"""Central review scheduler — providers are backends; no worker-owned fallback."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from bot.file_splitter import ReviewChunk
from bot.providers.base import Provider
from bot.quota_tracker import QuotaTracker
from bot.scheduling.health import HealthTracker
from bot.scheduling.market_score import score_provider
from bot.scheduling.run_context import ProviderRuntime, RunContext
from bot.scheduling.token_bucket import TokenBucket
from bot.scheduling.types import (
    SchedulerMeta,
    ScheduledChunk,
    SkipRecord,
)
from bot.schemas import ReviewResult


class ReviewScheduler:
    def __init__(
        self,
        providers: list[Provider],
        run: RunContext,
        *,
        max_workers: int = 4,
    ):
        self.providers = {p.name: p for p in providers}
        self.ctx = run
        self.max_workers = max(1, max_workers)
        self.meta = SchedulerMeta()

    @classmethod
    def from_providers(
        cls,
        providers: list[Provider],
        quota: QuotaTracker,
        config,
        *,
        repo_context: str | None = None,
        context_fingerprint: str = "",
    ) -> "ReviewScheduler":
        deadline = time.monotonic() + float(config.scheduler.wall_clock_sec)
        runtimes: dict[str, ProviderRuntime] = {}
        for p in providers:
            rt_cfg = getattr(config.providers, p.name)
            runtimes[p.name] = ProviderRuntime(
                name=p.name,
                bucket=TokenBucket(rt_cfg.rpm, capacity=rt_cfg.rpm),
                health=HealthTracker(100.0),
                max_context_tokens=rt_cfg.max_context_tokens,
                max_inflight=rt_cfg.max_inflight,
                nominal_latency_ms=rt_cfg.nominal_latency_ms,
                quality_prior=rt_cfg.quality_prior,
            )
        ctx = RunContext(
            providers=runtimes,
            quota=quota,
            deadline=deadline,
            repo_context=repo_context,
            context_fingerprint=context_fingerprint,
            health_threshold=config.scheduler.health_threshold,
        )
        return cls(providers, ctx, max_workers=config.scheduler.max_workers)

    def run(self, chunks: list[ReviewChunk]) -> tuple[list[ReviewResult], list[SkipRecord], SchedulerMeta]:
        queue = [
            ScheduledChunk(chunk=c, priority=50)
            for c in chunks
            if not c.unreviewable
        ]
        # Stable order: larger chunks first (more valuable), then FIFO
        queue.sort(key=lambda s: (-s.chunk.estimated_tokens, -s.priority))

        results: list[ReviewResult] = []
        skipped: list[SkipRecord] = []

        while queue and self.ctx.alive():
            # Drop cooling / exhausted state: try to dispatch as many as concurrency allows
            desired = self._desired_workers()
            batch: list[tuple[ScheduledChunk, str]] = []
            still_waiting: list[ScheduledChunk] = []

            for scheduled in queue:
                if len(batch) >= desired:
                    still_waiting.append(scheduled)
                    continue
                pick = self._pick_provider(scheduled)
                if pick is None:
                    still_waiting.append(scheduled)
                    continue
                if not self.ctx.try_acquire(pick):
                    still_waiting.append(scheduled)
                    continue
                batch.append((scheduled, pick))

            if not batch:
                # Nothing eligible right now — wait for shortest cooldown or give up
                wait = self._min_cooldown()
                if wait is None or wait > self.ctx.remaining_sec():
                    for s in still_waiting:
                        skipped.append(SkipRecord(s.chunk, "no eligible provider (rate limits / health)"))
                    self.meta.stop_reason = "providers_exhausted"
                    break
                logger.info(f"Scheduler waiting {wait:.1f}s for provider cooldown")
                time.sleep(min(wait, 15.0))
                queue = still_waiting
                continue

            # Dispatch batch — workers only call provider.review; no fallback inside
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {
                    pool.submit(self._dispatch, scheduled, name): (scheduled, name)
                    for scheduled, name in batch
                }
                for fut in as_completed(futures):
                    scheduled, name = futures[fut]
                    try:
                        outcome = fut.result()
                    except Exception as exc:
                        logger.warning(f"Dispatch error on {name}: {exc}")
                        self.ctx.release(name)
                        scheduled.attempted_providers.add(name)
                        still_waiting.append(scheduled)
                        continue

                    self.ctx.release(name)
                    scheduled.attempted_providers.add(name)
                    self.meta.dispatches += 1
                    self.meta.provider_picks.append(
                        {
                            "provider": name,
                            "tokens": scheduled.chunk.estimated_tokens,
                            "ok": outcome.ok,
                            "status": outcome.status_code,
                        }
                    )

                    if outcome.ok and outcome.result:
                        self._record_success(name, outcome.latency_ms)
                        self.ctx.quota.record(name)
                        results.append(outcome.result)
                    else:
                        self._record_failure(name, outcome)
                        # Requeue for a different provider if any remain
                        if self._has_untried_eligible(scheduled):
                            still_waiting.append(scheduled)
                        else:
                            skipped.append(
                                SkipRecord(
                                    scheduled.chunk,
                                    outcome.error or f"failed on {name}",
                                )
                            )

            queue = still_waiting

        if queue and not self.ctx.alive():
            self.meta.stop_reason = "deadline"
            for s in queue:
                skipped.append(SkipRecord(s.chunk, "wall-clock deadline"))
        elif queue and not skipped:
            for s in queue:
                skipped.append(SkipRecord(s.chunk, "incomplete"))

        self.meta.skipped = len(skipped)
        self.meta.health_snapshot = {
            name: rt.health.score for name, rt in self.ctx.providers.items()
        }
        return results, skipped, self.meta

    def _desired_workers(self) -> int:
        healthy = 0
        for name, rt in self.ctx.providers.items():
            if rt.health.is_cooling():
                continue
            if rt.health.score < self.ctx.health_threshold:
                continue
            if rt.bucket.remaining() < 1.0:
                continue
            if not self.ctx.quota.can_use(name):
                continue
            healthy += 1
        return max(1, min(self.max_workers, healthy or 1))

    def _pick_provider(self, scheduled: ScheduledChunk) -> str | None:
        best_name: str | None = None
        best_score = 0.0
        for name, provider in self.providers.items():
            if name in scheduled.attempted_providers:
                continue
            status = self.ctx.status(name)
            if not status:
                continue
            if status.in_flight >= status.max_inflight:
                continue
            if self.ctx.providers[name].health.is_cooling():
                continue
            if status.health < self.ctx.health_threshold:
                continue
            if status.rpm_remaining < 1.0:
                continue
            if not self.ctx.quota.can_use(name):
                continue
            sc = score_provider(
                status,
                scheduled,
                historical_quality=status.quality_prior,
            )
            if sc > best_score:
                best_score = sc
                best_name = name
        return best_name if best_score > 0 else None

    def _has_untried_eligible(self, scheduled: ScheduledChunk) -> bool:
        probe = ScheduledChunk(
            chunk=scheduled.chunk,
            priority=scheduled.priority,
            attempted_providers=set(scheduled.attempted_providers),
        )
        return self._pick_provider(probe) is not None

    def _min_cooldown(self) -> float | None:
        waits = [
            rt.health.cooling_remaining()
            for rt in self.ctx.providers.values()
            if rt.health.is_cooling()
        ]
        return min(waits) if waits else None

    def _dispatch(self, scheduled: ScheduledChunk, name: str):
        provider = self.providers[name]
        logger.info(
            f"Scheduler → {name} for chunk ({scheduled.chunk.estimated_tokens} tokens)"
        )
        return provider.review(scheduled.chunk, self.ctx)

    def _record_success(self, name: str, latency_ms: float) -> None:
        rt = self.ctx.providers[name]
        rt.health.record_success(latency_ms)

    def _record_failure(self, name: str, outcome) -> None:
        rt = self.ctx.providers[name]
        if outcome.timed_out:
            rt.health.record_timeout()
        elif outcome.is_rate_limited:
            rt.health.record_rate_limit(outcome.retry_after)
            # Soft-count failures so quota routing avoids hot providers
            self.ctx.quota.record_failure(name)
        elif outcome.is_payload_too_large:
            rt.health.record_payload_too_large()
        elif outcome.status_code and outcome.status_code >= 500:
            rt.health.record_server_error()
        else:
            rt.health.record_server_error()
