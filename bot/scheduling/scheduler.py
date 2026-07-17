"""Central review scheduler — providers are backends; no worker-owned fallback.

Quality-aware: complexity + security bias primary pick; optional Gemini escalate
after a successful but low-confidence Groq triage.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from bot.file_splitter import ReviewChunk
from bot.providers.base import Provider
from bot.quota_tracker import QuotaTracker
from bot.scheduling.complexity import (
    complexity_score,
    is_high_complexity,
    is_security_sensitive,
)
from bot.scheduling.health import HealthTracker
from bot.scheduling.history import ProviderHistory
from bot.scheduling.market_score import (
    TEMPLATE_OVERHEAD_TOKENS,
    effective_tokens,
    pick_reason,
    score_provider,
)
from bot.scheduling.priority import prioritize_chunk, sort_key
from bot.scheduling.run_context import ProviderRuntime, RunContext
from bot.scheduling.token_bucket import TokenBucket
from bot.escalate_ledger import EscalateLedger
from bot.scheduling.escalate import (
    EscalateTicket,
    new_ticket_id,
    truncate_diff,
)
from bot.scheduling.types import (
    ESCALATE_SCORE_THRESHOLD,
    MAX_RATE_LIMIT_ROUNDS,
    ProviderResult,
    SchedulerMeta,
    ScheduledChunk,
    SkipRecord,
)
from bot.schemas import ReviewResult, issues_from_parsed
from bot.utils import cache as review_cache


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
        prompt_overhead_tokens: int = 0,
        repo: str = "",
        pr_number: int = 0,
        installation_id: int | None = None,
        head_sha: str = "",
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
        cache_cfg = getattr(config, "cache", None)
        history = ProviderHistory()
        overhead = max(0, int(prompt_overhead_tokens))
        if overhead <= 0:
            overhead = TEMPLATE_OVERHEAD_TOKENS
        from bot.scheduling.semaphore import ProviderSemaphore

        max_cap_wait = float(
            getattr(config.scheduler, "max_capacity_wait_sec", 120.0) or 120.0
        )
        ctx = RunContext(
            providers=runtimes,
            quota=quota,
            deadline=deadline,
            repo_context=repo_context,
            context_fingerprint=context_fingerprint,
            health_threshold=config.scheduler.health_threshold,
            cache_enabled=bool(cache_cfg and cache_cfg.enabled),
            cache_ttl_hours=int(cache_cfg.ttl_hours) if cache_cfg else 24,
            history=history,
            prompt_overhead_tokens=overhead,
            repo=repo or "",
            pr_number=int(pr_number or 0),
            installation_id=installation_id,
            head_sha=head_sha or "",
            max_capacity_wait_sec=max_cap_wait,
            semaphore=ProviderSemaphore(),
        )
        logger.info(
            f"Provider history loaded ({len(getattr(history, '_stats', {}))} keys); "
            f"prompt_overhead_tokens={overhead}; "
            f"max_capacity_wait_sec={max_cap_wait:.0f}"
        )
        return cls(providers, ctx, max_workers=config.scheduler.max_workers)

    def run(self, chunks: list[ReviewChunk]) -> tuple[list[ReviewResult], list[SkipRecord], SchedulerMeta]:
        overhead = self.ctx.prompt_overhead_tokens
        queue = [
            ScheduledChunk(
                chunk=c,
                priority=prioritize_chunk(c),
                complexity=complexity_score(c, prompt_overhead_tokens=overhead),
            )
            for c in chunks
            if not c.unreviewable
        ]
        queue.sort(key=sort_key)
        if queue:
            self.meta.avg_complexity = sum(s.complexity for s in queue) / len(queue)

        results: list[ReviewResult] = []
        skipped: list[SkipRecord] = []
        pending_escalates: list[tuple[ScheduledChunk, ReviewResult, str]] = []

        while queue and self.ctx.alive():
            still_waiting: list[ScheduledChunk] = []
            for scheduled in queue:
                cached = self._try_cache_hit(scheduled)
                if cached is not None:
                    results.append(cached)
                    self.meta.cache_hits += 1
                    self.meta.provider_picks.append(
                        self._pick_meta(scheduled, "cache", ok=True, status=200, score=None)
                    )
                else:
                    still_waiting.append(scheduled)
            queue = still_waiting
            if not queue:
                break

            desired = self._desired_workers()
            batch: list[tuple[ScheduledChunk, str]] = []
            still_waiting = []

            for scheduled in queue:
                if not self.ctx.alive():
                    still_waiting.append(scheduled)
                    continue
                if len(batch) >= desired:
                    still_waiting.append(scheduled)
                    continue
                pick, pick_score = self._pick_provider_scored(scheduled)
                if pick is None:
                    # Soft reserve blocked every fit — spend the last free-tier
                    # slot rather than parking on a cooling OpenRouter/Groq.
                    pick, pick_score = self._pick_provider_scored(
                        scheduled, respect_soft_reserve=False
                    )
                if pick is None:
                    still_waiting.append(scheduled)
                    continue
                if not self.ctx.try_acquire(pick):
                    still_waiting.append(scheduled)
                    continue
                scheduled._last_pick_score = pick_score  # type: ignore[attr-defined]
                batch.append((scheduled, pick))

            if not batch:
                wait = self._min_wake_sec()
                remaining = self.ctx.remaining_sec()
                budget = self.ctx.capacity_budget_remaining()
                if wait is not None and wait > remaining:
                    wait = remaining - 5.0 if remaining >= 15.0 else None
                # Soft cap: wait for cooldowns, but don't park the job for 10+ minutes.
                if wait is not None and wait > budget:
                    if budget >= 20.0:
                        wait = budget
                    else:
                        wait = None
                if wait is None or wait <= 0 or budget < 5.0:
                    cool_hint = 0.0
                    if isinstance(self.ctx.history, ProviderHistory):
                        cool_hint = self.ctx.history.max_cooling_remaining()
                    for s in still_waiting:
                        reason = (
                            f"no eligible provider (priority={s.priority}; "
                            f"rate limits / health"
                        )
                        if cool_hint > 0:
                            mins = max(1, int((cool_hint + 59) // 60))
                            reason += f"; retry_in_approx_{mins}m"
                        if self.ctx.capacity_waited_sec > 0:
                            reason += (
                                f"; capacity_waited={self.ctx.capacity_waited_sec:.0f}s"
                            )
                        reason += ")"
                        skipped.append(SkipRecord(s.chunk, reason))
                    self.meta.stop_reason = "providers_exhausted"
                    self.meta.retry_after_sec = cool_hint if cool_hint > 0 else 90.0
                    logger.info(
                        f"Early defer after capacity wait "
                        f"{self.ctx.capacity_waited_sec:.0f}s/"
                        f"{self.ctx.max_capacity_wait_sec:.0f}s budget"
                    )
                    break
                sleep_for = min(wait, 15.0, budget)
                logger.info(
                    f"Scheduler waiting {sleep_for:.1f}s for provider capacity "
                    f"(waited {self.ctx.capacity_waited_sec:.0f}s/"
                    f"{self.ctx.max_capacity_wait_sec:.0f}s)"
                )
                time.sleep(sleep_for)
                self.ctx.capacity_waited_sec += sleep_for
                queue = still_waiting
                continue

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
                    pick_sc = getattr(scheduled, "_last_pick_score", None)
                    runner = getattr(self.providers.get(name), "_runner", None)
                    self.meta.provider_picks.append(
                        self._pick_meta(
                            scheduled,
                            name,
                            ok=outcome.ok,
                            status=outcome.status_code,
                            score=pick_sc,
                            max_tokens=getattr(runner, "_last_max_tokens", None),
                            latency_ms=outcome.latency_ms,
                            retry_after=outcome.retry_after,
                            error=outcome.error,
                        )
                    )

                    if outcome.ok and outcome.result:
                        self._record_success(name, outcome.latency_ms)
                        self._record_history(
                            name,
                            scheduled,
                            ok=True,
                            latency_ms=outcome.latency_ms,
                            result=outcome.result,
                        )
                        self.ctx.quota.record(name)
                        self._store_cache(scheduled, outcome.result)

                        final = outcome.result
                        if self._needs_escalate(scheduled, name, final):
                            reason = self._escalate_reason(scheduled, final)
                            pending_escalates.append((scheduled, final, reason))
                            logger.info(
                                f"Queued escalate ticket reason={reason} "
                                f"(defer multiplex; tokens={scheduled.chunk.estimated_tokens})"
                            )
                        else:
                            results.append(final)
                    else:
                        self._record_failure(name, outcome, scheduled)
                        self._record_history(
                            name,
                            scheduled,
                            ok=False,
                            latency_ms=outcome.latency_ms,
                        )
                        logger.info(
                            f"Provider attempt failed provider={name} "
                            f"status={outcome.status_code} "
                            f"error={outcome.error!r} "
                            f"retry_after={outcome.retry_after} "
                            f"latency_ms={outcome.latency_ms:.0f} "
                            f"capacity={outcome.is_capacity_failure}"
                        )
                        if self._has_untried_eligible(scheduled):
                            still_waiting.append(scheduled)
                        elif (
                            outcome.is_capacity_failure
                            and scheduled.rate_limit_rounds < MAX_RATE_LIMIT_ROUNDS
                            and self.ctx.remaining_sec() > 20
                            and self.ctx.capacity_budget_remaining() > 15
                        ):
                            scheduled.rate_limit_rounds += 1
                            scheduled.attempted_providers.clear()
                            logger.info(
                                f"Chunk capacity-exhausted on tried providers; "
                                f"requeue round {scheduled.rate_limit_rounds}/"
                                f"{MAX_RATE_LIMIT_ROUNDS} "
                                f"(last_error={outcome.error or outcome.status_code})"
                            )
                            still_waiting.append(scheduled)
                        else:
                            err = outcome.error or f"failed on {name}"
                            if outcome.is_capacity_failure and not err.startswith(
                                "capacity:"
                            ):
                                err = f"capacity:{err}"
                            skipped.append(SkipRecord(scheduled.chunk, err))

            still_waiting.sort(key=sort_key)
            queue = still_waiting

        if pending_escalates:
            results.extend(self._flush_pending_escalates(pending_escalates))

        if self.ctx.semaphore is not None:
            self.ctx.semaphore.release_all()

        if queue and not self.ctx.alive():
            self.meta.stop_reason = "deadline"
            for s in queue:
                skipped.append(
                    SkipRecord(
                        s.chunk,
                        f"wall-clock deadline (priority={s.priority} not reached)",
                    )
                )
        elif queue and not skipped:
            for s in queue:
                skipped.append(SkipRecord(s.chunk, f"incomplete (priority={s.priority})"))

        self.meta.skipped = len(skipped)
        self.meta.health_snapshot = self._blended_health_snapshot()
        if isinstance(self.ctx.history, ProviderHistory):
            self.ctx.history.save()
        return results, skipped, self.meta

    def _pick_meta(
        self,
        scheduled: ScheduledChunk,
        provider: str,
        *,
        ok: bool,
        status: int | None,
        score: float | None,
        max_tokens: int | None = None,
        latency_ms: float | None = None,
        retry_after: float | None = None,
        error: str | None = None,
    ) -> dict:
        tokens = effective_tokens(scheduled, self.ctx.prompt_overhead_tokens)
        reason = pick_reason(
            provider,
            tokens,
            complexity=scheduled.complexity,
            path_priority=scheduled.priority,
        )
        meta: dict = {
            "provider": provider,
            "tokens": scheduled.chunk.estimated_tokens,
            "effective_tokens": tokens,
            "priority": scheduled.priority,
            "complexity": round(scheduled.complexity, 3),
            "reason": reason,
            "ok": ok,
            "status": status,
        }
        if score is not None:
            meta["pick_score"] = round(score, 4)
        if max_tokens is not None:
            meta["max_tokens"] = max_tokens
        if latency_ms is not None:
            meta["latency_ms"] = round(latency_ms, 1)
        if retry_after is not None:
            meta["retry_after"] = retry_after
        if error:
            meta["error"] = error
        return meta

    def _escalate_reason(self, scheduled: ScheduledChunk, result: ReviewResult) -> str:
        if is_security_sensitive(scheduled.chunk):
            return "security"
        if is_high_complexity(scheduled.complexity):
            return "complexity"
        if self._is_vacuous_review(scheduled, result):
            return "vacuous"
        if result.score < ESCALATE_SCORE_THRESHOLD:
            return "low_score"
        if scheduled.chunk.estimated_tokens >= 2500 and not result.issues:
            return "empty_large"
        return "escalate"

    def _ticket_from_pending(
        self,
        scheduled: ScheduledChunk,
        groq_result: ReviewResult,
        reason: str,
    ) -> EscalateTicket:
        issues = [
            {
                "severity": i.severity,
                "file": i.file,
                "line": i.line,
                "message": i.message,
                "rule": i.rule,
                "suggestion": i.suggestion,
            }
            for i in (groq_result.issues or [])
        ]
        return EscalateTicket(
            ticket_id=new_ticket_id(),
            reason=reason,
            files=list(scheduled.chunk.files or []),
            estimated_tokens=scheduled.chunk.estimated_tokens,
            complexity=scheduled.complexity,
            priority=scheduled.priority,
            groq_summary=groq_result.summary or "",
            groq_score=float(groq_result.score),
            groq_issues=issues,
            contested_diff=truncate_diff(scheduled.chunk.parsed_diff.raw or ""),
            repo=self.ctx.repo,
            pr_number=self.ctx.pr_number,
            installation_id=self.ctx.installation_id,
            head_sha=self.ctx.head_sha,
        )

    def _flush_pending_escalates(
        self,
        pending: list[tuple[ScheduledChunk, ReviewResult, str]],
    ) -> list[ReviewResult]:
        """One Gemini multiplex call for all pending tickets; else ledger + soften."""
        if not pending:
            return []

        tickets = [
            self._ticket_from_pending(sched, groq, reason)
            for sched, groq, reason in pending
        ]
        self.meta.escalate_multiplex_tickets = len(tickets)

        if not self._can_escalate_now():
            waited = self._wait_for_gemini_escalate(max_wait=45.0)
            if not waited:
                return self._defer_or_soften(pending, tickets)

        upgraded = self._multiplex_escalate(tickets)
        if not upgraded:
            return self._defer_or_soften(pending, tickets)

        out: list[ReviewResult] = []
        for (scheduled, groq, _reason), ticket in zip(pending, tickets):
            result = upgraded.get(ticket.ticket_id)
            if result is not None:
                scheduled.escalated = True
                self.meta.escalations += 1
                self._store_cache(scheduled, result)
                out.append(result)
            else:
                # Partial multiplex miss → soften that ticket's Groq triage
                out.append(self._soften_incomplete_triage(groq))
                self.meta.escalation_blocked += 1
        meta = {
            "provider": "gemini",
            "escalation": True,
            "multiplex": True,
            "tickets": len(tickets),
            "resolved": len(upgraded),
            "ok": True,
            "status": 200,
        }
        self.meta.provider_picks.append(meta)
        return out

    def _defer_or_soften(
        self,
        pending: list[tuple[ScheduledChunk, ReviewResult, str]],
        tickets: list[EscalateTicket],
    ) -> list[ReviewResult]:
        """Enqueue for drain when possible; always return softened provisionals."""
        ledger = EscalateLedger()
        try:
            if self.ctx.repo and self.ctx.pr_number:
                ledger.cancel_pr(self.ctx.repo, self.ctx.pr_number)
            n = ledger.enqueue(tickets)
            self.meta.escalate_ledger_enqueued = n
            logger.info(f"Escalate deferred to ledger ({n} ticket(s))")
        except Exception as e:
            logger.warning(f"Escalate ledger enqueue failed: {e}")
        out = []
        for _sched, groq, _reason in pending:
            out.append(self._soften_incomplete_triage(groq))
            self.meta.escalation_blocked += 1
        return out

    def _multiplex_escalate(self, tickets: list[EscalateTicket]) -> dict[str, ReviewResult]:
        if "gemini" not in self.providers:
            return {}
        if not self.ctx.try_acquire("gemini"):
            return {}
        try:
            provider = self.providers["gemini"]
            runner = getattr(provider, "_runner", None)
            if runner is None or not hasattr(runner, "review_escalate_batch"):
                # Fallback: single-ticket full escalate via first chunk only
                logger.warning("Gemini runner missing review_escalate_batch; skip multiplex")
                return {}
            logger.info(f"Gemini multiplex escalate for {len(tickets)} ticket(s)")
            upgraded = runner.review_escalate_batch(tickets)
            self.meta.dispatches += 1
            if upgraded:
                # Approximate latency from first result
                sample = next(iter(upgraded.values()))
                self._record_success("gemini", sample.latency_ms)
                self.ctx.quota.record("gemini")
                if isinstance(self.ctx.history, ProviderHistory):
                    # Record one success against a synthetic small chunk key
                    pass
            return upgraded
        finally:
            self.ctx.release("gemini")

    @staticmethod
    def _is_vacuous_review(scheduled: ScheduledChunk, result: ReviewResult) -> bool:
        """High score + zero issues on a non-trivial diff is usually under-review."""
        if result.issues:
            return False
        if result.score < 8.5:
            return False
        return scheduled.chunk.estimated_tokens >= 1500

    def _needs_escalate(
        self,
        scheduled: ScheduledChunk,
        provider: str,
        result: ReviewResult,
    ) -> bool:
        """Whether deeper review is warranted (independent of Gemini availability)."""
        if provider != "groq":
            return False
        if "gemini" not in self.providers:
            return False
        if is_security_sensitive(scheduled.chunk):
            return True
        if is_high_complexity(scheduled.complexity):
            return True
        if result.score < ESCALATE_SCORE_THRESHOLD:
            return True
        if ReviewScheduler._is_vacuous_review(scheduled, result):
            return True
        if scheduled.chunk.estimated_tokens >= 2500 and not result.issues:
            return True
        return False

    def _can_escalate_now(self) -> bool:
        if "gemini" not in self.providers:
            return False
        if not self.ctx.quota.can_use("gemini"):
            return False
        if isinstance(self.ctx.history, ProviderHistory) and self.ctx.history.is_cooling("gemini"):
            return False
        rt = self.ctx.providers.get("gemini")
        if not rt or rt.health.is_cooling():
            return False
        if rt.health.score < self.ctx.health_threshold:
            return False
        return True

    @staticmethod
    def _soften_incomplete_triage(result: ReviewResult) -> ReviewResult:
        """Avoid publishing a fake production-ready score when escalate was needed but blocked."""
        result.score = min(float(result.score), 7.0)
        warning = "escalation_unavailable_vacuous_triage"
        if result.parse_warning:
            result.parse_warning = f"{result.parse_warning};{warning}"
        else:
            result.parse_warning = warning
        note = (
            "Deep review (Gemini) was deferred — provisional Groq triage only; "
            "re-run `/sorge` or wait for escalate drain when free-tier recovers."
        )
        recs = list(result.recommendations or [])
        if note not in recs:
            recs.insert(0, note)
        result.recommendations = recs
        if not (result.summary or "").strip():
            result.summary = "Partial triage only — escalate unavailable."
        elif "Partial triage" not in result.summary:
            result.summary = f"{result.summary.rstrip()} (partial triage; escalate unavailable)"
        return result

    def _should_escalate(
        self,
        scheduled: ScheduledChunk,
        provider: str,
        result: ReviewResult,
    ) -> bool:
        return self._needs_escalate(scheduled, provider, result) and self._can_escalate_now()

    def _wait_for_gemini_escalate(self, *, max_wait: float = 45.0) -> bool:
        """Block briefly for Gemini cool/health if escalate is needed."""
        if "gemini" not in self.ctx.providers:
            return False
        if not self.ctx.quota.can_use("gemini"):
            return False
        waits: list[float] = []
        rt = self.ctx.providers["gemini"]
        cool = rt.health.cooling_remaining()
        if cool > 0:
            waits.append(cool)
        if isinstance(self.ctx.history, ProviderHistory):
            hist = self.ctx.history.cooling_remaining("gemini")
            if hist > 0:
                waits.append(hist)
        if not waits:
            return self._can_escalate_now()
        wait = min(waits)
        if wait <= 0:
            return self._can_escalate_now()
        if wait > max_wait or wait > self.ctx.remaining_sec() - 5:
            return False
        logger.info(f"Waiting {wait:.1f}s for Gemini before escalate")
        time.sleep(wait)
        return self._can_escalate_now()

    def _escalate_to_gemini(self, scheduled: ScheduledChunk) -> ReviewResult | None:
        if "gemini" not in self.providers:
            return None
        if not self.ctx.try_acquire("gemini"):
            return None
        try:
            logger.info(
                f"Escalating chunk to gemini "
                f"(complexity={scheduled.complexity:.2f}, priority={scheduled.priority})"
            )
            outcome = self._dispatch(scheduled, "gemini")
            self.meta.dispatches += 1
            meta = self._pick_meta(
                scheduled,
                "gemini",
                ok=outcome.ok,
                status=outcome.status_code,
                score=None,
            )
            meta["escalation"] = True
            self.meta.provider_picks.append(meta)
            if outcome.ok and outcome.result:
                self._record_success("gemini", outcome.latency_ms)
                self._record_history(
                    "gemini",
                    scheduled,
                    ok=True,
                    latency_ms=outcome.latency_ms,
                    result=outcome.result,
                )
                self.ctx.quota.record("gemini")
                self._store_cache(scheduled, outcome.result)
                return outcome.result
            self._record_failure("gemini", outcome, scheduled)
            self._record_history("gemini", scheduled, ok=False, latency_ms=outcome.latency_ms)
            return None
        finally:
            self.ctx.release("gemini")

    def _try_cache_hit(self, scheduled: ScheduledChunk) -> ReviewResult | None:
        if not self.ctx.cache_enabled:
            return None
        raw = scheduled.chunk.parsed_diff.raw
        cached = review_cache.get_chunk(
            raw,
            self.ctx.cache_ttl_hours,
            context_fingerprint=self.ctx.context_fingerprint,
        )
        if cached is None or cached.get("parse_warning"):
            return None
        logger.info(
            f"Scheduler cache hit for chunk "
            f"(priority={scheduled.priority}, tokens={scheduled.chunk.estimated_tokens})"
        )
        return _result_from_cache_dict(cached)

    def _store_cache(self, scheduled: ScheduledChunk, result: ReviewResult) -> None:
        if not self.ctx.cache_enabled or result.parse_warning:
            return
        review_cache.set_chunk(
            scheduled.chunk.parsed_diff.raw,
            result.to_dict(),
            context_fingerprint=self.ctx.context_fingerprint,
        )

    def _desired_workers(self) -> int:
        slots = 0
        for name, rt in self.ctx.providers.items():
            if rt.health.is_cooling():
                continue
            if rt.health.score < self.ctx.health_threshold:
                continue
            if rt.bucket.remaining() < 1.0:
                continue
            if not self.ctx.quota.can_use(name):
                continue
            free = max(0, rt.max_inflight - rt.in_flight)
            slots += free
        if slots <= 0:
            return 1
        return max(1, min(self.max_workers, slots))

    def _pick_provider(self, scheduled: ScheduledChunk) -> str | None:
        name, _ = self._pick_provider_scored(scheduled)
        if name is None:
            name, _ = self._pick_provider_scored(
                scheduled, respect_soft_reserve=False
            )
        return name

    def _pick_provider_scored(
        self,
        scheduled: ScheduledChunk,
        *,
        respect_soft_reserve: bool = True,
    ) -> tuple[str | None, float]:
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
            if not self.ctx.quota.can_use(
                name, respect_soft_reserve=respect_soft_reserve
            ):
                continue
            if isinstance(self.ctx.history, ProviderHistory) and self.ctx.history.is_cooling(name):
                continue
            hist_q = status.quality_prior
            if isinstance(self.ctx.history, ProviderHistory):
                hist_q = self.ctx.history.quality(
                    name,
                    scheduled.chunk,
                    default=status.quality_prior,
                )
            key = self.ctx.quota.PROVIDER_KEYS.get(name, name)
            limit = max(1, self.ctx.quota.limits.get(key, 1))
            quota_frac = self.ctx.quota.remaining(name) / limit
            blended_hist = 0.85 * hist_q + 0.15 * max(0.0, min(1.0, quota_frac))
            sc = score_provider(
                status,
                scheduled,
                historical_quality=blended_hist,
                prompt_overhead_tokens=self.ctx.prompt_overhead_tokens,
            )
            if sc > best_score:
                best_score = sc
                best_name = name
        return (best_name, best_score) if best_score > 0 else (None, 0.0)

    def _has_untried_eligible(self, scheduled: ScheduledChunk) -> bool:
        probe = ScheduledChunk(
            chunk=scheduled.chunk,
            priority=scheduled.priority,
            complexity=scheduled.complexity,
            attempted_providers=set(scheduled.attempted_providers),
        )
        return self._pick_provider(probe) is not None

    def _min_wake_sec(self) -> float | None:
        waits: list[float] = []
        threshold = self.ctx.health_threshold
        for name, rt in self.ctx.providers.items():
            cool = rt.health.cooling_remaining()
            if cool > 0:
                waits.append(cool)
            if rt.health.score < threshold:
                recover = rt.health.seconds_until_score(threshold)
                if recover > 0:
                    waits.append(recover)
            if rt.bucket.remaining() < 1.0:
                refill = rt.bucket.time_until(1.0)
                if refill != float("inf"):
                    waits.append(refill)
            if isinstance(self.ctx.history, ProviderHistory):
                hist_cool = self.ctx.history.cooling_remaining(name)
                if hist_cool > 0:
                    waits.append(hist_cool)
        return min(waits) if waits else None

    def _dispatch(self, scheduled: ScheduledChunk, name: str) -> ProviderResult:
        provider = self.providers[name]
        eff = effective_tokens(scheduled, self.ctx.prompt_overhead_tokens)
        logger.info(
            f"Scheduler → {name} for chunk "
            f"(priority={scheduled.priority}, complexity={scheduled.complexity:.2f}, "
            f"tokens={scheduled.chunk.estimated_tokens}, effective={eff})"
        )
        return provider.review(scheduled.chunk, self.ctx)

    def _record_success(self, name: str, latency_ms: float) -> None:
        rt = self.ctx.providers[name]
        rt.health.record_success(latency_ms)

    def _record_history(
        self,
        name: str,
        scheduled: ScheduledChunk,
        *,
        ok: bool,
        latency_ms: float,
        result: ReviewResult | None = None,
    ) -> None:
        if isinstance(self.ctx.history, ProviderHistory):
            self.ctx.history.record(
                name,
                scheduled.chunk,
                ok=ok,
                latency_ms=latency_ms,
                review_score=result.score if result and ok else None,
                issues_found=len(result.issues) if result and ok else None,
            )

    def _blended_health_snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, rt in self.ctx.providers.items():
            success_n = max(0.0, min(1.0, rt.health.score / 100.0))
            lat_n = max(0.0, min(1.0, 1.0 - (rt.nominal_latency_ms / 2500.0)))
            key = self.ctx.quota.PROVIDER_KEYS.get(name, name)
            limit = max(1, self.ctx.quota.limits.get(key, 1))
            rem = self.ctx.quota.remaining(name)
            quota_n = rem / limit
            fail_pen = 0.5 if rt.health.is_cooling() else 0.0
            blended = (
                0.40 * success_n
                + 0.30 * lat_n
                + 0.20 * quota_n
                + 0.10 * (1.0 - fail_pen)
            ) * 100.0
            out[name] = round(blended, 1)
        return out

    def _record_failure(
        self,
        name: str,
        outcome: ProviderResult,
        scheduled: ScheduledChunk | None = None,
    ) -> None:
        rt = self.ctx.providers[name]
        if outcome.timed_out:
            rt.health.record_timeout()
        elif outcome.is_rate_limited:
            rt.health.record_rate_limit(outcome.retry_after)
            self.ctx.quota.record_failure(name)
            if isinstance(self.ctx.history, ProviderHistory):
                self.ctx.history.mark_rate_limited(name, retry_after=outcome.retry_after)
        elif outcome.is_capacity_failure and not outcome.is_payload_too_large:
            # Truncated/empty/5xx: soft-cool so concurrent jobs skip this provider briefly.
            cool = outcome.retry_after if outcome.retry_after is not None else 45.0
            rt.health.record_rate_limit(cool)
            if isinstance(self.ctx.history, ProviderHistory):
                self.ctx.history.mark_rate_limited(name, retry_after=cool)
            if outcome.status_code and outcome.status_code >= 500:
                rt.health.record_server_error()
        elif outcome.is_payload_too_large:
            rt.health.record_payload_too_large()
            if scheduled is not None:
                needed = scheduled.chunk.estimated_tokens + self.ctx.prompt_overhead_tokens
                new_cap = max(1, needed - 1)
                if new_cap < rt.max_context_tokens:
                    logger.info(
                        f"Provider {name}: 413 → max_context_tokens "
                        f"{rt.max_context_tokens} → {new_cap}"
                    )
                    rt.max_context_tokens = new_cap
        elif outcome.status_code and outcome.status_code >= 500:
            rt.health.record_server_error()
        else:
            rt.health.record_server_error()


def _result_from_cache_dict(data: dict) -> ReviewResult:
    issues = issues_from_parsed({"issues": data.get("issues", [])})
    return ReviewResult(
        summary=data.get("summary", ""),
        issues=issues,
        recommendations=data.get("recommendations", []),
        score=data.get("score", 7.0),
        latency_ms=data.get("latency_ms", 0.0),
        model=data.get("model", "cache"),
        tokens_used=data.get("tokens_used"),
        review_type=data.get("review_type", "cache"),
        parse_warning=data.get("parse_warning"),
    )
