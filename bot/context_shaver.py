"""Context Shaving — virtual context pool when Gemini is fully dead.

N cheap extract/Layer-0 shavings + joint review(s) with the shaving pool.
Strict fallback only: never engages while Gemini can still take the call.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

from bot.diff_parser import ParsedDiff, estimate_tokens
from bot.file_splitter import FileSplitter, ReviewChunk
from bot.providers.base import Provider
from bot.quota_tracker import QuotaTracker
from bot.review_aggregator import ReviewAggregator
from bot.runners.base import ReviewResult
from bot.scheduling.market_score import LANE_GROQ_MAX, LANE_OPENROUTER_MAX, TEMPLATE_OVERHEAD_TOKENS
from bot.scheduling.scheduler import ReviewScheduler
from bot.scheduling.types import SchedulerMeta
from bot.utils import shaving_cache

_EXPORT_RE = re.compile(
    r"^\+\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|fn|func|const|let|var|type|interface|struct|enum)\s+(\w+)",
    re.M,
)
_IMPORT_RE = re.compile(
    r"^\+\s*(?:import|from|require|use|using|#include)\b[^\n]*",
    re.M,
)
_SIDE_EFFECT_RE = re.compile(
    r"(?i)\b(fetch|axios|exec|subprocess|eval|sql|query|password|secret|token|auth|"
    r"permission|rmtree|unlink|writeFile|mkdir|redis|s3|http\.|requests\.|open\()\b"
)

EXTRACT_PROMPT = """Compress this code diff slice into dense JSON only (no markdown).
Schema:
{{"exports":[str],"imports":[str],"side_effects":[str],"fingerprints":[str],"notes":str}}
Rules: max 12 exports, 12 imports, 8 side_effects, 8 fingerprints. Be terse.

DIFF SLICE:
```diff
{diff}
```
"""


@dataclass
class Shaving:
    slice_id: str
    files: list[str]
    exports: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = "heuristic"  # heuristic | ast | llm | cache
    content_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shaving":
        return cls(
            slice_id=str(data.get("slice_id") or ""),
            files=list(data.get("files") or []),
            exports=list(data.get("exports") or []),
            imports=list(data.get("imports") or []),
            side_effects=list(data.get("side_effects") or []),
            fingerprints=list(data.get("fingerprints") or []),
            notes=str(data.get("notes") or ""),
            source=str(data.get("source") or "cache"),
            content_sha=str(data.get("content_sha") or ""),
        )


@dataclass
class ContextShaveMeta:
    enabled: bool = False
    engaged: bool = False
    reason: str = ""
    slices: int = 0
    extracts_llm: int = 0
    cache_hits: int = 0
    synthesize_mode: str = ""  # single | per_slice | skipped
    synthesize_provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gemini_fully_dead(
    quota: QuotaTracker,
    providers: list[Provider],
    *,
    history=None,
) -> bool:
    """True when Gemini cannot be used for this run at all."""
    names = {p.name for p in providers}
    if "gemini" not in names:
        return True
    if not quota.can_use("gemini", respect_soft_reserve=False):
        return True
    if history is not None and getattr(history, "is_cooling", lambda _n: False)("gemini"):
        return True
    return False


def max_remaining_fit_tokens(
    quota: QuotaTracker,
    providers: list[Provider],
    *,
    prompt_overhead: int,
) -> int:
    """Largest effective window among usable non-Gemini providers."""
    by_name = {p.name: p for p in providers}
    best = 0
    if "groq" in by_name and quota.can_use("groq"):
        best = max(best, LANE_GROQ_MAX)
    if "openrouter" in by_name and quota.can_use("openrouter"):
        best = max(best, LANE_OPENROUTER_MAX)
    # Net tokens available for diff after overhead
    return max(0, best - max(0, prompt_overhead))


def needs_oversize_fallback(
    chunks: list[ReviewChunk],
    *,
    fit_diff_tokens: int,
) -> bool:
    """True when at least one runnable chunk cannot fit remaining secondary lanes."""
    if fit_diff_tokens <= 0:
        return True
    for chunk in chunks:
        if chunk.unreviewable:
            continue
        if chunk.estimated_tokens > fit_diff_tokens:
            return True
    return False


def should_engage_context_shave(
    *,
    enabled: bool,
    quota: QuotaTracker,
    providers: list[Provider],
    chunks: list[ReviewChunk],
    prompt_overhead: int,
    history=None,
) -> tuple[bool, str]:
    """Strict Gemini-dead + oversize gate. Happy path always returns False."""
    if not enabled:
        return False, "flag_off"
    if not gemini_fully_dead(quota, providers, history=history):
        return False, "gemini_usable"
    fit = max_remaining_fit_tokens(quota, providers, prompt_overhead=prompt_overhead)
    if not needs_oversize_fallback(chunks, fit_diff_tokens=fit):
        return False, "fits_secondary"
    if fit <= 0 and not any(p.name in ("groq", "openrouter") for p in providers):
        return False, "no_secondary_provider"
    return True, "gemini_dead_oversize"


def content_sha(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def layer0_shaving(chunk: ReviewChunk, *, slice_id: str) -> Shaving:
    """Deterministic / free shaving from diff text (no LLM)."""
    raw = chunk.parsed_diff.raw or ""
    sha = content_sha(raw)
    exports = list(dict.fromkeys(_EXPORT_RE.findall(raw)))[:12]
    imports = [m.strip()[:120] for m in _IMPORT_RE.findall(raw)][:12]
    effects = list(dict.fromkeys(m.lower() for m in _SIDE_EFFECT_RE.findall(raw)))[:8]
    fingerprints = [
        f"files={len(chunk.files)}",
        f"tokens≈{chunk.estimated_tokens}",
        f"+{chunk.parsed_diff.lines_added}/-{chunk.parsed_diff.lines_deleted}",
    ]
    thin = len(exports) + len(imports) + len(effects) < 2
    return Shaving(
        slice_id=slice_id,
        files=list(chunk.files or []),
        exports=exports,
        imports=imports,
        side_effects=effects,
        fingerprints=fingerprints,
        notes="thin_slice" if thin else "",
        source="heuristic",
        content_sha=sha,
    )


def shaving_is_thin(shaving: Shaving) -> bool:
    return (
        shaving.notes == "thin_slice"
        or (len(shaving.exports) + len(shaving.imports) + len(shaving.side_effects) < 2)
    )


def pool_shavings(shavings: list[Shaving], *, max_chars: int = 6000) -> str:
    """Compact joint architectural map for synthesize / per-slice context."""
    parts = [
        "CONTEXT_SHAVINGS (virtual context pool — treat as ground truth for "
        "surroundings; review the DIFF; do not invent APIs absent here or in the diff):"
    ]
    for s in shavings:
        block = {
            "id": s.slice_id,
            "files": s.files[:8],
            "exports": s.exports[:12],
            "imports": s.imports[:8],
            "side_effects": s.side_effects[:8],
            "fingerprints": s.fingerprints[:6],
            "source": s.source,
        }
        if s.notes:
            block["notes"] = s.notes[:80]
        line = json.dumps(block, separators=(",", ":"))
        trial = "\n".join(parts + [line])
        if len(trial) > max_chars and len(parts) > 1:
            parts.append(f"... truncated {len(shavings) - (len(parts) - 1)} shavings")
            break
        parts.append(line)
    return "\n".join(parts)[:max_chars]


def _parse_extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_extract_via_provider(provider: Provider, chunk: ReviewChunk) -> dict[str, Any] | None:
    """Best-effort LLM extract using the provider runner's review path on a tiny prompt diff."""
    runner = getattr(provider, "_runner", None)
    if runner is None:
        return None
    prompt = EXTRACT_PROMPT.format(diff=(chunk.parsed_diff.raw or "")[:12000])
    # Reuse review pipeline by wrapping the extract prompt as a pseudo-diff body.
    fake = ParsedDiff(
        raw=prompt,
        files=list(chunk.files or ["extract"]),
        lines_added=max(1, (chunk.parsed_diff.lines_added or 1)),
        lines_deleted=0,
        files_changed=1,
    )
    # Prefer a direct complete hook if present.
    complete = getattr(runner, "complete_user_prompt", None)
    if callable(complete):
        try:
            raw = complete(prompt, max_tokens=512)
            return _parse_extract_json(raw or "")
        except Exception as exc:
            logger.debug(f"LLM extract complete_user_prompt failed: {exc}")
            return None
    try:
        result = runner.review(fake, repo_context=None, context_fingerprint="shave-extract")
        if result is None:
            return None
        # If the model returned structured review, fold summary into notes.
        return {
            "exports": [],
            "imports": [],
            "side_effects": [],
            "fingerprints": [],
            "notes": (result.summary or "")[:200],
        }
    except Exception as exc:
        logger.debug(f"LLM extract via review failed: {exc}")
        return None


def enrich_shaving_with_llm(
    shaving: Shaving,
    chunk: ReviewChunk,
    provider: Provider | None,
) -> Shaving:
    if provider is None:
        return shaving
    data = _llm_extract_via_provider(provider, chunk)
    if not data:
        return shaving
    exports = list(dict.fromkeys([*shaving.exports, *(data.get("exports") or [])]))[:12]
    imports = list(dict.fromkeys([*shaving.imports, *(data.get("imports") or [])]))[:12]
    effects = list(
        dict.fromkeys([*shaving.side_effects, *(data.get("side_effects") or [])])
    )[:8]
    fingerprints = list(
        dict.fromkeys([*shaving.fingerprints, *(data.get("fingerprints") or [])])
    )[:8]
    notes = str(data.get("notes") or shaving.notes)[:120]
    return Shaving(
        slice_id=shaving.slice_id,
        files=shaving.files,
        exports=exports,
        imports=imports,
        side_effects=effects,
        fingerprints=fingerprints,
        notes=notes,
        source="llm",
        content_sha=shaving.content_sha,
    )


class ContextShaver:
    """Build a virtual context pool and review when Gemini is fully dead."""

    def __init__(
        self,
        *,
        max_extracts: int = 12,
        shave_slice_budget: int = 3500,
        cache_ttl_hours: int = 168,
    ):
        self.max_extracts = max(0, max_extracts)
        self.shave_slice_budget = max(500, shave_slice_budget)
        self.cache_ttl_hours = cache_ttl_hours

    def slice_for_secondary(self, parsed_diff: ParsedDiff) -> list[ReviewChunk]:
        splitter = FileSplitter(
            chunk_budget=self.shave_slice_budget,
            max_chunk_tokens=self.shave_slice_budget,
        )
        return splitter.split(parsed_diff)

    def build_shavings(
        self,
        slices: list[ReviewChunk],
        *,
        extract_provider: Provider | None,
        use_cache: bool = True,
    ) -> tuple[list[Shaving], int, int]:
        shavings: list[Shaving] = []
        cache_hits = 0
        llm_extracts = 0
        for i, chunk in enumerate(slices):
            if chunk.unreviewable:
                continue
            sid = chunk.part_label or f"slice-{i+1}"
            base = layer0_shaving(chunk, slice_id=sid)
            if use_cache:
                cached = shaving_cache.get(base.content_sha, self.cache_ttl_hours)
                if cached:
                    s = Shaving.from_dict({**cached, "content_sha": base.content_sha})
                    s.source = "cache"
                    shavings.append(s)
                    cache_hits += 1
                    continue
            s = base
            if (
                extract_provider is not None
                and shaving_is_thin(s)
                and llm_extracts < self.max_extracts
            ):
                s = enrich_shaving_with_llm(s, chunk, extract_provider)
                llm_extracts += 1
            if use_cache:
                shaving_cache.set(s.content_sha, s.to_dict())
            shavings.append(s)
        return shavings, cache_hits, llm_extracts

    def run(
        self,
        parsed_diff: ParsedDiff,
        *,
        config,
        quota: QuotaTracker,
        providers: list[Provider],
        repo_context: str | None,
        context_fingerprint: str,
        prompt_overhead: int,
        repo: str = "",
        pr_number: int = 0,
        installation_id: int | None = None,
        head_sha: str = "",
    ) -> tuple[ReviewResult | None, ContextShaveMeta]:
        meta = ContextShaveMeta(enabled=True, engaged=True, reason="gemini_dead_oversize")
        try:
            return self._run_inner(
                parsed_diff,
                config=config,
                quota=quota,
                providers=providers,
                repo_context=repo_context,
                context_fingerprint=context_fingerprint,
                prompt_overhead=prompt_overhead,
                repo=repo,
                pr_number=pr_number,
                installation_id=installation_id,
                head_sha=head_sha,
                meta=meta,
            )
        except Exception as exc:
            logger.warning(f"Context shave failed open: {exc}")
            meta.engaged = False
            meta.reason = f"fail_open:{exc}"
            meta.synthesize_mode = "skipped"
            return None, meta

    def _run_inner(
        self,
        parsed_diff: ParsedDiff,
        *,
        config,
        quota: QuotaTracker,
        providers: list[Provider],
        repo_context: str | None,
        context_fingerprint: str,
        prompt_overhead: int,
        repo: str,
        pr_number: int,
        installation_id: int | None,
        head_sha: str,
        meta: ContextShaveMeta,
    ) -> tuple[ReviewResult | None, ContextShaveMeta]:
        slices = [c for c in self.slice_for_secondary(parsed_diff) if not c.unreviewable]
        if not slices:
            meta.reason = "no_slices"
            meta.synthesize_mode = "skipped"
            return None, meta
        meta.slices = len(slices)

        by_name = {p.name: p for p in providers}
        extract_provider = None
        if "groq" in by_name and quota.can_use("groq"):
            extract_provider = by_name["groq"]
        elif "openrouter" in by_name and quota.can_use("openrouter"):
            extract_provider = by_name["openrouter"]

        shavings, cache_hits, llm_n = self.build_shavings(
            slices, extract_provider=extract_provider, use_cache=True
        )
        meta.cache_hits = cache_hits
        meta.extracts_llm = llm_n
        for _ in range(llm_n):
            if extract_provider:
                quota.record(extract_provider.name)

        pool = pool_shavings(shavings)
        combined_context = pool
        if repo_context:
            combined_context = f"{pool}\n\n---\n\n{repo_context}"[:8000]

        total_tokens = estimate_tokens(parsed_diff.raw or "")
        fit = max_remaining_fit_tokens(quota, providers, prompt_overhead=prompt_overhead)
        secondary = [p for p in providers if p.name != "gemini"]
        if not secondary:
            meta.synthesize_mode = "skipped"
            meta.reason = "no_secondary"
            return None, meta

        # Prefer one joint synthesize when the full diff fits a secondary lane.
        if total_tokens <= fit:
            meta.synthesize_mode = "single"
            full_chunk = ReviewChunk(
                files=list(parsed_diff.files),
                parsed_diff=parsed_diff,
                estimated_tokens=total_tokens,
            )
            scheduler = ReviewScheduler.from_providers(
                secondary,
                quota,
                config,
                repo_context=combined_context,
                context_fingerprint=f"{context_fingerprint}:shave",
                prompt_overhead_tokens=prompt_overhead,
                repo=repo,
                pr_number=pr_number,
                installation_id=installation_id,
                head_sha=head_sha,
            )
            results, skipped, sched_meta = scheduler.run([full_chunk])
            meta.synthesize_provider = _first_ok_provider(sched_meta)
            merged = ReviewAggregator.merge(
                results,
                rung="context_shave",
                quota_snapshot=quota.snapshot(),
                skipped=skipped or None,
                scheduler_meta=sched_meta,
            )
            _attach_shave_meta(merged, meta)
            return merged, meta

        # Else: per-slice reviews each see the joint shaving pool (virtual context).
        meta.synthesize_mode = "per_slice"
        scheduler = ReviewScheduler.from_providers(
            secondary,
            quota,
            config,
            repo_context=combined_context,
            context_fingerprint=f"{context_fingerprint}:shave",
            prompt_overhead_tokens=prompt_overhead,
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
            head_sha=head_sha,
        )
        results, skipped, sched_meta = scheduler.run(slices)
        meta.synthesize_provider = _first_ok_provider(sched_meta)
        merged = ReviewAggregator.merge(
            results,
            rung="context_shave",
            quota_snapshot=quota.snapshot(),
            skipped=skipped or None,
            scheduler_meta=sched_meta,
        )
        _attach_shave_meta(merged, meta)
        return merged, meta


def _first_ok_provider(sched_meta: SchedulerMeta | None) -> str:
    if sched_meta is None:
        return ""
    for pick in getattr(sched_meta, "provider_picks", []) or []:
        if isinstance(pick, dict) and pick.get("ok"):
            return str(pick.get("provider") or "")
    return ""


def _attach_shave_meta(result: ReviewResult, meta: ContextShaveMeta) -> None:
    rm = dict(result.routing_meta or {})
    rm["context_shave"] = meta.to_dict()
    result.routing_meta = rm
