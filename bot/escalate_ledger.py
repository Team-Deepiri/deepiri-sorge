"""Durable escalate ticket ledger (Worker KV HTTP or local file fallback)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from bot.scheduling.escalate import EscalateTicket

DEFAULT_FILE = Path.home() / ".cache" / "sorge" / "escalate_ledger.json"


class EscalateLedger:
    """Enqueue / claim / ack escalate tickets for Gemini multiplex drain."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        secret: str | None = None,
        file_path: Path | None = None,
    ):
        self.base_url = (base_url or os.getenv("SORGE_LEDGER_URL") or "").rstrip("/")
        self.secret = secret or os.getenv("SORGE_LEDGER_SECRET") or ""
        self.file_path = file_path or Path(
            os.getenv("SORGE_LEDGER_FILE") or str(DEFAULT_FILE)
        )

    @property
    def configured(self) -> bool:
        """True when we can persist tickets (HTTP ledger or local file)."""
        return True  # file fallback always available

    @property
    def remote(self) -> bool:
        return bool(self.base_url and self.secret)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.secret}",
            "Content-Type": "application/json",
            "User-Agent": "deepiri-sorge-ledger",
        }

    def enqueue(self, tickets: list[EscalateTicket]) -> int:
        if not tickets:
            return 0
        if self.remote:
            return self._enqueue_remote(tickets)
        return self._enqueue_file(tickets)

    def cancel_pr(self, repo: str, pr_number: int) -> int:
        if self.remote:
            try:
                r = requests.post(
                    f"{self.base_url}/ledger/tickets/cancel",
                    headers=self._headers(),
                    json={"repo": repo, "pr_number": pr_number},
                    timeout=20,
                )
                r.raise_for_status()
                return int(r.json().get("cancelled") or 0)
            except requests.RequestException as e:
                logger.warning(f"Ledger cancel failed: {e}")
                return 0
        return self._cancel_file(repo, pr_number)

    def claim(self, limit: int = 8) -> list[EscalateTicket]:
        if self.remote:
            return self._claim_remote(limit)
        return self._claim_file(limit)

    def ack(self, ticket_ids: list[str], *, status: str = "done") -> None:
        if not ticket_ids:
            return
        if self.remote:
            try:
                requests.post(
                    f"{self.base_url}/ledger/tickets/ack",
                    headers=self._headers(),
                    json={"ticket_ids": ticket_ids, "status": status},
                    timeout=20,
                ).raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Ledger ack failed: {e}")
            return
        self._ack_file(ticket_ids, status=status)

    def pending_count(self) -> int:
        if self.remote:
            try:
                r = requests.get(
                    f"{self.base_url}/ledger/tickets",
                    headers=self._headers(),
                    params={"limit": 0, "count_only": "1"},
                    timeout=15,
                )
                if r.ok:
                    return int(r.json().get("pending") or 0)
            except requests.RequestException:
                pass
            return 0
        data = self._load_file()
        return sum(1 for t in data.get("tickets", []) if t.get("status") == "pending")

    def attach_comment(self, repo: str, pr_number: int, comment_id: int) -> int:
        """Stamp comment_id onto pending tickets for this PR (helps drain edit)."""
        if self.remote:
            try:
                r = requests.post(
                    f"{self.base_url}/ledger/tickets/attach_comment",
                    headers=self._headers(),
                    json={
                        "repo": repo,
                        "pr_number": pr_number,
                        "comment_id": comment_id,
                    },
                    timeout=20,
                )
                r.raise_for_status()
                return int(r.json().get("updated") or 0)
            except requests.RequestException as e:
                logger.warning(f"Ledger attach_comment failed: {e}")
                return 0
        data = self._load_file()
        n = 0
        for t in data.get("tickets", []):
            if (
                t.get("repo") == repo
                and int(t.get("pr_number") or 0) == pr_number
                and t.get("status") in ("pending", "claimed")
            ):
                t["comment_id"] = comment_id
                n += 1
        if n:
            self._save_file(data)
        return n

    def fetch_quota_used(self) -> dict[str, int]:
        if not self.remote:
            return {}
        try:
            r = requests.get(
                f"{self.base_url}/ledger/quota",
                headers=self._headers(),
                timeout=15,
            )
            r.raise_for_status()
            used = r.json().get("used") or {}
            return {str(k): int(v) for k, v in used.items()}
        except (requests.RequestException, TypeError, ValueError) as e:
            logger.warning(f"Ledger quota fetch failed: {e}")
            return {}

    def push_quota_used(self, used: dict[str, int]) -> None:
        if not self.remote:
            return
        try:
            requests.post(
                f"{self.base_url}/ledger/quota",
                headers=self._headers(),
                json={"used": used},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Ledger quota push failed: {e}")

    def fetch_provider_cooldowns(self) -> dict[str, float]:
        """Return provider -> cooldown_until unix ts from Worker KV."""
        if not self.remote:
            return {}
        try:
            r = requests.get(
                f"{self.base_url}/ledger/provider_status",
                headers=self._headers(),
                timeout=15,
            )
            r.raise_for_status()
            cool = r.json().get("cooldowns") or {}
            out: dict[str, float] = {}
            for k, v in cool.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
        except (requests.RequestException, TypeError, ValueError) as e:
            logger.warning(f"Ledger provider_status fetch failed: {e}")
            return {}

    def push_provider_cooldowns(self, cooldowns: dict[str, float]) -> None:
        if not self.remote:
            return
        try:
            requests.post(
                f"{self.base_url}/ledger/provider_status",
                headers=self._headers(),
                json={"cooldowns": cooldowns},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Ledger provider_status push failed: {e}")

    def acquire_slot(
        self,
        provider: str,
        *,
        holder_id: str,
        max_inflight: int = 1,
        ttl_sec: float = 180.0,
    ) -> bool:
        if not self.remote:
            return True
        try:
            r = requests.post(
                f"{self.base_url}/ledger/slots/acquire",
                headers=self._headers(),
                json={
                    "provider": provider,
                    "holder_id": holder_id,
                    "max_inflight": max_inflight,
                    "ttl_sec": ttl_sec,
                },
                timeout=15,
            )
            r.raise_for_status()
            return bool(r.json().get("ok"))
        except requests.RequestException as e:
            logger.warning(f"Ledger slot acquire failed ({provider}): {e}")
            # Fail open — don't block reviews if KV is down.
            return True

    def release_slot(self, provider: str, *, holder_id: str) -> None:
        if not self.remote:
            return
        try:
            requests.post(
                f"{self.base_url}/ledger/slots/release",
                headers=self._headers(),
                json={"provider": provider, "holder_id": holder_id},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Ledger slot release failed ({provider}): {e}")

    def schedule_review_retry(
        self,
        *,
        repo: str,
        pr_number: int,
        installation_id: int | None,
        delay_sec: float,
        comment_id: int | None = None,
    ) -> bool:
        """Enqueue one delayed sorge-review (Worker cron fires when due)."""
        if not self.remote:
            logger.info("Auto-retry skipped: ledger remote not configured")
            return False
        not_before = time.time() + max(30.0, float(delay_sec))
        try:
            r = requests.post(
                f"{self.base_url}/ledger/retries",
                headers=self._headers(),
                json={
                    "repo": repo,
                    "pr_number": pr_number,
                    "installation_id": installation_id,
                    "not_before": not_before,
                    "comment_id": comment_id,
                },
                timeout=20,
            )
            r.raise_for_status()
            logger.info(
                f"Auto-retry scheduled for {repo}#{pr_number} "
                f"in ~{delay_sec:.0f}s (not_before={not_before:.0f})"
            )
            return True
        except requests.RequestException as e:
            logger.warning(f"Auto-retry schedule failed: {e}")
            return False

    # --- remote ---

    def _enqueue_remote(self, tickets: list[EscalateTicket]) -> int:
        try:
            r = requests.post(
                f"{self.base_url}/ledger/tickets",
                headers=self._headers(),
                json={"tickets": [t.to_dict() for t in tickets]},
                timeout=30,
            )
            r.raise_for_status()
            n = int(r.json().get("enqueued") or len(tickets))
            logger.info(f"Ledger remote enqueued {n} ticket(s)")
            return n
        except requests.RequestException as e:
            logger.warning(f"Ledger remote enqueue failed, falling back to file: {e}")
            return self._enqueue_file(tickets)

    def _claim_remote(self, limit: int) -> list[EscalateTicket]:
        try:
            r = requests.get(
                f"{self.base_url}/ledger/tickets",
                headers=self._headers(),
                params={"limit": limit},
                timeout=30,
            )
            r.raise_for_status()
            items = r.json().get("tickets") or []
            return [EscalateTicket.from_dict(x) for x in items]
        except requests.RequestException as e:
            logger.warning(f"Ledger remote claim failed: {e}")
            return []

    # --- file ---

    def _load_file(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {"tickets": []}
        try:
            return json.loads(self.file_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"tickets": []}

    def _save_file(self, data: dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(data, indent=2))

    def _enqueue_file(self, tickets: list[EscalateTicket]) -> int:
        data = self._load_file()
        existing = data.setdefault("tickets", [])
        # supersede prior pending for same repo/pr
        repos = {(t.repo, t.pr_number) for t in tickets if t.repo and t.pr_number}
        for t in existing:
            key = (t.get("repo"), t.get("pr_number"))
            if key in repos and t.get("status") == "pending":
                t["status"] = "cancelled"
        for t in tickets:
            existing.append(t.to_dict())
        self._save_file(data)
        logger.info(f"Ledger file enqueued {len(tickets)} ticket(s) → {self.file_path}")
        return len(tickets)

    def _cancel_file(self, repo: str, pr_number: int) -> int:
        data = self._load_file()
        n = 0
        for t in data.get("tickets", []):
            if (
                t.get("repo") == repo
                and int(t.get("pr_number") or 0) == pr_number
                and t.get("status") == "pending"
            ):
                t["status"] = "cancelled"
                n += 1
        self._save_file(data)
        return n

    def _claim_file(self, limit: int) -> list[EscalateTicket]:
        data = self._load_file()
        claimed: list[EscalateTicket] = []
        now = time.time()
        for t in data.get("tickets", []):
            if len(claimed) >= limit:
                break
            if t.get("status") != "pending":
                continue
            t["status"] = "claimed"
            t["claimed_at"] = now
            claimed.append(EscalateTicket.from_dict(t))
        self._save_file(data)
        return claimed

    def _ack_file(self, ticket_ids: list[str], *, status: str) -> None:
        ids = set(ticket_ids)
        data = self._load_file()
        for t in data.get("tickets", []):
            if t.get("ticket_id") in ids:
                t["status"] = status
        self._save_file(data)
        # prune old done/cancelled (>2 days)
        cutoff = time.time() - 172800
        data["tickets"] = [
            t
            for t in data.get("tickets", [])
            if t.get("status") in ("pending", "claimed")
            or float(t.get("created_at") or 0) > cutoff
        ]
        self._save_file(data)
