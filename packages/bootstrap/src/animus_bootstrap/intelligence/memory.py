"""Memory manager — bridges gateway with memory backends."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from animus_bootstrap.intelligence.memory_backends.base import MemoryBackend
from animus_bootstrap.intelligence.memory_scores import MemoryScoreStore
from animus_bootstrap.intelligence.recall_log import RecallLog

logger = logging.getLogger(__name__)


DEFAULT_RECALL_OUTCOME_WINDOW_SECONDS = 1800  # 30 minutes
"""Window inside which a follow-on user action counts as positive outcome for a recall."""


RecallActionObserver = Callable[[datetime, datetime], Awaitable[int] | int]
"""Callable returning the count of relevant user actions in [since, until)."""


@dataclass
class MemoryContext:
    """Relevant memories injected into LLM prompt."""

    episodic: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)
    procedural: list[str] = field(default_factory=list)
    user_prefs: dict[str, str] = field(default_factory=dict)
    recall_id: str | None = None


class MemoryManager:
    """Bridges gateway with memory backends, with optional hit-rate ranking."""

    def __init__(
        self,
        backend: MemoryBackend,
        recall_log: RecallLog | None = None,
        score_store: MemoryScoreStore | None = None,
        action_observers: list[RecallActionObserver] | None = None,
        rerank_alpha: float = 0.3,
        outcome_window_seconds: int = DEFAULT_RECALL_OUTCOME_WINDOW_SECONDS,
    ) -> None:
        self._backend = backend
        self._recall_log = recall_log
        self._score_store = score_store
        self._action_observers: list[RecallActionObserver] = list(action_observers or [])
        self._rerank_alpha = float(rerank_alpha)
        self._outcome_window_seconds = int(outcome_window_seconds)
        self._observer_tasks: set[asyncio.Task[None]] = set()

    @property
    def recall_log(self) -> RecallLog | None:
        return self._recall_log

    @property
    def score_store(self) -> MemoryScoreStore | None:
        return self._score_store

    def add_action_observer(self, observer: RecallActionObserver) -> None:
        """Register an action observer used for implicit outcome capture."""
        self._action_observers.append(observer)

    async def recall(
        self,
        query: str,
        limit: int = 5,
        context_hint: str = "",
    ) -> MemoryContext:
        """Retrieve relevant memories for a query, rerank by hit rate, log."""
        episodic_results = await self._backend.search(query, memory_type="episodic", limit=limit)
        semantic_results = await self._backend.search(query, memory_type="semantic", limit=limit)
        procedural_results = await self._backend.search(
            query, memory_type="procedural", limit=limit
        )

        episodic_results = self._rerank(episodic_results)
        semantic_results = self._rerank(semantic_results)
        procedural_results = self._rerank(procedural_results)

        stats = await self._backend.get_stats()
        user_prefs: dict[str, str] = stats.get("user_preferences", {})

        recall_id = self._log_recall(
            query=query,
            context_hint=context_hint,
            episodic=episodic_results,
            semantic=semantic_results,
            procedural=procedural_results,
        )

        if recall_id is not None:
            self._spawn_recall_observer(recall_id)

        return MemoryContext(
            episodic=[r["content"] for r in episodic_results],
            semantic=[r["content"] for r in semantic_results],
            procedural=[r["content"] for r in procedural_results],
            user_prefs=user_prefs,
            recall_id=recall_id,
        )

    async def store_conversation(self, session_id: str, messages: list[dict]) -> None:
        """Store a conversation summary as episodic memory.

        Only stores the last user message and assistant response (not full
        history) to prevent memory bloat that overwhelms the system prompt.
        """
        last_user = ""
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not last_assistant:
                last_assistant = msg.get("content", "")[:500]
            elif msg.get("role") == "user" and not last_user:
                last_user = msg.get("content", "")[:200]
            if last_user and last_assistant:
                break

        if not last_user:
            return

        content = f"User asked: {last_user}\nAnimus replied: {last_assistant}"
        metadata = {"session_id": session_id, "message_count": len(messages)}
        await self._backend.store("episodic", content, metadata)

    async def store_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Store a knowledge triple in semantic memory."""
        content = f"{subject} {predicate} {obj}"
        metadata = {"subject": subject, "predicate": predicate, "object": obj}
        await self._backend.store("semantic", content, metadata)

    async def store_preference(self, key: str, value: str) -> None:
        """Store a user preference."""
        await self._backend.store(
            "procedural",
            f"User preference: {key} = {value}",
            {"preference_key": key, "preference_value": value},
        )
        if hasattr(self._backend, "store_preference"):
            await self._backend.store_preference(key, value)

    async def search(self, query: str, memory_type: str = "all", limit: int = 20) -> list[dict]:
        """Full-text search across memory stores."""
        return await self._backend.search(query, memory_type=memory_type, limit=limit)

    async def get_stats(self) -> dict:
        """Return memory statistics (counts per type, storage size)."""
        return await self._backend.get_stats()

    def record_explicit_feedback(self, recall_id: str, signal_value: float) -> None:
        """Backpropagate an explicit ±1 rating to every memory in this recall."""
        if self._recall_log is None or self._score_store is None:
            return
        for memory_id in self._recall_log.memory_ids_for(recall_id):
            self._score_store.record_outcome(memory_id, signal_value)

    async def shutdown(self) -> None:
        """Cancel pending observer tasks. Call before close()."""
        for task in list(self._observer_tasks):
            task.cancel()
        if self._observer_tasks:
            await asyncio.gather(*self._observer_tasks, return_exceptions=True)
            self._observer_tasks.clear()

    def close(self) -> None:
        """Close the underlying backend."""
        self._backend.close()

    def _rerank(self, results: list[dict]) -> list[dict]:
        """Rescore results by similarity-position * (1 + alpha * memory_score).

        Pure passthrough when score store is missing or alpha == 0.
        """
        if not results or self._score_store is None or self._rerank_alpha == 0.0:
            return results

        n = len(results)
        rescored: list[tuple[float, int, dict]] = []
        now = datetime.now(UTC)
        for idx, entry in enumerate(results):
            position_score = 1.0 - (idx / n) if n > 1 else 1.0
            memory_id = entry.get("id", "")
            mem_score = self._score_store.effective_score(memory_id, now=now) if memory_id else 0.0
            final = position_score * (1.0 + self._rerank_alpha * mem_score)
            entry["_rank_score"] = round(final, 6)
            entry["_memory_score"] = round(mem_score, 6)
            rescored.append((final, idx, entry))

        rescored.sort(key=lambda t: (-t[0], t[1]))
        return [entry for _, _, entry in rescored]

    def _log_recall(
        self,
        query: str,
        context_hint: str,
        episodic: list[dict],
        semantic: list[dict],
        procedural: list[dict],
    ) -> str | None:
        """Persist this recall and the memory IDs it returned."""
        if self._recall_log is None:
            return None

        memory_entries: list[tuple[str, str]] = []
        for entry in episodic:
            mid = entry.get("id", "")
            if mid:
                memory_entries.append((mid, "episodic"))
        for entry in semantic:
            mid = entry.get("id", "")
            if mid:
                memory_entries.append((mid, "semantic"))
        for entry in procedural:
            mid = entry.get("id", "")
            if mid:
                memory_entries.append((mid, "procedural"))

        try:
            return self._recall_log.log_recall(
                query=query,
                memory_entries=memory_entries,
                context_hint=context_hint,
            )
        except Exception:
            logger.exception("Failed to log recall")
            return None

    def _spawn_recall_observer(self, recall_id: str) -> None:
        """Spawn a background task that records implicit outcome at window close."""
        if (
            self._score_store is None
            or self._recall_log is None
            or not self._action_observers
            or self._outcome_window_seconds <= 0
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        recalled_at = datetime.now(UTC)
        task = loop.create_task(self._observe_recall_window(recall_id, recalled_at))
        self._observer_tasks.add(task)
        task.add_done_callback(self._observer_tasks.discard)

    async def _observe_recall_window(self, recall_id: str, recalled_at: datetime) -> None:
        try:
            await asyncio.sleep(self._outcome_window_seconds)
        except asyncio.CancelledError:
            return

        until = datetime.now(UTC)
        total = 0
        for observer in list(self._action_observers):
            try:
                result = observer(recalled_at, until)
                if asyncio.iscoroutine(result):
                    result = await result
                total += int(result or 0)
            except Exception:
                logger.exception("Recall action observer raised; skipping")

        signal = 1.0 if total > 0 else 0.0
        if self._recall_log is None or self._score_store is None:
            return
        try:
            for memory_id in self._recall_log.memory_ids_for(recall_id):
                self._score_store.record_outcome(memory_id, signal)
        except Exception:
            logger.exception("Failed to record recall outcome for %s", recall_id)
