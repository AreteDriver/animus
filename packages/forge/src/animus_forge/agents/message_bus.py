"""Agent-to-agent message bus for runtime communication.

Provides a lightweight in-process pub/sub system that agents use
to exchange messages during parallel execution. Messages are
topic-based with optional filtering and TTL.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class MessagePriority(StrEnum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """A message exchanged between agents.

    Attributes:
        id: Unique message identifier.
        topic: Topic/channel name (e.g. "build.progress", "review.findings").
        sender: Agent run_id or role name of the sender.
        payload: Message content (any serializable data).
        priority: Message priority for ordering.
        timestamp: Epoch time when message was created.
        ttl_seconds: Time-to-live in seconds (0 = no expiry).
        reply_to: Optional message ID this is replying to.
        metadata: Additional key-value metadata.
    """

    id: str
    topic: str
    sender: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = 0.0
    ttl_seconds: float = 0.0
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "sender": self.sender,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "reply_to": self.reply_to,
        }


# Type for subscriber callbacks
Subscriber = Any  # Callable[[AgentMessage], None]


class AgentMessageBus:
    """In-process pub/sub message bus for agent communication.

    Thread-safe. Agents publish to topics, subscribers receive
    matching messages. Supports topic wildcards, message history,
    and TTL-based expiry.

    Usage:
        bus = AgentMessageBus()

        # Subscribe to a topic
        def on_review(msg):
            print(f"Review from {msg.sender}: {msg.payload}")
        bus.subscribe("review.*", on_review)

        # Publish a message
        bus.publish("review.findings", sender="reviewer", payload={"issues": 3})

        # Query history
        msgs = bus.get_messages("review.findings", limit=10)
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the message bus.

        Args:
            max_history: Maximum messages to retain in history per topic.
        """
        self._max_history = max_history
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._history: dict[str, list[AgentMessage]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(
        self,
        topic: str,
        sender: str,
        payload: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: float = 0.0,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Publish a message to a topic.

        Args:
            topic: Topic name (e.g. "build.progress").
            sender: Sender identifier (agent role or run_id).
            payload: Message content.
            priority: Message priority.
            ttl_seconds: Time-to-live (0 = no expiry).
            reply_to: Optional message ID this replies to.
            metadata: Optional key-value metadata.

        Returns:
            The published AgentMessage.
        """
        msg = AgentMessage(
            id=f"msg-{uuid.uuid4().hex[:12]}",
            topic=topic,
            sender=sender,
            payload=payload,
            priority=priority,
            ttl_seconds=ttl_seconds,
            reply_to=reply_to,
            metadata=metadata or {},
        )

        with self._lock:
            # Store in history
            history = self._history[topic]
            history.append(msg)
            if len(history) > self._max_history:
                self._history[topic] = history[-self._max_history :]

            # Notify matching subscribers
            for pattern, callbacks in self._subscribers.items():
                if self._topic_matches(pattern, topic):
                    for callback in callbacks:
                        try:
                            callback(msg)
                        except Exception as e:
                            logger.warning(
                                "Message bus subscriber error on topic %s: %s",
                                topic,
                                e,
                            )

        return msg

    def subscribe(self, pattern: str, callback: Subscriber) -> None:
        """Subscribe to messages matching a topic pattern.

        Supports wildcards:
        - "build.*" matches "build.progress", "build.complete"
        - "*" matches all topics

        Args:
            pattern: Topic pattern to match.
            callback: Function(AgentMessage) called on match.
        """
        with self._lock:
            self._subscribers[pattern].append(callback)

    def unsubscribe(self, pattern: str, callback: Subscriber) -> bool:
        """Remove a subscription.

        Args:
            pattern: Topic pattern.
            callback: Previously registered callback.

        Returns:
            True if callback was found and removed.
        """
        with self._lock:
            if pattern in self._subscribers:
                try:
                    self._subscribers[pattern].remove(callback)
                    if not self._subscribers[pattern]:
                        del self._subscribers[pattern]
                    return True
                except ValueError:
                    return False
            return False

    def get_messages(
        self,
        topic: str,
        limit: int = 50,
        since: float = 0.0,
        sender: str | None = None,
    ) -> list[AgentMessage]:
        """Get messages from history.

        Args:
            topic: Exact topic name to query.
            limit: Maximum messages to return.
            since: Only messages after this epoch time.
            sender: Optional filter by sender.

        Returns:
            List of matching messages (newest last).
        """
        with self._lock:
            messages = self._history.get(topic, [])

            # Filter expired
            messages = [m for m in messages if not m.is_expired]

            # Filter by time
            if since > 0:
                messages = [m for m in messages if m.timestamp > since]

            # Filter by sender
            if sender:
                messages = [m for m in messages if m.sender == sender]

            return messages[-limit:]

    def get_topics(self) -> list[str]:
        """Get all topics with messages in history."""
        with self._lock:
            return list(self._history.keys())

    def get_thread(self, message_id: str) -> list[AgentMessage]:
        """Get a message thread (original + all replies).

        Args:
            message_id: ID of any message in the thread.

        Returns:
            List of messages in chronological order.
        """
        with self._lock:
            thread = []
            for messages in self._history.values():
                for msg in messages:
                    if msg.id == message_id or msg.reply_to == message_id:
                        thread.append(msg)
            thread.sort(key=lambda m: m.timestamp)
            return thread

    def clear_topic(self, topic: str) -> int:
        """Clear all messages for a topic.

        Args:
            topic: Topic to clear.

        Returns:
            Number of messages removed.
        """
        with self._lock:
            count = len(self._history.get(topic, []))
            self._history.pop(topic, None)
            return count

    def clear_all(self) -> int:
        """Clear all messages and subscriptions.

        Returns:
            Total messages removed.
        """
        with self._lock:
            total = sum(len(msgs) for msgs in self._history.values())
            self._history.clear()
            self._subscribers.clear()
            return total

    @property
    def message_count(self) -> int:
        """Total messages across all topics."""
        with self._lock:
            return sum(len(msgs) for msgs in self._history.values())

    @property
    def topic_count(self) -> int:
        """Number of active topics."""
        with self._lock:
            return len(self._history)

    @property
    def subscriber_count(self) -> int:
        """Total subscriber registrations."""
        with self._lock:
            return sum(len(cbs) for cbs in self._subscribers.values())

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Supports:
        - Exact match: "build.progress" matches "build.progress"
        - Wildcard suffix: "build.*" matches "build.progress", "build.complete"
        - Global wildcard: "*" matches everything

        Args:
            pattern: Subscription pattern.
            topic: Actual topic name.

        Returns:
            True if topic matches pattern.
        """
        if pattern == "*":
            return True
        if pattern == topic:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")
        return False
