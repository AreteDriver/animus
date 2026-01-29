"""Verify concrete classes conform to Protocol interfaces."""

import sys

import pytest

from animus.cognitive import AnthropicModel, MockModel, OllamaModel
from animus.learning.guardrails import GuardrailManager
from animus.memory import ChromaMemoryStore, LocalMemoryStore
from animus.protocols import (
    IntelligenceProvider,
    MemoryProvider,
    SafetyGuard,
    SyncProvider,
)
from animus.sync.client import SyncClient


class TestMemoryProviderConformance:
    def test_local_memory_store(self):
        assert issubclass(LocalMemoryStore, MemoryProvider)

    def test_chroma_memory_store(self):
        assert issubclass(ChromaMemoryStore, MemoryProvider)


class TestIntelligenceProviderConformance:
    def test_mock_model(self):
        assert issubclass(MockModel, IntelligenceProvider)

    def test_ollama_model(self):
        assert issubclass(OllamaModel, IntelligenceProvider)

    def test_anthropic_model(self):
        assert issubclass(AnthropicModel, IntelligenceProvider)


class TestSyncProviderConformance:
    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="Python < 3.12 @runtime_checkable Protocol cannot match @property to attribute annotation",
    )
    def test_sync_client(self):
        client = SyncClient.__new__(SyncClient)
        assert isinstance(client, SyncProvider)


class TestSafetyGuardConformance:
    def test_guardrail_manager(self):
        assert issubclass(GuardrailManager, SafetyGuard)
