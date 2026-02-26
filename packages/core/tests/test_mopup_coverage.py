"""Mop-up coverage tests for remaining gaps.

Covers: cognitive.py (model backends, enrich_context, think_with_tools),
        integrations/filesystem.py, integrations/oauth.py,
        voice.py (remaining), sync/client.py (remaining), sync/server.py (remaining)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ===================================================================
# Cognitive — Model Backends
# ===================================================================


class TestOllamaModel:
    """Test OllamaModel generate."""

    def test_generate_success(self):
        from animus.cognitive import ModelConfig, ModelProvider, OllamaModel

        config = ModelConfig(provider=ModelProvider.OLLAMA, model_name="llama3:8b")
        model = OllamaModel(config)

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Hello from Ollama"}}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = model.generate("test prompt", system="Be helpful")
        assert result == "Hello from Ollama"

    def test_generate_import_error(self):
        from animus.cognitive import ModelConfig, ModelProvider, OllamaModel

        config = ModelConfig(provider=ModelProvider.OLLAMA, model_name="llama3:8b")
        model = OllamaModel(config)

        with patch.dict("sys.modules", {"ollama": None}):
            result = model.generate("test")
        assert "not installed" in result

    def test_generate_exception(self):
        from animus.cognitive import ModelConfig, ModelProvider, OllamaModel

        config = ModelConfig(provider=ModelProvider.OLLAMA, model_name="llama3:8b")
        model = OllamaModel(config)

        mock_ollama = MagicMock()
        mock_ollama.chat.side_effect = ConnectionError("offline")
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = model.generate("test")
        assert "Error" in result

    def test_generate_stream(self):
        from animus.cognitive import ModelConfig, ModelProvider, OllamaModel

        config = ModelConfig(provider=ModelProvider.OLLAMA, model_name="llama3:8b")
        model = OllamaModel(config)

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "streamed"}}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):

            async def _collect():
                chunks = []
                async for chunk in model.generate_stream("test"):
                    chunks.append(chunk)
                return "".join(chunks)

            result = asyncio.run(_collect())
        assert "streamed" in result


class TestAnthropicModel:
    """Test AnthropicModel generate."""

    def test_generate_success(self):
        from animus.cognitive import AnthropicModel, ModelConfig, ModelProvider

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-6",
            api_key="test-key",
        )
        model = AnthropicModel(config)

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Hello from Claude")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = model.generate("test", system="Be helpful")
        assert result == "Hello from Claude"

    def test_generate_import_error(self):
        from animus.cognitive import AnthropicModel, ModelConfig, ModelProvider

        config = ModelConfig(provider=ModelProvider.ANTHROPIC, model_name="claude-sonnet-4-6")
        model = AnthropicModel(config)

        with patch.dict("sys.modules", {"anthropic": None}):
            result = model.generate("test")
        assert "not installed" in result

    def test_generate_exception(self):
        from animus.cognitive import AnthropicModel, ModelConfig, ModelProvider

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-6",
            api_key="bad-key",
        )
        model = AnthropicModel(config)

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = RuntimeError(
            "auth failed"
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = model.generate("test")
        assert "Error" in result

    def test_generate_stream(self):
        from animus.cognitive import AnthropicModel, ModelConfig, ModelProvider

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-6",
            api_key="key",
        )
        model = AnthropicModel(config)

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="stream")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):

            async def _collect():
                chunks = []
                async for chunk in model.generate_stream("test"):
                    chunks.append(chunk)
                return "".join(chunks)

            result = asyncio.run(_collect())
        assert "stream" in result


class TestOpenAIModel:
    """Test OpenAIModel generate."""

    def test_generate_success(self):
        from animus.cognitive import ModelConfig, ModelProvider, OpenAIModel

        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
        )
        model = OpenAIModel(config)

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from GPT"))]
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = model.generate("test", system="Be helpful")
        assert result == "Hello from GPT"

    def test_generate_import_error(self):
        from animus.cognitive import ModelConfig, ModelProvider, OpenAIModel

        config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4")
        model = OpenAIModel(config)

        with patch.dict("sys.modules", {"openai": None}):
            result = model.generate("test")
        assert "not installed" in result

    def test_generate_exception(self):
        from animus.cognitive import ModelConfig, ModelProvider, OpenAIModel

        config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="bad")
        model = OpenAIModel(config)

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("fail")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = model.generate("test")
        assert "Error" in result

    def test_generate_stream(self):
        from animus.cognitive import ModelConfig, ModelProvider, OpenAIModel

        config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4", api_key="key")
        model = OpenAIModel(config)

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="streamed"))]
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        with patch.dict("sys.modules", {"openai": mock_openai}):

            async def _collect():
                chunks = []
                async for chunk in model.generate_stream("test"):
                    chunks.append(chunk)
                return "".join(chunks)

            result = asyncio.run(_collect())
        assert "streamed" in result


class TestCreateModel:
    """Test create_model factory."""

    def test_create_mock(self):
        from animus.cognitive import MockModel, ModelConfig, create_model

        model = create_model(ModelConfig.mock())
        assert isinstance(model, MockModel)

    def test_create_unknown_raises(self):
        from animus.cognitive import ModelConfig, create_model

        config = ModelConfig.mock()
        config.provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unsupported"):
            create_model(config)


class TestCognitiveLayerEnrich:
    """Test CognitiveLayer _enrich_context and _build_system_prompt."""

    def test_enrich_context_with_entities(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        mock_entity = MagicMock()
        mock_entity.get_context_for_text.return_value = "Entity: Alice works at Corp"
        cognitive.entity_memory = mock_entity

        result = cognitive._enrich_context("Tell me about Alice", None)
        assert "Alice" in result

    def test_enrich_context_with_proactive(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        mock_nudge = MagicMock()
        mock_nudge.content = "You discussed this last week"
        mock_proactive = MagicMock()
        mock_proactive.generate_context_nudge.return_value = mock_nudge
        cognitive.proactive = mock_proactive

        result = cognitive._enrich_context("deployment", "Existing context")
        assert "last week" in result
        assert "Existing context" in result

    def test_enrich_context_entity_error(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        mock_entity = MagicMock()
        mock_entity.get_context_for_text.side_effect = RuntimeError("db error")
        cognitive.entity_memory = mock_entity

        result = cognitive._enrich_context("test", None)
        assert result is None  # No enrichment due to error

    def test_enrich_context_proactive_error(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        mock_proactive = MagicMock()
        mock_proactive.generate_context_nudge.side_effect = RuntimeError("fail")
        cognitive.proactive = mock_proactive

        result = cognitive._enrich_context("test", None)
        assert result is None

    def test_build_system_prompt_with_learning(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        mock_pref = MagicMock()
        mock_pref.confidence = 0.9
        mock_pref.value = "Prefer concise answers"
        mock_learning = MagicMock()
        mock_learning.get_preferences.return_value = [mock_pref]
        cognitive.learning = mock_learning

        from animus.cognitive import ReasoningMode

        prompt = cognitive._build_system_prompt(None, ReasoningMode.QUICK)
        assert "concise" in prompt

    def test_think_with_entity_extraction(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="Hello"))
        mock_entity = MagicMock()
        mock_entity.get_context_for_text.return_value = None
        cognitive.entity_memory = mock_entity

        cognitive.think("test prompt")
        mock_entity.extract_and_link.assert_called_once_with("test prompt")

    def test_think_entity_extraction_error(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="Hello"))
        mock_entity = MagicMock()
        mock_entity.get_context_for_text.return_value = None
        mock_entity.extract_and_link.side_effect = RuntimeError("extract fail")
        cognitive.entity_memory = mock_entity

        # Should not raise
        result = cognitive.think("test prompt")
        assert result == "Hello"


class TestCognitiveThinkWithTools:
    """Test think_with_tools method."""

    def test_think_with_tools_no_tool_calls(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.tools import ToolRegistry

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="No tools needed."))
        registry = ToolRegistry()

        result = cognitive.think_with_tools("What time is it?", tools=registry)
        assert result == "No tools needed."

    def test_think_with_tools_with_tool_call(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.tools import Tool, ToolRegistry, ToolResult

        # First call returns tool invocation, second returns final answer
        cognitive = CognitiveLayer(ModelConfig.mock(default_response="The time is 10:00 AM"))
        # Override model.generate to return tool call first, then final answer
        call_count = {"n": 0}
        original_generate = cognitive.primary.generate

        def sequenced_generate(prompt, system=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return '```tool\n{"tool": "get_time", "params": {}}\n```'
            return original_generate(prompt, system)

        cognitive.primary.generate = sequenced_generate

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_time",
                description="Get time",
                parameters={},
                handler=lambda p: ToolResult(tool_name="get_time", success=True, output="10:00 AM"),
            )
        )

        result = cognitive.think_with_tools("What time is it?", tools=registry)
        assert "10:00" in result

    def test_think_with_tools_unknown_tool(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.tools import ToolRegistry

        response_map = {
            "test": '```tool\n{"tool": "nonexistent", "params": {}}\n```',
        }
        cognitive = CognitiveLayer(
            ModelConfig.mock(
                default_response="Done.",
                response_map=response_map,
            )
        )
        registry = ToolRegistry()

        result = cognitive.think_with_tools("test", tools=registry)
        assert isinstance(result, str)

    def test_think_with_tools_approval_denied(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.tools import Tool, ToolRegistry, ToolResult

        response_map = {
            "delete": '```tool\n{"tool": "danger", "params": {}}\n```',
        }
        cognitive = CognitiveLayer(
            ModelConfig.mock(
                default_response="Cancelled.",
                response_map=response_map,
            )
        )
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="danger",
                description="Dangerous op",
                parameters={},
                handler=lambda p: ToolResult(tool_name="danger", success=True, output="done"),
                requires_approval=True,
            )
        )

        result = cognitive.think_with_tools(
            "delete everything",
            tools=registry,
            approval_callback=lambda name, params: False,
        )
        assert isinstance(result, str)

    def test_parse_tool_calls_invalid_json(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        calls = cognitive._parse_tool_calls("```tool\nnot json\n```")
        assert calls == []

    def test_parse_tool_calls_no_tool_name(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        cognitive = CognitiveLayer(ModelConfig.mock())
        calls = cognitive._parse_tool_calls('```tool\n{"params": {}}\n```')
        assert calls == []


# ===================================================================
# Integrations Filesystem
# ===================================================================


class TestFilesystemIntegration:
    """Cover remaining filesystem integration gaps."""

    def test_search_tool(self, tmp_path: Path):
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        (tmp_path / "hello.txt").write_text("hello world")
        # Manually add to index since we haven't connected
        fs._index[str(tmp_path / "hello.txt")] = FileEntry(
            path=str(tmp_path / "hello.txt"),
            name="hello.txt",
            extension=".txt",
            size=11,
            modified=datetime.now(),
            is_dir=False,
        )

        result = asyncio.run(fs._tool_search("hello"))
        assert result.success is True

    def test_search_no_results(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        result = asyncio.run(fs._tool_search("nonexistent"))
        assert result.success is True
        assert result.output["count"] == 0

    def test_connect_disconnect(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        result = asyncio.run(fs.connect({"paths": [str(tmp_path)]}))
        assert result is True
        assert fs.is_connected

        result = asyncio.run(fs.disconnect())
        assert result is True


# ===================================================================
# OAuth2 Flow
# ===================================================================


class TestOAuth2FlowInit:
    """Test OAuth2Flow initialization."""

    def test_init_not_available(self):
        from animus.integrations.oauth import GOOGLE_AUTH_AVAILABLE

        if not GOOGLE_AUTH_AVAILABLE:
            from animus.integrations.oauth import OAuth2Flow

            with pytest.raises(ImportError, match="not installed"):
                OAuth2Flow("id", "secret", ["scope"])


class TestSaveLoadToken:
    """Test token save/load."""

    def test_save_and_load_token(self, tmp_path: Path):
        from animus.integrations.oauth import OAuth2Token, load_token, save_token

        token = OAuth2Token(
            access_token="at",
            refresh_token="rt",
            token_type="Bearer",
            expires_at=datetime(2099, 1, 1),
            scopes=["s1"],
        )
        path = tmp_path / "token.json"
        save_token(token, path)
        loaded = load_token(path)
        assert loaded is not None
        assert loaded.access_token == "at"

    def test_load_token_missing(self, tmp_path: Path):
        from animus.integrations.oauth import load_token

        result = load_token(tmp_path / "missing.json")
        assert result is None

    def test_load_token_corrupt(self, tmp_path: Path):
        from animus.integrations.oauth import load_token

        path = tmp_path / "bad.json"
        path.write_text("not json")
        result = load_token(path)
        assert result is None


# ===================================================================
# Sync Client additional
# ===================================================================


class TestSyncClientAdditional:
    """Additional SyncClient tests."""

    def _make_state(self, tmp_path: Path):
        from animus.sync.state import SyncableState

        return SyncableState(data_dir=tmp_path)

    def test_sync_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret")
        result = asyncio.run(client.sync())
        assert result.success is False

    def test_push_changes_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret")
        result = asyncio.run(client.push_changes(MagicMock()))
        assert result is False

    def test_ping_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret")
        result = asyncio.run(client.ping())
        assert result is None


# ===================================================================
# Sync Server additional
# ===================================================================


class TestSyncServerAdditional:
    """Additional SyncServer tests."""

    def test_init(self, tmp_path: Path):
        from animus.sync.server import SyncServer
        from animus.sync.state import SyncableState

        state = SyncableState(data_dir=tmp_path)
        server = SyncServer(state, port=9999, shared_secret="secret")
        assert server.port == 9999

    def test_get_peers_empty(self, tmp_path: Path):
        from animus.sync.server import SyncServer
        from animus.sync.state import SyncableState

        state = SyncableState(data_dir=tmp_path)
        server = SyncServer(state, shared_secret="secret")
        assert server.get_peers() == []


# ===================================================================
# Voice additional
# ===================================================================


class TestVoiceAdditional:
    """Additional voice coverage."""

    def test_transcribe_microphone_success(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        mock_sd = MagicMock()
        mock_np = MagicMock()
        mock_np.float32 = "float32"
        mock_np.concatenate.return_value = MagicMock()

        # Mock the recording
        mock_sd.rec.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " Recorded text "}
        vi._model = mock_model

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": mock_np}):
            # The function records then transcribes
            result = vi.transcribe_microphone(duration=1)
        assert isinstance(result, str)
