"""
Tests for Gorgon Integration — GorgonClient, GorgonIntegration, delegation heuristic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from animus.cognitive import (
    CognitiveLayer,
    ModelConfig,
    should_delegate_to_gorgon,
)

# =============================================================================
# Delegation Heuristic Tests
# =============================================================================


class TestShouldDelegateToGorgon:
    """Tests for the should_delegate_to_gorgon heuristic."""

    def test_simple_question_not_delegated(self):
        assert should_delegate_to_gorgon("What time is it?") is False

    def test_greeting_not_delegated(self):
        assert should_delegate_to_gorgon("Hello, how are you?") is False

    def test_two_patterns_triggers(self):
        # "write tests" + "refactor"
        assert should_delegate_to_gorgon("Write tests and refactor the module") is True

    def test_one_pattern_two_keywords(self):
        # "implement" pattern + "codebase" + "architecture" keywords
        assert (
            should_delegate_to_gorgon("Implement changes across the codebase architecture") is True
        )

    def test_long_technical_prompt(self):
        # > 500 chars with 1 pattern
        long_prompt = "Review code " + "x " * 300
        assert len(long_prompt) > 500
        assert should_delegate_to_gorgon(long_prompt) is True

    def test_short_single_pattern_not_enough(self):
        # Only 1 pattern, 0 keywords, short prompt
        assert should_delegate_to_gorgon("debug this") is False

    def test_audit_plus_security(self):
        # "audit" pattern + "security audit" keyword + "codebase" keyword
        assert should_delegate_to_gorgon("Run a security audit on the codebase") is True

    def test_create_test_and_review(self):
        # "create a test" + "review code"
        assert should_delegate_to_gorgon("Create a test suite then review code for issues") is True

    def test_benchmark_plus_pipeline(self):
        assert should_delegate_to_gorgon("Benchmark the pipeline and optimize the queries") is True


# =============================================================================
# GorgonClient Tests
# =============================================================================


class TestGorgonClient:
    """Tests for GorgonClient HTTP client."""

    def test_init_defaults(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()
        assert client.base_url == "http://localhost:8000"
        assert client.api_key is None
        assert client.timeout == 30.0
        assert client._client is None

    def test_init_custom(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient(url="http://gorgon:9000/", api_key="secret", timeout=60.0)
        assert client.base_url == "http://gorgon:9000"
        assert client.api_key == "secret"
        assert client.timeout == 60.0

    def test_ensure_client_creates_httpx(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient(api_key="test-key")

        async def _run():
            http = await client._ensure_client()
            assert http is not None
            assert "Authorization" in http.headers
            assert http.headers["Authorization"] == "Bearer test-key"
            await client.close()

        asyncio.run(_run())

    def test_ensure_client_no_auth_header_without_key(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        async def _run():
            http = await client._ensure_client()
            assert "Authorization" not in http.headers
            await client.close()

        asyncio.run(_run())

    def test_close_when_no_client(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        async def _run():
            await client.close()  # should not raise

        asyncio.run(_run())

    def test_close_when_client_exists(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        async def _run():
            await client._ensure_client()
            assert client._client is not None
            await client.close()
            assert client._client is None

        asyncio.run(_run())

    def test_submit_task(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task-1", "status": "pending"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.submit_task("Test", "Description", priority=3)
            assert result["id"] == "task-1"
            mock_http.post.assert_awaited_once_with(
                "/v1/tasks",
                json={"title": "Test", "description": "Description", "priority": 3},
            )

        asyncio.run(_run())

    def test_submit_task_with_agent_role(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task-2"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            await client.submit_task("T", "D", agent_role="tester")
            call_args = mock_http.post.call_args
            assert call_args[1]["json"]["agent_role"] == "tester"

        asyncio.run(_run())

    def test_get_task(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "t1", "status": "completed"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.get_task("t1")
            assert result["status"] == "completed"
            mock_http.get.assert_awaited_once_with("/v1/tasks/t1")

        asyncio.run(_run())

    def test_cancel_task(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "t1", "status": "cancelled"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.cancel_task("t1")
            assert result["status"] == "cancelled"

        asyncio.run(_run())

    def test_get_stats(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"pending": 2, "completed": 5}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.get_stats()
            assert result["pending"] == 2

        asyncio.run(_run())

    def test_list_tasks(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "t1"}, {"id": "t2"}]
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.list_tasks(limit=5, status="pending")
            assert len(result) == 2
            mock_http.get.assert_awaited_once_with(
                "/v1/tasks", params={"limit": 5, "status": "pending"}
            )

        asyncio.run(_run())

    def test_list_tasks_no_status_filter(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            await client.list_tasks()
            call_args = mock_http.get.call_args
            assert "status" not in call_args[1]["params"]

        asyncio.run(_run())

    def test_check_health(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.check_health()
            assert result["status"] == "ok"

        asyncio.run(_run())

    def test_submit_and_wait_immediate_complete(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()

        submit_resp = MagicMock()
        submit_resp.json.return_value = {"id": "t1"}
        submit_resp.raise_for_status = MagicMock()

        get_resp = MagicMock()
        get_resp.json.return_value = {"id": "t1", "status": "completed", "result": "ok"}
        get_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = submit_resp
        mock_http.get.return_value = get_resp
        mock_http.is_closed = False
        client._client = mock_http

        async def _run():
            result = await client.submit_and_wait("T", "D", max_wait=10.0)
            assert result["status"] == "completed"

        asyncio.run(_run())


# =============================================================================
# GorgonIntegration Tests
# =============================================================================


class TestGorgonIntegration:
    """Tests for GorgonIntegration BaseIntegration."""

    def test_init(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        assert integration.name == "gorgon"
        assert integration.display_name == "Gorgon"
        assert integration._client is None

    def test_get_tools(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        tools = integration.get_tools()
        names = {t.name for t in tools}
        assert names == {
            "gorgon_delegate",
            "gorgon_status",
            "gorgon_check",
            "gorgon_list",
            "gorgon_cancel",
            "gorgon_execute",
            "gorgon_execution_status",
            "gorgon_executions",
            "gorgon_approvals",
            "gorgon_approve",
        }

    def test_connect_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        mock_client_cls = MagicMock()
        mock_client_inst = AsyncMock()
        mock_client_inst.check_health.return_value = {"status": "ok"}
        mock_client_cls.return_value = mock_client_inst

        async def _run():
            with patch("animus.integrations.gorgon.GorgonClient", mock_client_cls):
                result = await integration.connect({"url": "http://test:8000"})
            assert result is True
            assert integration.is_connected

        asyncio.run(_run())

    def test_connect_failure(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        mock_client_cls = MagicMock()
        mock_client_inst = AsyncMock()
        mock_client_inst.check_health.side_effect = ConnectionError("refused")
        mock_client_inst.close = AsyncMock()
        mock_client_cls.return_value = mock_client_inst

        async def _run():
            with patch("animus.integrations.gorgon.GorgonClient", mock_client_cls):
                result = await integration.connect({"url": "http://bad:9999"})
            assert result is False
            assert not integration.is_connected

        asyncio.run(_run())

    def test_connect_no_httpx(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            with patch("animus.integrations.gorgon.HTTPX_AVAILABLE", False):
                result = await integration.connect({})
            assert result is False

        asyncio.run(_run())

    def test_disconnect(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()

        async def _run():
            result = await integration.disconnect()
            assert result is True
            assert integration._client is None

        asyncio.run(_run())

    def test_verify_healthy(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.check_health.return_value = {"status": "ok"}

        async def _run():
            assert await integration.verify() is True

        asyncio.run(_run())

    def test_verify_no_client(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            assert await integration.verify() is False

        asyncio.run(_run())

    def test_verify_failure(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.check_health.side_effect = Exception("timeout")

        async def _run():
            assert await integration.verify() is False

        asyncio.run(_run())

    # Tool handler tests

    def test_tool_delegate_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_delegate(task="do stuff")
            assert result.success is False
            assert "Not connected" in result.error

        asyncio.run(_run())

    def test_tool_delegate_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.submit_task.return_value = {"id": "t1", "status": "pending"}

        async def _run():
            result = await integration._tool_delegate(task="Review code")
            assert result.success is True
            assert result.output["id"] == "t1"

        asyncio.run(_run())

    def test_tool_delegate_wait(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.submit_and_wait.return_value = {
            "id": "t1",
            "status": "completed",
        }

        async def _run():
            result = await integration._tool_delegate(task="Build feature", wait=True)
            assert result.success is True
            integration._client.submit_and_wait.assert_awaited_once()

        asyncio.run(_run())

    def test_tool_stats_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_stats()
            assert result.success is False

        asyncio.run(_run())

    def test_tool_stats_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.get_stats.return_value = {"pending": 3}

        async def _run():
            result = await integration._tool_stats()
            assert result.success is True
            assert result.output["pending"] == 3

        asyncio.run(_run())

    def test_tool_check_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.get_task.return_value = {"id": "t1", "status": "completed"}

        async def _run():
            result = await integration._tool_check(task_id="t1")
            assert result.success is True

        asyncio.run(_run())

    def test_tool_list_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.list_tasks.return_value = [{"id": "t1"}]

        async def _run():
            result = await integration._tool_list(limit=3)
            assert result.success is True
            assert len(result.output) == 1

        asyncio.run(_run())

    def test_tool_cancel_success(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.cancel_task.return_value = {"id": "t1", "status": "cancelled"}

        async def _run():
            result = await integration._tool_cancel(task_id="t1")
            assert result.success is True

        asyncio.run(_run())

    def test_tool_delegate_error(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        integration._client = AsyncMock()
        integration._client.submit_task.side_effect = Exception("network error")

        async def _run():
            result = await integration._tool_delegate(task="fail")
            assert result.success is False
            assert "Delegation failed" in result.error

        asyncio.run(_run())


# =============================================================================
# CognitiveLayer Delegation Tests
# =============================================================================


class TestCognitiveLayerDelegation:
    """Tests for CognitiveLayer.delegate_to_gorgon method."""

    def test_delegate_without_client_falls_back(self):
        config = ModelConfig.mock(default_response="local response")
        cognitive = CognitiveLayer(primary_config=config)

        async def _run():
            result = await cognitive.delegate_to_gorgon("do something")
            assert result["fallback"] is True
            assert result["response"] == "local response"

        asyncio.run(_run())

    def test_delegate_with_client_submits(self):
        config = ModelConfig.mock()
        mock_client = AsyncMock()
        mock_client.submit_task.return_value = {"id": "t1", "status": "pending"}

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon("build feature")
            assert result["id"] == "t1"
            mock_client.submit_task.assert_awaited_once()

        asyncio.run(_run())

    def test_delegate_with_wait(self):
        config = ModelConfig.mock()
        mock_client = AsyncMock()
        mock_client.submit_and_wait.return_value = {
            "id": "t1",
            "status": "completed",
        }

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon("build it", wait=True)
            assert result["status"] == "completed"
            mock_client.submit_and_wait.assert_awaited_once()

        asyncio.run(_run())

    def test_delegate_error_falls_back(self):
        config = ModelConfig.mock(default_response="fallback reply")
        mock_client = AsyncMock()
        mock_client.submit_task.side_effect = Exception("connection refused")

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon("do work")
            assert result["fallback"] is True
            assert result["response"] == "fallback reply"

        asyncio.run(_run())

    def test_delegate_workflow_execute(self):
        config = ModelConfig.mock()
        mock_client = AsyncMock()
        mock_client.execute_workflow.return_value = {
            "execution_id": "exec-1",
            "status": "running",
        }

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon(
                "run pipeline", workflow_id="wf-1", variables={"env": "staging"}
            )
            assert result["execution_id"] == "exec-1"
            mock_client.execute_workflow.assert_awaited_once_with(
                "wf-1", variables={"env": "staging"}
            )
            mock_client.submit_task.assert_not_awaited()

        asyncio.run(_run())

    def test_delegate_workflow_execute_and_wait(self):
        config = ModelConfig.mock()
        mock_client = AsyncMock()
        mock_client.execute_and_wait.return_value = {
            "execution_id": "exec-1",
            "status": "completed",
        }

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon(
                "run pipeline", workflow_id="wf-1", wait=True
            )
            assert result["status"] == "completed"
            mock_client.execute_and_wait.assert_awaited_once_with(
                "wf-1", variables=None, on_approval=None
            )

        asyncio.run(_run())

    def test_delegate_workflow_with_approval_callback(self):
        config = ModelConfig.mock()
        mock_client = AsyncMock()
        mock_client.execute_and_wait.return_value = {
            "execution_id": "exec-1",
            "status": "completed",
        }

        callback = AsyncMock(return_value={"approve": True, "reason": "LGTM"})
        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon(
                "deploy", workflow_id="wf-1", wait=True, on_approval=callback
            )
            assert result["status"] == "completed"
            mock_client.execute_and_wait.assert_awaited_once_with(
                "wf-1", variables=None, on_approval=callback
            )

        asyncio.run(_run())

    def test_delegate_workflow_error_falls_back(self):
        config = ModelConfig.mock(default_response="fallback")
        mock_client = AsyncMock()
        mock_client.execute_workflow.side_effect = ConnectionError("refused")

        cognitive = CognitiveLayer(primary_config=config, gorgon_client=mock_client)

        async def _run():
            result = await cognitive.delegate_to_gorgon("deploy", workflow_id="wf-1")
            assert result["fallback"] is True
            assert result["response"] == "fallback"

        asyncio.run(_run())


# =============================================================================
# Config Tests
# =============================================================================


class TestGorgonConfig:
    """Tests for GorgonConfig in animus config."""

    def test_default_values(self):
        from animus.config import GorgonConfig

        cfg = GorgonConfig()
        assert cfg.enabled is False
        assert cfg.url == "http://localhost:8000"
        assert cfg.api_key is None
        assert cfg.timeout == 30.0
        assert cfg.poll_interval == 5.0
        assert cfg.max_wait == 300.0
        assert cfg.auto_delegate is False

    def test_env_override(self):
        from animus.config import GorgonConfig

        with patch.dict(
            "os.environ",
            {"GORGON_ENABLED": "true", "GORGON_URL": "http://remote:9000"},
        ):
            cfg = GorgonConfig()
            assert cfg.enabled is True
            assert cfg.url == "http://remote:9000"

    def test_integration_config_has_gorgon(self):
        from animus.config import IntegrationConfig

        ic = IntegrationConfig()
        assert hasattr(ic, "gorgon")
        assert ic.gorgon.enabled is False

    def test_gorgon_in_config_to_dict(self):
        from animus.config import AnimusConfig

        config = AnimusConfig()
        d = config.to_dict()
        assert "gorgon" in d["integrations"]


# =============================================================================
# Integration __init__ Tests
# =============================================================================


class TestIntegrationsInit:
    """Tests for gorgon in integrations __init__."""

    def test_gorgon_integration_importable(self):
        from animus.integrations.gorgon import GorgonIntegration

        assert GorgonIntegration is not None

    def test_gorgon_in_all(self):
        import animus.integrations

        assert "GorgonIntegration" in animus.integrations.__all__


# =============================================================================
# GorgonClient Execution API Tests
# =============================================================================


class TestGorgonClientExecutions:
    """Tests for execution API methods on GorgonClient."""

    def _make_client(self):
        from animus.integrations.gorgon import GorgonClient

        client = GorgonClient()
        mock_http = AsyncMock()
        mock_http.is_closed = False
        client._client = mock_http
        return client, mock_http

    def _mock_response(self, data):
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    def test_execute_workflow(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response(
            {"execution_id": "exec-1", "status": "running", "poll_url": "/v1/executions/exec-1"}
        )

        async def _run():
            result = await client.execute_workflow("my-workflow", {"key": "val"})
            assert result["execution_id"] == "exec-1"
            mock_http.post.assert_awaited_once_with(
                "/v1/workflows/my-workflow/execute",
                json={"variables": {"key": "val"}},
            )

        asyncio.run(_run())

    def test_execute_workflow_no_variables(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"execution_id": "exec-2"})

        async def _run():
            await client.execute_workflow("wf-1")
            call_args = mock_http.post.call_args
            assert call_args[1]["json"] == {"variables": {}}

        asyncio.run(_run())

    def test_get_execution(self):
        client, mock_http = self._make_client()
        mock_http.get.return_value = self._mock_response({"id": "exec-1", "status": "completed"})

        async def _run():
            result = await client.get_execution("exec-1")
            assert result["status"] == "completed"
            mock_http.get.assert_awaited_once_with("/v1/executions/exec-1")

        asyncio.run(_run())

    def test_list_executions(self):
        client, mock_http = self._make_client()
        mock_http.get.return_value = self._mock_response({"data": [], "total": 0, "page": 1})

        async def _run():
            result = await client.list_executions(page=2, page_size=10, status="running")
            assert result["total"] == 0
            mock_http.get.assert_awaited_once_with(
                "/v1/executions",
                params={"page": 2, "page_size": 10, "status": "running"},
            )

        asyncio.run(_run())

    def test_list_executions_no_status_filter(self):
        client, mock_http = self._make_client()
        mock_http.get.return_value = self._mock_response({"data": [], "total": 5})

        async def _run():
            await client.list_executions()
            call_args = mock_http.get.call_args
            assert "status" not in call_args[1]["params"]

        asyncio.run(_run())

    def test_get_approval_status(self):
        client, mock_http = self._make_client()
        mock_http.get.return_value = self._mock_response(
            {
                "execution_id": "exec-1",
                "pending_approvals": [
                    {"token": "abc123", "step_id": "gate", "prompt": "Deploy?", "status": "pending"}
                ],
                "total_tokens": 1,
            }
        )

        async def _run():
            result = await client.get_approval_status("exec-1")
            assert len(result["pending_approvals"]) == 1
            assert result["pending_approvals"][0]["token"] == "abc123"
            mock_http.get.assert_awaited_once_with("/v1/executions/exec-1/approval")

        asyncio.run(_run())

    def test_resume_execution_with_token(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response(
            {"status": "approved", "execution_id": "exec-1"}
        )

        async def _run():
            result = await client.resume_execution(
                "exec-1", token="abc123", approve=True, approved_by="user"
            )
            assert result["status"] == "approved"
            mock_http.post.assert_awaited_once_with(
                "/v1/executions/exec-1/resume",
                json={"token": "abc123", "approve": True, "approved_by": "user"},
            )

        asyncio.run(_run())

    def test_resume_execution_reject_with_reason(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"status": "rejected"})

        async def _run():
            await client.resume_execution("exec-1", token="abc", approve=False, reason="Not ready")
            call_args = mock_http.post.call_args
            body = call_args[1]["json"]
            assert body["approve"] is False
            assert body["reason"] == "Not ready"

        asyncio.run(_run())

    def test_resume_execution_no_token(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"status": "success"})

        async def _run():
            await client.resume_execution("exec-1")
            call_args = mock_http.post.call_args
            assert call_args[1]["json"] is None

        asyncio.run(_run())

    def test_cancel_execution(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"status": "success"})

        async def _run():
            result = await client.cancel_execution("exec-1")
            assert result["status"] == "success"
            mock_http.post.assert_awaited_once_with("/v1/executions/exec-1/cancel")

        asyncio.run(_run())

    def test_execute_and_wait_completes(self):
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response(
            {"execution_id": "exec-1", "status": "running"}
        )
        mock_http.get.return_value = self._mock_response(
            {"id": "exec-1", "status": "completed", "result": "done"}
        )

        async def _run():
            result = await client.execute_and_wait("wf-1", poll_interval=0.01)
            assert result["status"] == "completed"

        asyncio.run(_run())

    def test_execute_and_wait_approval_no_callback(self):
        """Without on_approval, returns immediately on awaiting_approval."""
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response(
            {"execution_id": "exec-1", "status": "running"}
        )
        mock_http.get.return_value = self._mock_response(
            {"id": "exec-1", "status": "awaiting_approval"}
        )

        async def _run():
            result = await client.execute_and_wait("wf-1", poll_interval=0.01)
            assert result["status"] == "awaiting_approval"

        asyncio.run(_run())

    def test_execute_and_wait_approval_with_callback(self):
        """With on_approval, calls callback, resumes, continues polling."""
        client, mock_http = self._make_client()

        # First call: POST execute_workflow
        exec_resp = self._mock_response({"execution_id": "exec-1"})

        # Track GET call count for get_execution
        call_count = 0

        def _get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            url = args[0] if args else kwargs.get("url", "")
            if "/approval" in str(url):
                return self._mock_response(
                    {
                        "execution_id": "exec-1",
                        "pending_approvals": [
                            {"token": "tok-1", "status": "pending", "prompt": "Deploy?"}
                        ],
                    }
                )
            # First poll: awaiting_approval, second: completed
            if call_count <= 1:
                return self._mock_response({"id": "exec-1", "status": "awaiting_approval"})
            return self._mock_response({"id": "exec-1", "status": "completed"})

        mock_http.post.return_value = exec_resp
        mock_http.get.side_effect = _get_side_effect

        callback = AsyncMock(return_value={"approve": True, "reason": "LGTM"})

        async def _run():
            result = await client.execute_and_wait("wf-1", poll_interval=0.01, on_approval=callback)
            assert result["status"] == "completed"
            callback.assert_awaited_once()
            # Verify resume was called with the token
            resume_calls = [c for c in mock_http.post.call_args_list if "/resume" in str(c)]
            assert len(resume_calls) == 1

        asyncio.run(_run())

    def test_execute_and_wait_timeout(self):
        """Returns last status on timeout."""
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"execution_id": "exec-1"})
        mock_http.get.return_value = self._mock_response({"id": "exec-1", "status": "running"})

        async def _run():
            result = await client.execute_and_wait("wf-1", poll_interval=0.01, max_wait=0.05)
            assert result["status"] == "running"

        asyncio.run(_run())

    def test_submit_and_wait_awaiting_approval_terminal(self):
        """submit_and_wait treats awaiting_approval as terminal."""
        client, mock_http = self._make_client()
        mock_http.post.return_value = self._mock_response({"id": "task-1"})
        mock_http.get.return_value = self._mock_response(
            {"id": "task-1", "status": "awaiting_approval"}
        )

        async def _run():
            result = await client.submit_and_wait("T", "D", poll_interval=0.01)
            assert result["status"] == "awaiting_approval"

        asyncio.run(_run())


# =============================================================================
# GorgonIntegration Execution Tool Tests
# =============================================================================


class TestGorgonIntegrationExecutionTools:
    """Tests for execution and approval tools on GorgonIntegration."""

    def _make_integration(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        mock_client = AsyncMock()
        integration._client = mock_client
        integration._connected = True
        return integration, mock_client

    def test_tool_execute_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_execute(workflow_id="wf-1")
            assert result.success is False
            assert "Not connected" in result.error

        asyncio.run(_run())

    def test_tool_execute_success(self):
        integration, mock_client = self._make_integration()
        mock_client.execute_workflow.return_value = {"execution_id": "exec-1"}

        async def _run():
            result = await integration._tool_execute(workflow_id="wf-1", variables={"k": "v"})
            assert result.success is True
            assert result.output["execution_id"] == "exec-1"
            mock_client.execute_workflow.assert_awaited_once_with("wf-1", {"k": "v"})

        asyncio.run(_run())

    def test_tool_execute_with_wait(self):
        integration, mock_client = self._make_integration()
        mock_client.execute_and_wait.return_value = {"status": "completed"}

        async def _run():
            result = await integration._tool_execute(workflow_id="wf-1", wait=True)
            assert result.success is True
            mock_client.execute_and_wait.assert_awaited_once()

        asyncio.run(_run())

    def test_tool_execute_error(self):
        integration, mock_client = self._make_integration()
        mock_client.execute_workflow.side_effect = RuntimeError("Connection refused")

        async def _run():
            result = await integration._tool_execute(workflow_id="wf-1")
            assert result.success is False
            assert "Connection refused" in result.error

        asyncio.run(_run())

    def test_tool_execution_status_success(self):
        integration, mock_client = self._make_integration()
        mock_client.get_execution.return_value = {"id": "exec-1", "status": "running"}

        async def _run():
            result = await integration._tool_execution_status(execution_id="exec-1")
            assert result.success is True
            assert result.output["status"] == "running"

        asyncio.run(_run())

    def test_tool_execution_status_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_execution_status(execution_id="exec-1")
            assert result.success is False

        asyncio.run(_run())

    def test_tool_executions_success(self):
        integration, mock_client = self._make_integration()
        mock_client.list_executions.return_value = {"data": [{"id": "e1"}], "total": 1}

        async def _run():
            result = await integration._tool_executions(status="running", limit=5)
            assert result.success is True
            assert result.output["total"] == 1
            mock_client.list_executions.assert_awaited_once_with(page_size=5, status="running")

        asyncio.run(_run())

    def test_tool_approvals_success(self):
        integration, mock_client = self._make_integration()
        mock_client.get_approval_status.return_value = {
            "pending_approvals": [{"token": "abc", "prompt": "Deploy?"}],
            "total_tokens": 1,
        }

        async def _run():
            result = await integration._tool_approvals(execution_id="exec-1")
            assert result.success is True
            assert len(result.output["pending_approvals"]) == 1

        asyncio.run(_run())

    def test_tool_approvals_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_approvals(execution_id="exec-1")
            assert result.success is False

        asyncio.run(_run())

    def test_tool_approve_success(self):
        integration, mock_client = self._make_integration()
        mock_client.resume_execution.return_value = {"status": "approved"}

        async def _run():
            result = await integration._tool_approve(
                execution_id="exec-1", token="tok-1", approve=True
            )
            assert result.success is True
            assert "Approved" in result.error  # ToolResult uses error field for messages
            mock_client.resume_execution.assert_awaited_once_with(
                "exec-1", token="tok-1", approve=True, approved_by="animus", reason=""
            )

        asyncio.run(_run())

    def test_tool_approve_reject(self):
        integration, mock_client = self._make_integration()
        mock_client.resume_execution.return_value = {"status": "rejected"}

        async def _run():
            result = await integration._tool_approve(
                execution_id="exec-1", token="tok-1", approve=False, reason="Not ready"
            )
            assert result.success is True
            assert "Rejected" in result.error

        asyncio.run(_run())

    def test_tool_approve_not_connected(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()

        async def _run():
            result = await integration._tool_approve(
                execution_id="exec-1", token="tok-1", approve=True
            )
            assert result.success is False

        asyncio.run(_run())

    def test_get_tools_includes_execution_tools(self):
        from animus.integrations.gorgon import GorgonIntegration

        integration = GorgonIntegration()
        tools = integration.get_tools()
        tool_names = {t.name for t in tools}
        assert "gorgon_execute" in tool_names
        assert "gorgon_execution_status" in tool_names
        assert "gorgon_executions" in tool_names
        assert "gorgon_approvals" in tool_names
        assert "gorgon_approve" in tool_names
        # Legacy tools still present
        assert "gorgon_delegate" in tool_names
        assert "gorgon_status" in tool_names
        assert len(tools) == 10  # 5 legacy + 5 new
