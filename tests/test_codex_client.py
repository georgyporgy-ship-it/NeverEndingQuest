import io
import json
from types import SimpleNamespace

import pytest

import model_config
from core.ai import api_client
from core.ai.codex_client import (
    CodexAuthenticationError,
    CodexAppServer,
    CodexProvider,
    CodexProtocolError,
    CodexTimeoutError,
    CodexTransportError,
)


MODELS = [
    {
        "model": "account-default",
        "displayName": "Account Default",
        "hidden": False,
        "isDefault": True,
        "defaultReasoningEffort": "medium",
        "supportedReasoningEfforts": [
            {"reasoningEffort": "low"},
            {"reasoningEffort": "medium"},
            {"reasoningEffort": "high"},
        ],
    },
    {
        "model": "account-strong",
        "displayName": "Account Strong",
        "hidden": False,
        "isDefault": False,
        "defaultReasoningEffort": "high",
        "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
    },
]


class FakeRpc:
    installed = True

    def __init__(self, *, authenticated=True, models=None, text='{"ok":true}'):
        self.authenticated = authenticated
        self.models = MODELS if models is None else models
        self.text = text
        self.requests = []
        self.notifications = []
        self.thread_counter = 0
        self.restart_count = 0

    def request(self, method, params=None, timeout=None, ensure_started=True):
        self.requests.append((method, params, timeout))
        if method == "account/read":
            account = (
                {"type": "chatgpt", "email": "player@example.com", "planType": "plus"}
                if self.authenticated else None
            )
            return {"account": account, "requiresOpenaiAuth": True}
        if method == "account/login/start":
            return {
                "type": "chatgptDeviceCode",
                "loginId": "login-1",
                "verificationUrl": "https://auth.openai.com/codex/device",
                "userCode": "ABCD-1234",
            }
        if method == "account/logout":
            self.authenticated = False
            return {}
        if method == "model/list":
            return {"data": self.models, "nextCursor": None}
        if method == "thread/start":
            self.thread_counter += 1
            return {"thread": {"id": f"thread-{self.thread_counter}", "ephemeral": False}}
        if method in {"thread/inject_items", "thread/delete"}:
            return {}
        if method == "turn/start":
            thread_id = params["threadId"]
            turn_id = f"turn-{self.thread_counter}"
            self.notifications = [
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": thread_id,
                        "tokenUsage": {
                            "totalTokenUsage": {
                                "inputTokens": 11,
                                "cachedInputTokens": 3,
                                "outputTokens": 7,
                                "reasoningOutputTokens": 2,
                                "totalTokens": 18,
                            }
                        },
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": thread_id,
                        "item": {"type": "agentMessage", "text": self.text},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id, "status": "completed", "error": None},
                    },
                },
            ]
            return {"turn": {"id": turn_id, "status": "inProgress"}}
        raise AssertionError(f"unexpected method {method}")

    def wait_notification(self, predicate, timeout):
        for notification in self.notifications:
            if predicate(notification):
                return notification
        raise CodexTimeoutError("fake timeout")

    def restart(self):
        self.restart_count += 1

    def stop(self):
        pass


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "user_settings.json"
    monkeypatch.setattr(model_config, "_USER_SETTINGS_FILE", str(path))
    return path


def complete(provider, **overrides):
    kwargs = {
        "messages": [
            {"role": "system", "content": "System context"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Current request"},
        ],
        "tier": "strong",
        "reasoning_effort": "high",
        "response_format": None,
        "response_format_provided": True,
        "timeout": 5,
    }
    kwargs.update(overrides)
    return provider.complete(**kwargs)


def test_authenticated_text_request_is_normalized_and_deleted(isolated_settings):
    rpc = FakeRpc(text="A clean answer")
    result = complete(CodexProvider(rpc))
    assert result["content"] == "A clean answer"
    assert result["model"] == "account-default"
    assert result["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }
    thread = next(params for method, params, _ in rpc.requests if method == "thread/start")
    assert "ephemeral" not in thread
    assert "baseInstructions" not in thread
    assert thread["approvalPolicy"] == "never"
    assert thread["sandbox"] == "read-only"
    injected = next(params for method, params, _ in rpc.requests if method == "thread/inject_items")
    assert injected["items"][0]["role"] == "developer"
    turn = next(params for method, params, _ in rpc.requests if method == "turn/start")
    assert "text_elements" not in turn["input"][0]
    assert any(method == "thread/delete" for method, _, _ in rpc.requests)


def test_each_request_uses_a_new_thread_without_hidden_history(isolated_settings):
    rpc = FakeRpc(text="answer")
    provider = CodexProvider(rpc)
    complete(provider)
    complete(provider, messages=[{"role": "user", "content": "second"}])
    starts = [params for method, params, _ in rpc.requests if method == "thread/start"]
    turns = [params for method, params, _ in rpc.requests if method == "turn/start"]
    assert len(starts) == 2
    assert turns[0]["threadId"] != turns[1]["threadId"]


def test_existing_messages_are_injected_and_final_user_starts_turn(isolated_settings):
    rpc = FakeRpc(text="answer")
    complete(CodexProvider(rpc))
    injected = next(params for method, params, _ in rpc.requests if method == "thread/inject_items")
    assert [item["role"] for item in injected["items"]] == [
        "developer", "developer", "assistant"
    ]
    turn = next(params for method, params, _ in rpc.requests if method == "turn/start")
    assert turn["input"][0]["text"] == "Current request"


def test_strict_schema_and_reasoning_are_translated(isolated_settings):
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    rpc = FakeRpc()
    complete(
        CodexProvider(rpc),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        },
    )
    turn = next(params for method, params, _ in rpc.requests if method == "turn/start")
    assert turn["outputSchema"] == schema
    assert turn["effort"] == "high"
    assert "Return only" not in turn["input"][0]["text"]


def test_manual_model_override_preserves_callsite_reasoning(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "", "balanced": "", "strong": "", "premium": "manual"}
    )
    manual = {
        "model": "manual",
        "displayName": "Manual",
        "hidden": False,
        "isDefault": False,
        "defaultReasoningEffort": "high",
        "supportedReasoningEfforts": [
            {"reasoningEffort": "none"},
            {"reasoningEffort": "high"},
        ],
    }
    rpc = FakeRpc(models=[manual, MODELS[0]], text="manual answer")
    result = complete(
        CodexProvider(rpc),
        tier="premium",
        preferred_model="gpt-5.6-luna",
        reasoning_effort="none",
    )
    turn = next(params for method, params, _ in rpc.requests if method == "turn/start")
    assert result["model"] == "manual"
    assert turn["effort"] == "none"


def test_generic_json_mode_adds_json_instruction_not_empty_schema(isolated_settings):
    rpc = FakeRpc()
    complete(
        CodexProvider(rpc),
        response_format={"type": "json_object"},
    )
    turn = next(params for method, params, _ in rpc.requests if method == "turn/start")
    assert "outputSchema" not in turn
    assert "Return only one valid JSON value" in turn["input"][0]["text"]


def test_unauthenticated_state_fails_before_inference(isolated_settings):
    rpc = FakeRpc(authenticated=False)
    with pytest.raises(CodexAuthenticationError):
        complete(CodexProvider(rpc))
    assert not any(method == "thread/start" for method, _, _ in rpc.requests)


def test_inference_reuses_managed_chatgpt_session_without_forced_refresh(isolated_settings):
    rpc = FakeRpc(text="answer")
    complete(CodexProvider(rpc))
    account_request = next(params for method, params, _ in rpc.requests if method == "account/read")
    assert account_request == {"refreshToken": False}


def test_app_server_unavailable_is_reported_without_start_attempt():
    class UnavailableRpc(FakeRpc):
        installed = False

        def request(self, *args, **kwargs):
            raise AssertionError("status must not start an unavailable app-server")

    status = CodexProvider(UnavailableRpc()).status()
    assert status["installed"] is False
    assert status["authenticated"] is False
    assert "not found" in status["error"].lower()


def test_model_discovery_status_login_and_logout(isolated_settings):
    rpc = FakeRpc()
    provider = CodexProvider(rpc)
    status = provider.status(refresh_models=True)
    assert status["authenticated"] is True
    assert [item["id"] for item in status["models"]] == ["account-default", "account-strong"]
    login = provider.begin_device_login()
    assert login["verification_url"].startswith("https://")
    assert login["user_code"] == "ABCD-1234"
    provider.logout()
    assert rpc.authenticated is False


def test_hidden_models_are_not_exposed(isolated_settings):
    hidden = dict(MODELS[0], model="hidden", hidden=True)
    rpc = FakeRpc(models=[hidden, MODELS[1]])
    assert [item["id"] for item in CodexProvider(rpc).list_models()] == ["account-strong"]


def test_unavailable_route_falls_back_and_records_warning(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "retired", "balanced": "", "strong": "", "premium": ""},
        auto_replace_unavailable=True,
    )
    selected, warning = model_config.select_codex_model("cheap", [
        {"id": "replacement", "display_name": "Replacement", "is_default": True}
    ])
    assert selected == "replacement"
    assert "retired" in warning and "replacement" in warning
    settings = model_config.get_codex_settings()
    assert settings["routes"]["cheap"] == "replacement"
    assert settings["last_fallback"]["unavailable_model"] == "retired"


def test_healthy_route_is_not_auto_upgraded(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "known-good", "balanced": "", "strong": "", "premium": ""},
        auto_replace_unavailable=True,
    )
    selected, warning = model_config.select_codex_model("cheap", [
        {"id": "known-good", "is_default": False},
        {"id": "new-model", "is_default": True},
    ])
    assert selected == "known-good"
    assert warning is None


def test_automatic_route_uses_account_default_not_another_tier(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "", "balanced": "", "strong": "chosen-strong", "premium": ""},
        auto_replace_unavailable=True,
    )
    selected, warning = model_config.select_codex_model("cheap", [
        {"id": "account-default", "is_default": True},
        {"id": "chosen-strong", "is_default": False},
    ])
    assert selected == "account-default"
    assert warning is None


def test_automatic_route_prefers_matching_openai_model(isolated_settings):
    selected, warning = model_config.select_codex_model(
        "cheap",
        [
            {"id": "account-default", "is_default": True},
            {"id": "gpt-5.6-luna", "is_default": False},
        ],
        preferred_model="gpt-5.6-luna",
    )
    assert selected == "gpt-5.6-luna"
    assert warning is None


def test_manual_codex_route_overrides_matching_openai_model(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "manual", "balanced": "", "strong": "", "premium": ""}
    )
    selected, warning = model_config.select_codex_model(
        "cheap",
        [
            {"id": "manual", "is_default": False},
            {"id": "gpt-5.6-luna", "is_default": True},
        ],
        preferred_model="gpt-5.6-luna",
    )
    assert selected == "manual"
    assert warning is None


def test_nearest_configured_tier_is_used_for_retired_model(isolated_settings):
    model_config.persist_codex_routing(
        routes={"cheap": "cheap", "balanced": "retired", "strong": "strong", "premium": "premium"},
        auto_replace_unavailable=False,
    )
    selected, _ = model_config.select_codex_model("balanced", [
        {"id": "cheap"}, {"id": "strong"}, {"id": "premium"}
    ])
    assert selected == "strong"
    assert model_config.get_codex_settings()["routes"]["balanced"] == "retired"


def test_transport_crash_restarts_and_retries(isolated_settings):
    class CrashOnceRpc(FakeRpc):
        def __init__(self):
            super().__init__(text="recovered")
            self.crashed = False

        def request(self, method, params=None, timeout=None, ensure_started=True):
            if method == "thread/start" and not self.crashed:
                self.crashed = True
                raise CodexTransportError("process exited")
            return super().request(method, params, timeout, ensure_started)

    rpc = CrashOnceRpc()
    assert complete(CodexProvider(rpc))["content"] == "recovered"
    assert rpc.restart_count == 1


def test_authentication_loss_during_turn_is_actionable(isolated_settings):
    class AuthLossRpc(FakeRpc):
        def request(self, method, params=None, timeout=None, ensure_started=True):
            result = super().request(method, params, timeout, ensure_started)
            if method == "turn/start":
                self.notifications[-1]["params"]["turn"] = {
                    "id": result["turn"]["id"],
                    "status": "failed",
                    "error": "401 unauthorized",
                }
            return result

    with pytest.raises(CodexAuthenticationError, match="authentication was lost"):
        complete(CodexProvider(AuthLossRpc()))


def test_malformed_catalogue_and_empty_agent_message_fail(isolated_settings):
    with pytest.raises(CodexProtocolError, match="malformed catalogue"):
        CodexProvider(FakeRpc(models="not-a-list")).list_models()

    rpc = FakeRpc(text="")
    with pytest.raises(CodexProtocolError, match="without an agent message"):
        complete(CodexProvider(rpc))


def test_malformed_json_from_app_server_unblocks_notification_waiter():
    class Process:
        stdout = io.StringIO("not-json\n")

    rpc = CodexAppServer(binary="codex")
    rpc._generation = 1
    rpc._read_stdout(Process(), 1)
    with pytest.raises(CodexProtocolError, match="Malformed app-server JSON"):
        rpc.wait_notification(lambda _message: False, timeout=0.1)


def test_first_request_starts_transport_before_registering_waiter(monkeypatch):
    rpc = CodexAppServer(binary="codex")
    events = []

    def start():
        events.append("start")
        assert rpc._pending == {}

    def send(message, ensure_started=True):
        events.append(("send", ensure_started))
        with rpc._pending_lock:
            rpc._pending[message["id"]].put({"id": message["id"], "result": {"ok": True}})

    monkeypatch.setattr(rpc, "start", start)
    monkeypatch.setattr(rpc, "_send", send)

    assert rpc.request("account/read", {"refreshToken": False}) == {"ok": True}
    assert events == ["start", ("send", False)]


def test_timeout_is_reported_without_hidden_retry(isolated_settings):
    class TimeoutRpc(FakeRpc):
        def wait_notification(self, predicate, timeout):
            raise CodexTimeoutError("deadline")

    rpc = TimeoutRpc()
    with pytest.raises(CodexTimeoutError):
        complete(CodexProvider(rpc))
    assert rpc.restart_count == 0


def test_provider_switching_preserves_codex_routes(isolated_settings):
    original = model_config.get_provider()
    routes = {tier: f"model-{tier}" for tier in model_config.CODEX_TIERS}
    try:
        model_config.persist_codex_routing(routes=routes, auto_replace_unavailable=False)
        model_config.set_provider("codex_oauth")
        model_config.set_provider("openai")
        model_config.set_provider("codex_oauth")
        assert model_config.get_codex_settings()["routes"] == routes
    finally:
        model_config.set_provider(original)


def test_codex_automatic_profiles_match_openai_models_efforts_and_retries():
    from model_registry import CALLSITE_BINDINGS

    for task_id, binding in CALLSITE_BINDINGS.items():
        openai_ladder = binding.profiles_for("openai")
        for attempt, profile_name in enumerate(openai_ladder):
            openai_profile = getattr(model_config, profile_name)
            codex_profile = model_config.codex_callsite_config(
                task_id, attempt=attempt
            )
            assert codex_profile["codex_preferred_model"] == openai_profile["model"]
            assert codex_profile["reasoning_effort"] == openai_profile.get(
                "reasoning_effort", "none"
            )

        final_codex_profile = model_config.codex_callsite_config(
            task_id, attempt=len(openai_ladder) + 3
        )
        final_openai_profile = getattr(model_config, openai_ladder[-1])
        assert final_codex_profile["codex_preferred_model"] == final_openai_profile["model"]
        assert final_codex_profile["reasoning_effort"] == final_openai_profile.get(
            "reasoning_effort", "none"
        )


def test_codex_reasoning_is_clamped_only_to_discovered_model_capabilities():
    assert CodexProvider._effort(
        "none", {"supported_reasoning_efforts": ["none", "low", "high"]}
    ) == "none"
    assert CodexProvider._effort(
        "none", {"supported_reasoning_efforts": ["low", "high"]}
    ) == "low"
    assert CodexProvider._effort(
        "high", {"supported_reasoning_efforts": ["low", "medium"]}
    ) == "medium"


@pytest.mark.parametrize("provider", ["openai", "legacy", "lmstudio"])
def test_existing_openai_compatible_provider_route_is_unchanged(monkeypatch, provider):
    seen = {}

    class Message:
        content = "ok"

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Raw:
        choices = [Choice()]
        usage = None
        model = "existing"
        id = "response"

    def fake(messages, model, temperature, selected_provider, response_format, **kwargs):
        seen.update(provider=selected_provider, model=model, kwargs=kwargs)
        return Raw()

    monkeypatch.setattr(api_client, "_openai_completion", fake)
    result = api_client.create_completion(
        [{"role": "user", "content": "hi"}],
        "existing-model",
        _request_provider=provider,
        response_format=None,
    )
    assert result.choices[0].message.content == "ok"
    assert seen["provider"] == provider
    assert seen["model"] == "existing-model"


def test_codex_api_route_returns_openai_shape(monkeypatch, isolated_settings):
    class Provider:
        def complete(self, **kwargs):
            assert kwargs["tier"] == "premium"
            assert kwargs["preferred_model"] == "gpt-5.6-luna"
            assert kwargs["reasoning_effort"] == "none"
            return {
                "content": json.dumps({"answer": 42}),
                "model": "discovered-model",
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                "id": "turn-1",
                "finish_reason": "stop",
            }

    import core.ai.codex_client as codex_client
    monkeypatch.setattr(codex_client, "get_codex_provider", lambda: Provider())
    response = api_client.create_completion(
        [{"role": "user", "content": "hi"}],
        "api-provider-model-that-must-not-leak",
        _request_provider="codex_oauth",
        task_id="T067",
        response_format={"type": "json_object"},
    )
    assert response.choices[0].message.content == '{"answer": 42}'
    assert response.model == "discovered-model"
    assert response.provider == "codex_oauth"


def test_codex_api_route_preserves_openai_retry_effort(monkeypatch, isolated_settings):
    seen = {}

    class Provider:
        def complete(self, **kwargs):
            seen.update(kwargs)
            return {
                "content": "retry",
                "model": kwargs["preferred_model"],
                "usage": {},
                "id": "turn-retry",
                "finish_reason": "stop",
            }

    import core.ai.codex_client as codex_client
    monkeypatch.setattr(codex_client, "get_codex_provider", lambda: Provider())
    api_client.create_completion(
        [{"role": "user", "content": "retry"}],
        "irrelevant-api-model",
        retry_attempt=2,
        _request_provider="codex_oauth",
        task_id="T097",
        response_format=None,
    )
    assert seen["preferred_model"] == "gpt-5.6-luna"
    assert seen["reasoning_effort"] == "medium"


def test_codex_capture_boundary_preserves_retry_rung(monkeypatch, isolated_settings):
    from utils.capture.multi_model_capture import capture_and_fanout

    seen = {}

    class Provider:
        def complete(self, **kwargs):
            seen.update(kwargs)
            return {
                "content": "retry",
                "model": kwargs["preferred_model"],
                "usage": {},
                "id": "turn-retry",
                "finish_reason": "stop",
            }

    import core.ai.codex_client as codex_client
    monkeypatch.setattr(codex_client, "get_codex_provider", lambda: Provider())
    response = capture_and_fanout(
        "T097",
        api_client.create_completion,
        messages=[{"role": "user", "content": "retry"}],
        model="codex-tier:cheap",
        response_format=None,
        _request_provider="codex_oauth",
        _callsite_attempt=2,
    )
    assert response.choices[0].message.content == "retry"
    assert seen["reasoning_effort"] == "medium"


def test_startup_wizard_routes_t092_through_codex(monkeypatch):
    import utils.startup_wizard as startup_wizard

    class Scope:
        @staticmethod
        def is_superseded():
            return False

    seen = {}

    def fake_capture(task_id, target, **kwargs):
        seen.update(task_id=task_id, target=target, kwargs=kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))]
        )

    monkeypatch.setattr(model_config, "MODEL_PROVIDER", "codex_oauth")
    monkeypatch.setattr(startup_wizard, "capture_and_fanout", fake_capture)
    monkeypatch.setattr(startup_wizard, "_emit_startup_phase", lambda _phase: None)
    monkeypatch.setattr(startup_wizard, "status_processing_ai", lambda: None)
    monkeypatch.setattr(startup_wizard, "status_ready", lambda: None)

    result = startup_wizard.get_ai_response(
        [{"role": "user", "content": "start"}],
        persist_response=False,
        live_scope=Scope(),
    )

    assert result == "ready"
    assert seen["task_id"] == "T092"
    assert seen["kwargs"]["_request_provider"] == "codex_oauth"
    assert seen["kwargs"]["model"] == "codex-tier:cheap"
    assert seen["kwargs"]["codex_preferred_model"] == "gpt-5.6-luna"
    assert seen["kwargs"]["reasoning_effort"] == "none"


@pytest.mark.skipif(
    __import__("os").environ.get("NEQ_CODEX_LIVE_TEST") != "1",
    reason="set NEQ_CODEX_LIVE_TEST=1 to use the locally authenticated Codex account",
)
def test_optional_live_codex_smoke():
    from core.ai.codex_client import get_codex_provider

    result = get_codex_provider().complete(
        messages=[{"role": "user", "content": "Return exactly: live-ok"}],
        tier="cheap",
        reasoning_effort="low",
        response_format=None,
        response_format_provided=True,
        timeout=120,
    )
    assert "live-ok" in result["content"].lower()
