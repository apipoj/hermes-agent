"""Timeout tuning must not turn a built-in provider into a custom endpoint."""

from hermes_cli.config import load_config_readonly
from hermes_cli.model_switch import switch_model
from hermes_cli.providers import resolve_provider_full, resolve_user_provider
from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout


def test_codex_switch_uses_native_catalog_with_timeout_only_config(tmp_path, monkeypatch):
    """The gateway/CLI switch must never probe ChatGPT's nonexistent /models endpoint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: 9router\n  default: cx/gpt-6-sol\n"
        "providers:\n  openai-codex:\n    request_timeout_seconds: 1800\n"
        "    stale_timeout_seconds: 900\n"
    )
    configured = load_config_readonly()["providers"]

    def codex_runtime(*_args, **_kwargs):
        return {
            "provider": "openai-codex",
            "api_key": "test-only-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        }

    def no_custom_probe(*_args, **_kwargs):
        raise AssertionError("Codex subscription must not request /models")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", codex_runtime)
    monkeypatch.setattr("hermes_cli.models.probe_api_models", no_custom_probe)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *_a, **_kw: None)
    result = switch_model(
        raw_input="gpt-6-sol", current_provider="9router", current_model="cx/gpt-6-sol",
        explicit_provider="openai-codex", user_providers=configured,
    )
    assert result.success, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.api_mode == "codex_responses"
    assert result.base_url == "https://chatgpt.com/backend-api/codex"


def test_timeout_only_codex_entry_keeps_builtin_route_and_timeouts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "providers:\n  openai-codex:\n    request_timeout_seconds: 1800\n"
        "    stale_timeout_seconds: 900\n"
    )
    configured = {
        "openai-codex": {"request_timeout_seconds": 1800, "stale_timeout_seconds": 900}
    }
    assert resolve_user_provider("openai-codex", configured) is None
    provider = resolve_provider_full("openai-codex", configured)
    assert provider is not None
    assert provider.source != "user-config"
    assert get_provider_request_timeout("openai-codex", "gpt-6-sol") == 1800
    assert get_provider_stale_timeout("openai-codex", "gpt-6-sol") == 900


def test_timeout_only_bedrock_entry_keeps_native_transport():
    configured = {"bedrock": {"stale_timeout_seconds": 600}}
    provider = resolve_provider_full("bedrock", configured)
    assert provider is not None
    assert provider.transport == "bedrock_converse"
    assert provider.auth_type == "aws_sdk"


def test_configured_endpoint_still_overrides_builtin():
    configured = {
        "openai-codex": {
            "base_url": "https://example.test/v1",
            "key_env": "EXAMPLE_KEY",
        }
    }
    provider = resolve_user_provider("openai-codex", configured)
    assert provider is not None
    assert provider.source == "user-config"
    assert provider.base_url == "https://example.test/v1"
