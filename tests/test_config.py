from pathlib import Path

import pytest

from agentic_rag.config import ConfigurationError, LLMProvider, Settings

_ENVIRONMENT_KEYS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "KNOWLEDGE_BASE_PATH",
    "RETRIEVAL_MAX_RESULTS",
    "RETRIEVAL_MIN_SCORE",
    "RETRIEVAL_MIN_COVERAGE",
    "ENABLE_TRACING",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit tests must not inherit credentials from a developer's local .env file.
    monkeypatch.setattr("agentic_rag.config.load_dotenv", lambda: False)
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_openai_defaults_work_without_key_for_retrieval_only() -> None:
    settings = Settings.from_env(require_credentials=False)
    assert settings.provider is LLMProvider.OPENAI
    assert settings.model == "gpt-5-mini"
    assert settings.knowledge_base_path == Path("knowledge_base.txt")
    assert settings.min_query_coverage == 0.40


def test_openai_key_is_required_for_live_run() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env(require_credentials=True)


def test_azure_requires_every_provider_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        Settings.from_env(require_credentials=True)


def test_complete_azure_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    settings = Settings.from_env()
    assert settings.provider is LLMProvider.AZURE
    assert settings.azure_deployment == "gpt-5-mini"
    assert "secret" not in repr(settings)


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ConfigurationError, match=r"openai.*azure"):
        Settings.from_env(require_credentials=False)


def test_invalid_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_TRACING", "sometimes")
    with pytest.raises(ConfigurationError, match="boolean"):
        Settings.from_env(require_credentials=False)
