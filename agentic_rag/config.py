"""Environment-only configuration with explicit provider validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised for incomplete or unsafe runtime configuration."""


class LLMProvider(StrEnum):
    OPENAI = "openai"
    AZURE = "azure"


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    provider: LLMProvider
    knowledge_base_path: Path
    model: str = "gpt-5-mini"
    max_results: int = 5
    min_score: float = 0.15
    min_query_coverage: float = 0.40
    enable_tracing: bool = False
    api_key: str | None = field(default=None, repr=False)
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str | None = None

    @classmethod
    def from_env(cls, *, require_credentials: bool = True) -> Settings:
        load_dotenv()
        raw_provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value).casefold()
        try:
            provider = LLMProvider(raw_provider)
        except ValueError as exc:
            raise ConfigurationError("LLM_PROVIDER must be 'openai' or 'azure'") from exc

        try:
            max_results = int(os.getenv("RETRIEVAL_MAX_RESULTS", "5"))
            min_score = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.15"))
            min_query_coverage = float(os.getenv("RETRIEVAL_MIN_COVERAGE", "0.40"))
        except ValueError as exc:
            raise ConfigurationError("Retrieval settings must be numeric") from exc

        knowledge_base = Path(os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.txt"))
        enable_tracing = _as_bool(os.getenv("ENABLE_TRACING"), default=False)

        if provider is LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if require_credentials and not api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY is not set. Export it locally or use --retrieval-only."
                )
            return cls(
                provider=provider,
                knowledge_base_path=knowledge_base,
                model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                max_results=max_results,
                min_score=min_score,
                min_query_coverage=min_query_coverage,
                enable_tracing=enable_tracing,
                api_key=api_key,
            )

        azure_values = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        }
        missing = [name for name, value in azure_values.items() if not value]
        if require_credentials and missing:
            raise ConfigurationError(
                "Azure configuration is incomplete. Missing: " + ", ".join(missing)
            )
        return cls(
            provider=provider,
            knowledge_base_path=knowledge_base,
            model=azure_values["AZURE_OPENAI_DEPLOYMENT"] or "gpt-5-mini",
            max_results=max_results,
            min_score=min_score,
            min_query_coverage=min_query_coverage,
            enable_tracing=enable_tracing,
            api_key=azure_values["AZURE_OPENAI_API_KEY"],
            azure_endpoint=azure_values["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=azure_values["AZURE_OPENAI_DEPLOYMENT"],
            azure_api_version=azure_values["AZURE_OPENAI_API_VERSION"],
        )
