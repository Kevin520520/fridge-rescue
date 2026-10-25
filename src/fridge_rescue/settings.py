"""Environment-backed settings for Fridge Rescue.

Secrets are read from the local environment or .env file and are never logged.
"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """External services and local path configuration."""

    llm_provider: Literal["deepseek", "openai"] = "deepseek"
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str | None = None
    vision_model: str | None = None
    source_dataset_file: str = "epicurious_recipes.csv"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        """Accept harmless case/whitespace variations but reject unknown providers."""

        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("deepseek_model", "openai_model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("LLM model name must not be blank")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def deepseek_url_must_not_be_blank(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("DeepSeek base URL must not be blank")
        return value

    @field_validator("openai_base_url", "vision_model")
    @classmethod
    def normalize_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().rstrip("/") or None

    @property
    def raw_recipe_path(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / self.source_dataset_file

    @property
    def active_provider(self) -> Literal["deepseek", "openai"]:
        """Provider selected for both the agentic and baseline routes."""

        return self.llm_provider

    @property
    def active_api_key(self) -> SecretStr | None:
        """Return the selected provider's secret without converting it to plain text."""

        if self.active_provider == "deepseek":
            return self.deepseek_api_key
        return self.openai_api_key

    @property
    def active_model(self) -> str:
        if self.active_provider == "deepseek":
            return self.deepseek_model
        return self.openai_model

    @property
    def active_base_url(self) -> str | None:
        if self.active_provider == "deepseek":
            return self.deepseek_base_url
        return self.openai_base_url

    @property
    def active_vision_model(self) -> str:
        """Reuse the active provider's key with a vision-capable model."""

        if self.vision_model is not None:
            return self.vision_model
        if self.active_provider == "deepseek":
            return "deepseek-flash"
        return self.openai_model

    @property
    def active_key_env_name(self) -> str:
        if self.active_provider == "deepseek":
            return "DEEPSEEK_API_KEY"
        return "OPENAI_API_KEY"

    @property
    def has_valid_llm_key(self) -> bool:
        key = self.active_api_key
        if key is None:
            return False
        value = key.get_secret_value().strip()
        return bool(value and not value.startswith("replace_"))

def get_settings() -> Settings:
    """Load settings lazily from the active environment."""

    return Settings()
