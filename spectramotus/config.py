"""Settings: API keys and repo paths. Values come from the environment and the repo-root `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]

# LivePortrait on Apple Silicon needs CPU fallback for a few unsupported MPS ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    fal_key: str | None = Field(default=None, alias="FAL_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    runwayml_api_secret: str | None = Field(default=None, alias="RUNWAYML_API_SECRET")
    huggingface_api_key: str | None = Field(default=None, alias="HUGGINGFACE_API_KEY")
    clipdrop_api_key: str | None = Field(default=None, alias="CLIPDROP_API_KEY")

    budget_usd: float = Field(default=8.0, alias="SPECTRAMOTUS_BUDGET_USD")
    canvas: str = Field(default="portrait", alias="SPECTRAMOTUS_CANVAS")

    work_dir: Path = REPO_ROOT / "work"
    packs_dir: Path = REPO_ROOT / "packs"
    models_dir: Path = REPO_ROOT / "models"
    weights_dir: Path = REPO_ROOT / "pretrained_weights"
    drivers_dir: Path = REPO_ROOT / "driving-performances"
    ledger_path: Path = REPO_ROOT / "ledger.sqlite"

    def key_status(self) -> dict[str, bool]:
        return {
            "FAL_KEY": bool(self.fal_key),
            "ANTHROPIC_API_KEY": bool(self.anthropic_api_key),
            "GOOGLE_API_KEY": bool(self.google_api_key),
            "RUNWAYML_API_SECRET": bool(self.runwayml_api_secret),
        }


settings = Settings()
