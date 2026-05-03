from __future__ import annotations

import os

from app.exceptions import MissingHfTokenError


class TokenValidator:
    """Ensure Hugging Face auth token is present when required."""

    def __init__(self, hf_token: str | None) -> None:
        self._hf_token = hf_token

    def ensure_hf_token(self) -> None:
        if os.environ.get("HF_TOKEN"):
            return
        if self._hf_token:
            os.environ["HF_TOKEN"] = self._hf_token
            return
        raise MissingHfTokenError(
            "HF_TOKEN environment variable is not set; it is required for longform transcription."
        )

