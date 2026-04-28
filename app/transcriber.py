"""GigaAM v3 model wrapper (Hugging Face).

Loads the model once at startup via ``transformers.AutoModel`` and exposes a
single ``transcribe`` method that accepts raw audio bytes and returns the
recognised text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """Singleton wrapper around the GigaAM v3 model loaded from Hugging Face."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        from transformers import AutoModel  # type: ignore[import]  # imported lazily to allow mocking in tests

        logger.info(
            "Loading GigaAM model '%s' (revision '%s') on device '%s' …",
            settings.model_repo,
            settings.model_revision,
            settings.device,
        )
        self._model = AutoModel.from_pretrained(
            settings.model_repo,
            revision=settings.model_revision,
            trust_remote_code=True,
        )
        self._model = self._model.to(settings.device)
        self._model.eval()
        logger.info("GigaAM model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, filename: str = "audio") -> str:
        """Transcribe raw audio bytes and return the recognised text.

        The audio is written to a temporary file so that the model's internal
        audio-loading pipeline can determine the codec from the file extension
        when it is present in *filename*.
        """
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result: str = self._model.transcribe(tmp.name)
        return result.strip()

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "Transcriber":
        if cls._instance is None:
            cls._instance = cls()
        assert cls._instance is not None
        return cls._instance


def get_transcriber() -> Transcriber:
    """FastAPI dependency that returns the shared :class:`Transcriber` instance."""
    return Transcriber.get_instance()
