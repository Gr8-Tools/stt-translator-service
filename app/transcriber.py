"""GigaAM v3 model wrapper.

Loads the model once at startup and exposes a single ``transcribe`` method
that accepts a file-like object with raw audio bytes and returns the
recognised text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """Singleton wrapper around the GigaAM v3 e2e_rnnt model."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        import gigaam  # type: ignore[import]  # imported lazily to allow mocking in tests

        logger.info(
            "Loading GigaAM model '%s' on device '%s' …",
            settings.model_name,
            settings.device,
        )
        self._model = gigaam.load_model(
            settings.model_name,
            device=settings.device
        )
        logger.info("GigaAM model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, filename: str = "audio") -> str:
        """Transcribe raw audio bytes and return the recognised text.

        The audio is written to a temporary file so that GigaAM's internal
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
        return cls._instance


def get_transcriber() -> Transcriber:
    """FastAPI dependency that returns the shared :class:`Transcriber` instance."""
    return Transcriber.get_instance()
