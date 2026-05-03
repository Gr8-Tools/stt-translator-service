"""Hugging Face GigaAM v3 model wrapper.

Loads the model once at startup and exposes a single ``transcribe`` method
that accepts raw audio bytes and returns recognised text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import settings
from app.model_loader import ModelLoader
from app.token_validator import TokenValidator
from app.wav_converter import WavConverter

logger = logging.getLogger(__name__)


class Transcriber:
    """Singleton wrapper around the GigaAM v3 Hugging Face model."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        loader = ModelLoader(settings.model_id, settings.model_revision, settings.device)
        self._model, self._device = loader.load()
        self._wav_converter = WavConverter()
        self._token_validator = TokenValidator(settings.hf_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, filename: str = "audio") -> str:
        """Transcribe raw audio bytes and return the recognised text.

        The audio is written to a temporary file so that the internal
        audio-loading pipeline can determine the codec from the file extension.
        """
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            wav_path = self._wav_converter.ensure_wav(Path(tmp.name))
            try:
                result = self._model.transcribe(str(wav_path))
            except ValueError as exc:
                if "Too long wav file" in str(exc) and hasattr(self._model, "transcribe_longform"):
                    logger.info("Falling back to longform transcription for '%s'.", filename)
                    self._token_validator.ensure_hf_token()
                    result = self._model.transcribe_longform(str(wav_path))
                else:
                    raise
            finally:
                if wav_path != Path(tmp.name):
                    wav_path.unlink(missing_ok=True)

        return self._extract_text(result)

    @staticmethod
    def _extract_text(result: object) -> str:
        if isinstance(result, dict) and "text" in result:
            text = result["text"]
        else:
            text = str(result)
        return text.strip()


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
