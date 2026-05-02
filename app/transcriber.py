"""Hugging Face GigaAM v3 model wrapper.

Loads the model once at startup and exposes a single ``transcribe`` method
that accepts raw audio bytes and returns recognised text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """Singleton wrapper around the GigaAM v3 Hugging Face model."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel

        device = settings.device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available; falling back to CPU.")
                device = "cpu"
            else:
                count = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(count)]
                logger.info("CUDA devices available (%d): %s", count, ", ".join(names))

        logger.info(
            "Loading Hugging Face model '%s' (revision '%s') on device '%s' …",
            settings.model_id,
            settings.model_revision,
            device,
        )

        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "revision": settings.model_revision,
        }
        if device.startswith("cuda") and settings.fp16_encoder:
            load_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModel.from_pretrained(settings.model_id, **load_kwargs)
        try:
            self._model.to(device)
        except Exception:  # pragma: no cover - depends on model implementation
            logger.warning("Model does not support .to(%s); using default device.", device)

        logger.info("Hugging Face GigaAM model loaded successfully.")

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
            result = self._model.transcribe(tmp.name)

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
