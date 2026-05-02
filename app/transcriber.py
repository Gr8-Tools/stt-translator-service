"""Hugging Face GigaAM v3 model wrapper.

Loads the model once at startup and exposes a single ``transcribe`` method
that accepts raw audio bytes and returns recognised text.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class MissingHfTokenError(RuntimeError):
    """Raised when longform transcription requires an HF token but none is set."""


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
                # Avoid reproducibility warning and improve matmul perf on recent GPUs.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self._device = device

        logger.info(
            "Loading Hugging Face model '%s' (revision '%s') on device '%s' …",
            settings.model_id,
            settings.model_revision,
            device,
        )

        logger.info("Model is loaded. Configuring...")

        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "revision": settings.model_revision,
        }

        if hasattr(torch.serialization, "add_safe_globals"):
            try:
                from pyannote.audio.core.task import Problem, Resolution, Specifications

                torch.serialization.add_safe_globals(
                    [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
                )
            except Exception:  # pragma: no cover - pyannote may not be available yet
                torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

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
            wav_path = self._ensure_wav(Path(tmp.name))
            try:
                result = self._model.transcribe(str(wav_path))
            except ValueError as exc:
                if "Too long wav file" in str(exc) and hasattr(self._model, "transcribe_longform"):
                    logger.info("Falling back to longform transcription for '%s'.", filename)
                    self._ensure_hf_token()
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

    @staticmethod
    def _ensure_wav(input_path: Path) -> Path:
        if input_path.suffix.lower() == ".wav":
            return input_path
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to convert audio to WAV but was not found in PATH.")
        output_path = input_path.with_suffix(".wav")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    @staticmethod
    def _ensure_hf_token() -> None:
        if os.environ.get("HF_TOKEN"):
            return
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token
            return
        raise MissingHfTokenError(
            "HF_TOKEN environment variable is not set; it is required for longform transcription."
        )

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
