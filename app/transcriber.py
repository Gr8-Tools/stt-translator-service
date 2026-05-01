"""GigaAM v3 model wrapper (Hugging Face).

Loads the model once at startup via ``transformers.AutoModel`` and exposes a
single ``transcribe`` method that accepts raw audio bytes and returns the
recognised text.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def _no_meta_device():
    """Patch ``init_empty_weights`` to a no-op for the duration of the block.

    GigaAM's remote code (``modeling_gigaam.py``) instantiates
    ``FeatureExtractor`` and other Hydra components that create real CPU tensors
    inside ``__init__``.  In ``transformers >= 4.40`` (and especially >= 4.57)
    ``from_pretrained`` may still activate ``accelerate``'s
    ``init_empty_weights`` context manager even when ``low_cpu_mem_usage=False``
    is passed, placing the model skeleton on the *meta* device.  When the
    custom code then creates a normal CPU tensor the two worlds collide and
    accelerate raises:

        RuntimeError: Tensor on device cpu is not on the expected device meta!

    Replacing ``init_empty_weights`` with a no-op ensures all tensors are
    allocated normally on CPU; the model is then moved to the target device
    after ``from_pretrained`` returns.
    """

    @contextmanager
    def _noop(*_args, **_kwargs):  # type: ignore[misc]
        yield

    import importlib  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # Patch accelerate in all known module locations across versions.
    accel_modules = []
    for module_name in (
        "accelerate",
        "accelerate.utils.modeling",
        "accelerate.big_modeling",
        "accelerate.utils",
    ):
        try:
            accel_modules.append(importlib.import_module(module_name))
        except ImportError:
            continue

    # Optionally patch transformers as well (location changed across versions)
    _tmu = None
    _orig_tmu = None
    try:
        import transformers.modeling_utils as _tmu_mod  # noqa: PLC0415
        _tmu = _tmu_mod
        _orig_tmu = getattr(_tmu_mod, "init_empty_weights", None)
    except ImportError:
        _tmu_mod = None  # noqa: F841

    accel_originals: dict[object, object] = {}
    for mod in accel_modules:
        original = getattr(mod, "init_empty_weights", None)
        if original is not None:
            accel_originals[mod] = original

    # Force default device to CPU while loading to avoid meta defaults.
    _orig_default_device = None
    if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
        try:
            _orig_default_device = torch.get_default_device()
            torch.set_default_device("cpu")
        except Exception:  # pragma: no cover - defensive for old/odd torch builds
            _orig_default_device = None

    try:
        for mod in accel_originals:
            mod.init_empty_weights = _noop  # type: ignore[assignment]
        if _tmu is not None and _orig_tmu is not None:
            _tmu.init_empty_weights = _noop  # type: ignore[assignment]
        yield
    finally:
        for mod, original in accel_originals.items():
            mod.init_empty_weights = original  # type: ignore[assignment]
        if _tmu is not None and _orig_tmu is not None:
            _tmu.init_empty_weights = _orig_tmu  # type: ignore[assignment]
        if _orig_default_device is not None:
            try:
                torch.set_default_device(_orig_default_device)
            except Exception:  # pragma: no cover - defensive restore
                pass


class Transcriber:
    """Singleton wrapper around the GigaAM v3 model loaded from Hugging Face."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        from transformers import AutoModel  # type: ignore[import]
        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415

        logger.info(
            "Loading GigaAM model '%s' (revision '%s') on device '%s' …",
            settings.model_repo,
            settings.model_revision,
            settings.device,
        )
        logger.info(
            "Torch %s | Torchaudio %s | CUDA available: %s | device count: %s",
            torch.__version__,
            torchaudio.__version__,
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "STT_DEVICE=%s requested, but CUDA is not available. "
                "Set STT_DEVICE=cpu or ensure GPU runtime is enabled.",
                settings.device,
            )
        # _no_meta_device() patches init_empty_weights to a no-op so that
        # GigaAM's Hydra-instantiated components (FeatureExtractor, etc.) can
        # create real CPU tensors without conflicting with accelerate's
        # meta-device bookkeeping.  After loading the model is moved to the
        # requested device explicitly.
        with _no_meta_device():
            self._model = AutoModel.from_pretrained(
                settings.model_repo,
                revision=settings.model_revision,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
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
