from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _is_cache_missing_error(exc: Exception) -> bool:
    try:
        from huggingface_hub import errors as hub_errors

        if isinstance(exc, hub_errors.LocalEntryNotFoundError):
            return True
    except Exception:
        pass
    message = str(exc)
    return (
        "LocalEntryNotFoundError" in message
        or "Cannot find the requested files in the disk cache" in message
        or "couldn't connect" in message
    )


class ModelLoader:
    """Load the Hugging Face model and configure device/runtime helpers."""

    def __init__(self, model_id: str, model_revision: str, device: str) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._requested_device = device

    def load(self):
        import torch
        from transformers import AutoModel

        device = self._requested_device
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

        logger.info(
            "Loading Hugging Face model '%s' (revision '%s') on device '%s' …",
            self._model_id,
            self._model_revision,
            device,
        )

        if hasattr(torch.serialization, "add_safe_globals"):
            try:
                from pyannote.audio.core.task import Problem, Resolution, Specifications

                torch.serialization.add_safe_globals(
                    [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
                )
            except Exception:  # pragma: no cover - pyannote may not be available yet
                torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

        local_only = _is_truthy(os.environ.get("HF_HUB_OFFLINE")) or _is_truthy(
            os.environ.get("TRANSFORMERS_OFFLINE")
        )

        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "revision": self._model_revision,
            "local_files_only": local_only,
        }

        try:
            model = AutoModel.from_pretrained(self._model_id, **load_kwargs)
        except Exception as exc:
            if local_only and _is_cache_missing_error(exc):
                hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
                raise RuntimeError(
                    "Offline mode is enabled, but the model files are missing from the cache. "
                    "Run the service once with internet access to populate the cache, or set "
                    f"STT_MODEL_ID to a local path. HF_HOME={hf_home}."
                ) from exc
            raise
        try:
            model.to(device)
        except Exception:  # pragma: no cover - depends on model implementation
            logger.warning("Model does not support .to(%s); using default device.", device)

        logger.info("Hugging Face GigaAM model loaded successfully.")
        return model, device

