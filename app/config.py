from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # Hugging Face model identifier and revision.
    model_id: str = "ai-sage/GigaAM-v3"
    model_revision: str = "e2e_rnnt"

    # Enable FP16 inference on CUDA (recommended for speed/memory).
    fp16_encoder: bool = True

    # Device for model inference.
    # Use "cuda" for GPU (recommended) or "cpu" for CPU-only environments.
    device: str = "cuda"

    # Maximum audio file size accepted by the API (bytes, default 50 MB).
    max_audio_size_bytes: int = 50 * 1024 * 1024

    # Allowed MIME types for uploaded audio files.
    allowed_content_types: list[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/aac",
    ]

    model_config = ConfigDict(env_prefix="STT_", env_file=".env")


settings = Settings()
