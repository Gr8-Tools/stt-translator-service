"""FastAPI application — Speech-to-Text service backed by GigaAM v3 e2e_rnnt."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import ErrorResponse, TranscriptionResponse
from app.transcriber import MissingHfTokenError, Transcriber, get_transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="STT Translator Service",
    description=(
        "A simple Speech-to-Text backend that accepts audio files and returns "
        "the transcribed text using the Hugging Face **GigaAM v3** model."
    ),
    version="1.0.0",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"], summary="Service health-check")
async def health() -> dict[str, str]:
    """Return a simple liveness signal."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    tags=["transcription"],
    summary="Transcribe an audio file to text",
    status_code=status.HTTP_200_OK,
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    transcriber: Transcriber = Depends(get_transcriber),
) -> TranscriptionResponse:
    """Accept an uploaded audio file and return its transcription.

    Supported formats: WAV, MP3, OGG, FLAC, AAC.

    The model runs on GPU (CUDA) by default; set the ``STT_DEVICE`` environment
    variable to ``cpu`` to force CPU inference.
    """
    # --- validate content-type -------------------------------------------
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Allowed types: {', '.join(settings.allowed_content_types)}"
            ),
        )

    # --- read & size-guard -----------------------------------------------
    audio_bytes = await file.read()
    if len(audio_bytes) > settings.max_audio_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(audio_bytes)} bytes exceeds the maximum allowed "
                f"size of {settings.max_audio_size_bytes} bytes."
            ),
        )
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- transcribe -------------------------------------------------------
    filename = file.filename or "audio"
    logger.info("Transcribing file '%s' (%d bytes) …", filename, len(audio_bytes))
    try:
        text = transcriber.transcribe(audio_bytes, filename=filename)
    except MissingHfTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        if "Too long wav file" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Audio is too long for short-form transcription. "
                    "Please use longform transcription or split the audio."
                ),
            ) from exc
        raise
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription error: {exc}",
        ) from exc

    logger.info("Transcription result: %r", text)
    return TranscriptionResponse(text=text)
