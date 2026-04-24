from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Successful transcription result."""

    text: str

    model_config = {"json_schema_extra": {"examples": [{"text": "Привет, как дела?"}]}}


class ErrorResponse(BaseModel):
    """Error details returned by the API."""

    detail: str

    model_config = {
        "json_schema_extra": {"examples": [{"detail": "Unsupported audio format"}]}
    }
