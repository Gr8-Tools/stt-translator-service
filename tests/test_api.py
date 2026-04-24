"""Tests for the STT Translator Service API.

The GigaAM model is **mocked** so that these tests run without a GPU and
without downloading any model weights.
"""

from __future__ import annotations

import io
import struct
import wave
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.transcriber import Transcriber, get_transcriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid PCM WAV file in memory."""
    n_frames = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


SAMPLE_WAV = _make_wav_bytes()
MOCK_TRANSCRIPTION = "Тестовая транскрипция"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_transcriber():
    """Replace the Transcriber dependency with a mock for every test."""
    mock = MagicMock(spec=Transcriber)
    mock.transcribe.return_value = MOCK_TRANSCRIPTION
    app.dependency_overrides[get_transcriber] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Transcription — happy path
# ---------------------------------------------------------------------------


class TestTranscribeHappyPath:
    def test_wav_file_returns_text(self, client: TestClient, mock_transcriber) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", SAMPLE_WAV, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == MOCK_TRANSCRIPTION
        mock_transcriber.transcribe.assert_called_once()

    def test_transcriber_receives_correct_bytes(
        self, client: TestClient, mock_transcriber: MagicMock
    ) -> None:
        client.post(
            "/transcribe",
            files={"file": ("speech.wav", SAMPLE_WAV, "audio/wav")},
        )
        call_args = mock_transcriber.transcribe.call_args
        assert call_args.kwargs.get("audio_bytes") == SAMPLE_WAV or (
            len(call_args.args) > 0 and call_args.args[0] == SAMPLE_WAV
        )

    def test_mp3_content_type_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("audio.mp3", b"\xff\xfb\x90\x00" * 10, "audio/mpeg")},
        )
        assert response.status_code == 200

    def test_ogg_content_type_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("audio.ogg", b"OggS" + b"\x00" * 20, "audio/ogg")},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Transcription — validation errors
# ---------------------------------------------------------------------------


class TestTranscribeValidation:
    def test_empty_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_unsupported_content_type_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("video.mp4", b"\x00" * 100, "video/mp4")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_file_too_large_returns_413(self, client: TestClient) -> None:
        from app.config import settings

        oversized = b"\x00" * (settings.max_audio_size_bytes + 1)
        response = client.post(
            "/transcribe",
            files={"file": ("big.wav", oversized, "audio/wav")},
        )
        assert response.status_code == 413

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/transcribe")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Transcription — model error propagation
# ---------------------------------------------------------------------------


class TestTranscribeModelError:
    def test_model_exception_returns_500(
        self, client: TestClient, mock_transcriber
    ) -> None:
        mock_transcriber.transcribe.side_effect = RuntimeError("CUDA OOM")
        response = client.post(
            "/transcribe",
            files={"file": ("audio.wav", SAMPLE_WAV, "audio/wav")},
        )
        assert response.status_code == 500
        assert "CUDA OOM" in response.json()["detail"]
