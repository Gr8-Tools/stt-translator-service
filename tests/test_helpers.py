"""Tests for helper utilities (WavConverter, TokenValidator)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.exceptions import MissingHfTokenError
from app.token_validator import TokenValidator
from app.wav_converter import WavConverter


def test_wav_converter_passthrough(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")

    converter = WavConverter()
    assert converter.ensure_wav(wav_path) == wav_path


def test_wav_converter_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.wav_converter.shutil.which", lambda _: None)
    converter = WavConverter()

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        converter.ensure_wav(tmp_path / "sample.mp3")


def test_wav_converter_invokes_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.wav_converter.shutil.which", lambda _: "ffmpeg")
    run_mock = MagicMock()
    monkeypatch.setattr("app.wav_converter.subprocess.run", run_mock)

    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"mp3")

    converter = WavConverter()
    output_path = converter.ensure_wav(input_path)

    assert output_path == tmp_path / "sample.wav"
    run_mock.assert_called_once()


def test_token_validator_uses_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env_token")
    validator = TokenValidator(hf_token=None)
    validator.ensure_hf_token()


def test_token_validator_sets_env_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    validator = TokenValidator(hf_token="settings_token")
    validator.ensure_hf_token()
    assert __import__("os").environ.get("HF_TOKEN") == "settings_token"


def test_token_validator_raises_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    validator = TokenValidator(hf_token=None)

    with pytest.raises(MissingHfTokenError):
        validator.ensure_hf_token()
