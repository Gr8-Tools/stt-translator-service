from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class WavConverter:
    """Convert arbitrary audio files to WAV using ffmpeg."""

    def ensure_wav(self, input_path: Path) -> Path:
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

