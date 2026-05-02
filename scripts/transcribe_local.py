"""Simple CLI to transcribe a single audio file using the HF GigaAM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.transcriber import Transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a local audio file.")
    parser.add_argument("--file", required=True, help="Path to the audio file")
    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()
    transcriber = Transcriber.get_instance()
    text = transcriber.transcribe(audio_bytes, filename=audio_path.name)
    print(text)


if __name__ == "__main__":
    main()

