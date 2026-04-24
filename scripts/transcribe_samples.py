"""Send sample audio files to the running STT service and print transcriptions.

Usage:
  python scripts/transcribe_samples.py --url http://localhost:8000 --dir .resources

The server must be running (e.g. via `docker compose up --build`).
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

import httpx


DEFAULT_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in DEFAULT_MIME_BY_EXT:
        return DEFAULT_MIME_BY_EXT[ext]
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio samples via the STT service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--dir", default=".resources", help="Directory with audio samples")
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern inside --dir (default: '*')",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds (default: 600)",
    )

    args = parser.parse_args(argv)

    base_url = args.url.rstrip("/")
    samples_dir = Path(args.dir)
    if not samples_dir.exists() or not samples_dir.is_dir():
        print(f"Samples directory not found: {samples_dir}", file=sys.stderr)
        return 2

    files = sorted([p for p in samples_dir.glob(args.pattern) if p.is_file()])
    if not files:
        print(f"No files matched: {samples_dir / args.pattern}", file=sys.stderr)
        return 2

    timeout = httpx.Timeout(args.timeout, connect=30.0)

    ok = 0
    failed = 0

    with httpx.Client(timeout=timeout) as client:
        # Quick health check
        try:
            r = client.get(f"{base_url}/health")
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Health-check failed: {exc}", file=sys.stderr)
            return 1

        for path in files:
            mime = _guess_mime(path)
            print(f"\n==> {path.name} ({mime})")
            data = path.read_bytes()
            try:
                resp = client.post(
                    f"{base_url}/transcribe",
                    files={"file": (path.name, data, mime)},
                )
                resp.raise_for_status()
                payload = resp.json()
                print(payload.get("text", payload))
                ok += 1
            except httpx.HTTPStatusError as exc:
                failed += 1
                body = exc.response.text
                print(f"HTTP {exc.response.status_code}: {body}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"Request failed: {exc}", file=sys.stderr)

    print(f"\nDone. ok={ok}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

