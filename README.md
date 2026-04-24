# stt-translator-service

A lightweight Python backend that receives audio files and returns their transcription using
**[GigaAM v3 e2e\_rnnt](https://github.com/salute-developers/GigaAM)** — Sber's GPU-accelerated
end-to-end RNN-T speech recognition model.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Web framework | **FastAPI** |
| ASGI server | **Uvicorn** |
| STT model | **GigaAM v3 e2e\_rnnt** |
| Audio I/O | **ffmpeg** (via GigaAM's internal pipeline) |
| Containerisation | **Docker** + **NVIDIA Container Toolkit** |

---

## Project structure

```
stt-translator-service/
├── app/
│   ├── config.py        # Pydantic-Settings configuration
│   ├── main.py          # FastAPI app & endpoints
│   ├── schemas.py       # Pydantic request/response models
│   └── transcriber.py   # GigaAM model wrapper (singleton)
├── tests/
│   └── test_api.py      # pytest test suite (model is mocked)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick start

### 1 — Docker (recommended, requires GPU + NVIDIA Container Toolkit)

```bash
docker compose up --build
```

You can override runtime settings without editing `docker-compose.yml`:

```bash
# Example: force CPU inference
STT_DEVICE=cpu docker compose up --build
```

The service will be available at `http://localhost:8000`.

#### Try the bundled sample audio files

The repository includes a `.resources/` folder with a couple of small audio files.
After the service is running, you can send them to the API.

**Option A (PowerShell, using `curl.exe`):**

```powershell
curl.exe -X POST http://localhost:8000/transcribe -F "file=@.resources\test_wav.wav;type=audio/wav"
curl.exe -X POST http://localhost:8000/transcribe -F "file=@.resources\test_mp3.mp3;type=audio/mpeg"
```

**Option B (Python helper script):**

```bash
python scripts/transcribe_samples.py --url http://localhost:8000 --dir .resources
```

### 2 — Local (Python ≥ 3.11)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (CPU mode for testing without a GPU)
STT_DEVICE=cpu uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Note: on Windows, `gigaam` may not install due to native dependencies.
> For real transcription, prefer Docker/WSL2 (Linux). The test suite still runs
> on any platform because it mocks the model.

---

## API reference

### `GET /health`

Liveness probe.

```json
{"status": "ok"}
```

### `POST /transcribe`

Transcribe an uploaded audio file.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | binary | Audio file (WAV, MP3, OGG, FLAC, AAC) |

**Response 200**

```json
{"text": "Привет, как дела?"}
```

**Error responses**

| Status | Cause |
|--------|-------|
| 400 | Empty file or unsupported content type |
| 413 | File exceeds 50 MB limit |
| 422 | Missing `file` field |
| 500 | Internal transcription error |

#### Example (cURL)

```bash
curl -X POST http://localhost:8000/transcribe \
     -F "file=@/path/to/audio.wav"
```

---

## Configuration

All settings are read from environment variables (prefix `STT_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL_NAME` | `v3_rnnt` | GigaAM model variant (`v3_rnnt` or `v3_ctc`) |
| `STT_DEVICE` | `cuda` | PyTorch device (`cuda` or `cpu`) |
| `STT_MAX_AUDIO_SIZE_BYTES` | `52428800` | Maximum accepted file size (50 MB) |

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite mocks the GigaAM model so no GPU or model weights are required.