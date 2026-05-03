# stt-translator-service

A lightweight Python backend that receives audio files and returns their transcription using
**[GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3)** via Hugging Face Transformers.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Web framework | **FastAPI** |
| ASGI server | **Uvicorn** |
| STT model | **GigaAM v3 (Hugging Face)** |
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

> Note: on Windows, the Hugging Face model stack may require CUDA-enabled wheels.
> For real transcription, prefer Docker/WSL2 (Linux). The test suite still runs
> on any platform because it mocks the model.

---

## API reference

### Base URL

`http://<host>:8000`

### `GET /health`

Liveness probe.

**Response 200**

```json
{"status": "ok"}
```

### `POST /transcribe`

Transcribe an uploaded audio file.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | binary | Audio file (WAV, MP3, OGG, FLAC, AAC) |

**Content types**

`audio/wav`, `audio/x-wav`, `audio/wave`, `audio/mpeg`, `audio/mp3`, `audio/ogg`, `audio/flac`, `audio/x-flac`, `audio/aac`

**Response 200**

```json
{"text": "Привет, как дела?"}
```

**Error responses**

| Status | When it happens | Example |
|--------|------------------|---------|
| 400 | Empty file, unsupported content type, missing HF token for longform | `{"detail":"Unsupported content type 'video/mp4'. Allowed types: audio/wav, ..."}` |
| 413 | File exceeds size limit | `{"detail":"File size 99999999 bytes exceeds the maximum allowed size of 52428800 bytes."}` |
| 422 | Missing `file` field | FastAPI validation error |
| 500 | Internal transcription error | `{"detail":"Transcription error: ..."}` |

**Examples**

PowerShell (Windows):

```powershell
curl.exe -X POST http://localhost:8000/transcribe -F "file=@.resources\test_wav.wav;type=audio/wav"
```

cURL (Linux/macOS):

```bash
curl -X POST http://localhost:8000/transcribe \
     -F "file=@/path/to/audio.wav"
```

Python (requests):

```python
import requests

with open("/path/to/audio.mp3", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": ("audio.mp3", f, "audio/mpeg")},
        timeout=600,
    )
print(resp.json())
```

**Notes**

- Long files may require longform transcription; if HF token is required, set `HF_TOKEN` or `STT_HF_TOKEN`.
- The service converts non-WAV formats to WAV internally using `ffmpeg`.

---

## Configuration

All settings are read from environment variables (prefix `STT_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL_ID` | `ai-sage/GigaAM-v3` | Hugging Face model id |
| `STT_MODEL_REVISION` | `e2e_rnnt` | Model revision (e.g. `ssl`, `ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt`) |
| `STT_FP16_ENCODER` | `false` | Enable FP16 inference on CUDA |
| `STT_DEVICE` | `cuda` | PyTorch device (`cuda` or `cpu`) |
| `STT_MAX_AUDIO_SIZE_BYTES` | `52428800` | Maximum accepted file size (50 MB) |
| `STT_HF_TOKEN` | *(empty)* | Hugging Face token for longform segmentation |

### Offline mode (no internet)

Use `.env.example` as a template. The recommended offline flags are:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/root/.cache/huggingface
```

Run the container once with internet to populate the cache, then keep the
`hf_cache` volume mounted for offline runs. If offline mode is enabled and the
cache does not contain the model files, startup will fail with a clear error.

**Cache priming options:**

1) Temporarily disable offline flags, start the service once, then stop it:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 docker compose up --build
```

2) Alternatively, set `STT_MODEL_ID` to a local path that already contains the
downloaded Hugging Face snapshot.

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite mocks the GigaAM model so no GPU or model weights are required.

---

## Local model runner

A tiny CLI is included for direct transcription without the API:

```bash
python scripts/transcribe_local.py --file /path/to/audio.wav
```
