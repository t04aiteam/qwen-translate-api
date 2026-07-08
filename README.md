# Qwen3.5-4B NLP API

A FastAPI service for **text translation and summarization**, powered by
[unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
running locally via **llama-cpp-python**. The quantised GGUF checkpoint
(`Qwen3.5-4B-Q4_K_M.gguf`, ~2.5 GB) is downloaded from the Hugging Face Hub on
first startup and cached under `~/.cache/huggingface`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | |
| ~3 GB disk | GGUF checkpoint downloads on first startup |
| GPU (optional) | Requires a CUDA build of `llama-cpp-python` (see below). CPU-only works too |

---

## Setup

### 1. Clone & install

```bash
git clone <this-repo>
cd qwen35-translation-api

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The default `llama-cpp-python` wheel is **CPU-only**. For GPU inference,
install a CUDA build instead:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. Configure (optional)

All settings have working defaults. To override them:

```bash
cp .env.example .env
```

Note: the app does not load `.env` by itself — pass `--env-file .env` to
uvicorn (shown below) or export the variables in your shell.

### 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 7276 --env-file .env
# or (uses defaults from the environment, port 7276):
python main.py
```

Interactive docs: **http://localhost:7276/docs**

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GGUF_REPO` | No | `unsloth/Qwen3.5-4B-GGUF` | Hugging Face repo id of the GGUF checkpoint |
| `GGUF_FILE` | No | `Qwen3.5-4B-Q4_K_M.gguf` | Filename inside the repo (pick another quant to trade speed/quality) |
| `HF_TOKEN` | No | — | HF access token (only needed for private repos) |
| `N_GPU_LAYERS` | No | `-1` | Layers offloaded to GPU. `-1` = all, `0` = CPU-only |
| `N_CTX` | No | `8192` | Context window size (tokens) |
| `CUDA_VISIBLE_DEVICES` | No | — | Pin the service to a specific GPU (e.g. `0`) |

---

## API Reference

No authentication — intended for trusted local-network deployment.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service liveness check, returns the loaded GGUF filename |
| `POST` | `/translate` | Translate text between languages |
| `POST` | `/summarize` | Summarize text in its original language |

Both POST endpoints accept **`application/json`** or **`multipart/form-data`**
with the same field names.

### `POST /translate` — fields

| Field | Required | Default | Description |
|---|---|---|---|
| `text` | **Yes** | — | Text to translate |
| `input_language` | **Yes** | — | Language of the input, as a plain name (`"English"`, `"Vietnamese"`, …) |
| `target_language` | **Yes** | — | Language to translate into |
| `max_new_tokens` | No | `512` | Generation cap (16–4096) |
| `temperature` | No | `0.1` | Sampling temperature (0–2). Keep low for translation |

### `POST /summarize` — fields

| Field | Required | Default | Description |
|---|---|---|---|
| `text` | **Yes** | — | Text to summarize |
| `language` | No | `English` | Language of the input; the summary is returned in the same language |
| `length` | No | `medium` | `short` (1–2 sentences), `medium` (1 paragraph), `long` (several paragraphs) |
| `max_new_tokens` | No | `512` | Generation cap (16–4096) |
| `temperature` | No | `0.1` | Sampling temperature (0–2) |

---

## Example Requests

### Health check

```bash
curl http://localhost:7276/health
# {"status":"ok","model":"Qwen3.5-4B-Q4_K_M.gguf"}
```

### Translate (JSON)

```bash
curl -X POST http://localhost:7276/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The weather is beautiful today.",
    "input_language": "English",
    "target_language": "Vietnamese"
  }'
```

```json
{
  "translation": "Hôm nay thời tiết thật đẹp.",
  "input_language": "English",
  "target_language": "Vietnamese",
  "model": "Qwen3.5-4B-Q4_K_M.gguf"
}
```

### Translate (form-data)

```bash
curl -X POST http://localhost:7276/translate \
  -F text="Hello, how are you?" \
  -F input_language="English" \
  -F target_language="Japanese"
```

### Summarize

```bash
curl -X POST http://localhost:7276/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
    "language": "English",
    "length": "short"
  }'
```

```json
{
  "summary": "AI is machine-demonstrated intelligence ...",
  "language": "English",
  "length": "short",
  "model": "Qwen3.5-4B-Q4_K_M.gguf"
}
```

---

## Notes

- **Reasoning tags**: Qwen3.5 may emit `<think>…</think>` reasoning blocks; the
  API strips them automatically before returning the result.
- **First startup**: downloads the ~2.5 GB GGUF file from the Hub; subsequent
  starts load from the local cache.
- **Quantisation**: set `GGUF_FILE` to another file from the repo (e.g.
  `Q8_0` for higher quality, `Q2_K` for smaller/faster) — no code changes needed.
- **Concurrency**: a single model instance serves all requests through a
  thread-pool executor. For higher throughput run multiple replicas (each loads
  its own model copy).
