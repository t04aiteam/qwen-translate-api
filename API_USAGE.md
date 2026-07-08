# Qwen3.5-4B NLP API — Usage (Postman)

Text **translation** and **summarization** powered by `Qwen/Qwen3.5-4B-GGUF`.
Translation supports ~201 languages.

Endpoints:
- [`POST /translate`](#endpoint-1-translate) — translate text between languages
- [`POST /summarize`](#endpoint-2-summarize) — summarize text in its original language

Base URL (default): `http://localhost:8000`. No authentication is required for
either endpoint. Both accept **raw JSON** or **multipart/form-data**.

---

## Endpoint 1: Translate

```
POST /translate
```

Base URL (default): `http://localhost:8000`

Full URL: `http://localhost:8000/translate`

### Postman setup (JSON — recommended)

1. **Method**: `POST`
2. **URL**: `http://localhost:8000/translate`
3. **Body** tab → select **raw** → choose **JSON** from the type dropdown.
4. Paste the request body:

   ```json
   {
     "text": "The weather is beautiful today.",
     "input_language": "English",
     "target_language": "Vietnamese",
     "max_new_tokens": 256,
     "temperature": 0.1
   }
   ```

5. Click **Send**.

### Request fields

| Field             | Type   | Required | Default | Notes                                                        |
|-------------------|--------|----------|---------|--------------------------------------------------------------|
| `text`            | string | Yes      | —       | The text to translate.                                       |
| `input_language`  | string | Yes      | —       | Source language name, e.g. `"English"`.                      |
| `target_language` | string | Yes      | —       | Target language name, e.g. `"Vietnamese"`.                   |
| `max_new_tokens`  | int    | No       | `512`   | 16–4096. Cap on generated tokens.                            |
| `temperature`     | float  | No       | `0.1`   | 0.0–2.0. Lower = more deterministic (recommended for translation). |

> Languages are passed as **full names** (`"English"`, `"French"`, `"Japanese"`),
> not ISO codes.

### Response

- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "translation": "Hôm nay thời tiết thật đẹp.",
  "input_language": "English",
  "target_language": "Vietnamese",
  "model": "Qwen3.5-4B-GGUF"
}
```

### Alternative: form-data

The same endpoint also accepts **multipart/form-data**. In Postman:
**Body → form-data**, then add text keys `text`, `input_language`,
`target_language` (and optionally `max_new_tokens`, `temperature`).

---

## Endpoint 2: Summarize

```
POST /summarize
```

Full URL: `http://localhost:8000/summarize`

### Postman setup (JSON — recommended)

1. **Method**: `POST`
2. **URL**: `http://localhost:8000/summarize`
3. **Body** tab → select **raw** → choose **JSON** from the type dropdown.
4. Paste the request body:

   ```json
   {
     "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals...",
     "language": "English",
     "length": "medium",
     "max_new_tokens": 512,
     "temperature": 0.1
   }
   ```

5. Click **Send**.

### Request fields

| Field            | Type   | Required | Default     | Notes                                                                        |
|------------------|--------|----------|-------------|------------------------------------------------------------------------------|
| `text`           | string | Yes      | —           | The text to summarize.                                                       |
| `language`       | string | No       | `"English"` | Language of the input; the summary is returned in the same language.         |
| `length`         | string | No       | `"medium"`  | `short` (1–2 sentences), `medium` (1 paragraph), `long` (several paragraphs). |
| `max_new_tokens` | int    | No       | `512`       | 16–4096. Cap on generated tokens.                                            |
| `temperature`    | float  | No       | `0.1`       | 0.0–2.0. Lower = more deterministic.                                         |

### Response

- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "summary": "Artificial intelligence refers to machine-demonstrated intelligence, distinct from the natural intelligence of humans and animals...",
  "language": "English",
  "length": "medium",
  "model": "Qwen3.5-4B-GGUF"
}
```

### Alternative: form-data

`/summarize` also accepts **multipart/form-data**. In Postman:
**Body → form-data**, then add text keys `text`, `language`, `length`
(and optionally `max_new_tokens`, `temperature`).

---

## Error responses (both endpoints)

| Status | Meaning                                                |
|--------|--------------------------------------------------------|
| `422`  | Missing/invalid field (e.g. no `text`, out-of-range param). |
| `500`  | Inference / unexpected server error.                   |

---

## Quick checks

- **Health probe**: `GET http://localhost:8000/health`
  ```json
  { "status": "ok", "model": "Qwen3.5-4B-GGUF" }
  ```
- **Interactive docs**: open `http://localhost:8000/docs` in a browser.
