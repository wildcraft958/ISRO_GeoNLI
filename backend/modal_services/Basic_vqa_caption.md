Here’s a sample `API_USE.md` you can drop into `backend/modal_services/API_USE.md`:

```markdown
# Qwen3 VLM Services – API Usage Guide

This document describes how to integrate with the **VQA** and **Captioning** Modal services defined in `vqa.py` and `captioning.py`.

Both services expose **OpenAI-compatible** HTTP endpoints:

- `POST /v1/chat/completions` – synchronous inference
- `POST /v1/chat/completions/async` – submit async job
- `GET  /v1/chat/completions/status/{call_id}` – poll async job
- `GET  /health` – basic health check

The request/response format matches the OpenAI Chat Completions API.

---

## 1. Base URLs

Each service is deployed as a separate Modal app with its own base URL.

- **Captioning service (`captioning.py`)**  
  Example:  
  `CAPTION_BASE_URL=https://<org>--qwen3-vlm-captioning.modal.run`

- **VQA service (`vqa.py`)**  
  Example:  
  `VQA_BASE_URL=https://<org>--qwen3-vlm-vqa.modal.run`

You’ll get the exact URLs from the Modal dashboard after deployment.

---

## 2. Message Format

Both services accept a `messages` array in **OpenAI chat format**:

### 2.1 Text-only request

```json
{
  "model": "qwen-vqa-special",
  "messages": [
    {
      "role": "user",
      "content": "What objects do you see in this image?"
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

### 2.2 Multimodal request (image + text)

The `content` becomes a list of typed items:

```json
{
  "model": "qwen-vqa-special",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the main land cover type here?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/satellite.png"
          }
        }
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

Supported `image_url.url` formats:

- Remote URL: `https://.../image.png`
- Data URI: `data:image/png;base64,<base64-bytes>`

The backend will:
- Download / decode the image
- Pass it to the model via `multi_modal_data={"image": image}`

---

## 3. Synchronous Inference

### 3.1 Captioning (sync)

**Endpoint:**

```text
POST {CAPTION_BASE_URL}/v1/chat/completions
```

**Example (curl):**

```bash
curl -X POST "$CAPTION_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-caption-special",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this satellite image in detail."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/satellite.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Response (shape):**

```json
{
  "id": "chatcmpl-modal",
  "object": "chat.completion",
  "model": "qwen-caption-special",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A detailed caption about the image..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 45,
    "total_tokens": 168
  }
}
```

### 3.2 VQA (sync)

**Endpoint:**

```text
POST {VQA_BASE_URL}/v1/chat/completions
```

**Example (curl):**

```bash
curl -X POST "$VQA_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vqa-special",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "How many large ships are visible in this bay?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/harbor.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

**Response shape** is the same as captioning; the content will be an answer instead of a caption.

---

## 4. Asynchronous Job Queue

For long-running inference or higher concurrency, use the async pattern provided in both services.

### 4.1 Submit Async Job

**Endpoint:**

```text
POST {SERVICE_BASE_URL}/v1/chat/completions/async
```

- `SERVICE_BASE_URL` = `CAPTION_BASE_URL` or `VQA_BASE_URL`

**Request body** is the same as synchronous mode.

**Example (Python):**

```python
import requests
import time

CAPTION_BASE_URL = "https://<org>--qwen3-vlm-captioning.modal.run"

payload = {
    "model": "qwen-caption-special",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        }
    ],
    "max_tokens": 256,
    "temperature": 0.7,
}

resp = requests.post(f"{CAPTION_BASE_URL}/v1/chat/completions/async", json=payload)
resp.raise_for_status()
call_id = resp.json()["call_id"]
print("Submitted job:", call_id)
```

**Response:**

```json
{
  "call_id": "fc-abc123xyz",
  "status": "pending",
  "model": "qwen-caption-special"
}
```

### 4.2 Poll Job Status

**Endpoint:**

```text
GET {SERVICE_BASE_URL}/v1/chat/completions/status/{call_id}
```

**Example (Python):**

```python
def poll_result(base_url: str, call_id: str, timeout_s: int = 120, interval_s: int = 2):
    import requests, time

    start = time.time()
    while True:
        r = requests.get(f"{base_url}/v1/chat/completions/status/{call_id}")
        if r.status_code == 200:
            return r.json()  # Completed
        elif r.status_code == 202:
            # Still pending
            if time.time() - start > timeout_s:
                raise TimeoutError("Job timed out while waiting for completion")
            time.sleep(interval_s)
        elif r.status_code == 404:
            raise RuntimeError("Job expired or not found")
        else:
            r.raise_for_status()

# Usage:
result = poll_result(CAPTION_BASE_URL, call_id)
print(result["choices"][0]["message"]["content"])
```

**Status codes:**
- `200` – job completed, returns full OpenAI-style response
- `202` – job still pending
- `404` – job expired or invalid `call_id`
- `500` – server error (see `error` + `traceback` in body)

---

## 5. Health Checks

Each service exposes a simple health endpoint:

```text
GET {SERVICE_BASE_URL}/health
```

**Example:**

```bash
curl "$VQA_BASE_URL/health"
# => {"status": "ok", "model": "qwen-vqa-special"}
```

Use this in readiness/liveness probes or basic monitoring.

---

## 6. Integration Patterns

### 6.1 In Your Backend (FastAPI example)

```python
import httpx
from fastapi import APIRouter

router = APIRouter()
VQA_BASE_URL = "https://<org>--qwen3-vlm-vqa.modal.run"

@router.post("/api/vqa")
async def vqa_endpoint(image_url: str, question: str):
    async with httpx.AsyncClient(timeout=30) as client:
        payload = {
            "model": "qwen-vqa-special",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        resp = await client.post(f"{VQA_BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}
```

Same pattern applies to captioning; only `BASE_URL` and `model` differ.

---

## 7. Error Handling

- The services wrap errors in JSON with `error` and `traceback` fields (500 responses).
- For async:
  - Handle `404` (expired/invalid job) and `202` (still pending).
- For sync:
  - Treat non-2xx as errors; log `response.text` for debugging.

---

## 8. Summary

- **VQA** and **Captioning** share the same API interface (OpenAI Chat format).
- Use **sync** endpoints for low-latency, short requests.
- Use **async** endpoints + **status** polling for heavier workloads.
- Always send **both** text and image in the `messages` array when doing multimodal inference.

This should be enough for other developers to integrate these services cleanly into any app (Python, JS, etc.).