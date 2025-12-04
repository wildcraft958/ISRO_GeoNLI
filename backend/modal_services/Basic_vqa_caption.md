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

## 6. Backend Integration Patterns

### 6.1 Environment Configuration

Add these environment variables to your backend configuration:

```bash
# .env or environment variables
VQA_BASE_URL=https://<org>--qwen3-vlm-vqa.modal.run
CAPTION_BASE_URL=https://<org>--qwen3-vlm-captioning.modal.run

# Optional: Configure timeouts
VQA_TIMEOUT=120
CAPTION_TIMEOUT=120
```

### 6.2 Updating ModalServiceClient

Update `app/services/modal_client.py` to use the new OpenAI-compatible endpoints:

```python
import httpx
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class ModalServiceClient:
    """Client for calling Modal-deployed VLM services with OpenAI-compatible API."""
    
    def __init__(self):
        self.vqa_base_url = os.getenv(
            "VQA_BASE_URL", 
            "https://<org>--qwen3-vlm-vqa.modal.run"
        )
        self.caption_base_url = os.getenv(
            "CAPTION_BASE_URL",
            "https://<org>--qwen3-vlm-captioning.modal.run"
        )
        self.timeout = int(os.getenv("VQA_TIMEOUT", "120"))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def call_vqa(
        self, 
        image_url: str, 
        query: str,
        vqa_type: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> dict:
        """
        Call VQA service using OpenAI-compatible format.
        
        Returns:
            dict: {
                "answer": str,
                "confidence": float (optional),
                "vqa_type": str (if provided)
            }
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": "qwen-vqa-special",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await client.post(
                f"{self.vqa_base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data["choices"][0]["message"]["content"]
            result = {"answer": answer}
            
            if vqa_type:
                result["vqa_type"] = vqa_type
            
            return result
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def call_captioning(
        self, 
        image_url: str,
        prompt: str = "Describe this satellite image in detail.",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> dict:
        """
        Call captioning service using OpenAI-compatible format.
        
        Returns:
            dict: {"caption": str}
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": "qwen-caption-special",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await client.post(
                f"{self.caption_base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            caption = data["choices"][0]["message"]["content"]
            return {"caption": caption}
    
    # Async job methods (for batch processing)
    async def call_vqa_async(
        self,
        image_url: str,
        query: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """Submit async VQA job and return call_id."""
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": "qwen-vqa-special",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await client.post(
                f"{self.vqa_base_url}/v1/chat/completions/async",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["call_id"]
    
    async def poll_vqa_result(self, call_id: str, timeout: int = 300) -> dict:
        """Poll for async VQA result."""
        import asyncio
        
        async with httpx.AsyncClient(timeout=10) as client:
            start_time = asyncio.get_event_loop().time()
            
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"VQA job {call_id} timed out")
                
                response = await client.get(
                    f"{self.vqa_base_url}/v1/chat/completions/status/{call_id}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                    return {"answer": answer}
                elif response.status_code == 202:
                    await asyncio.sleep(2)  # Poll every 2 seconds
                    continue
                elif response.status_code == 404:
                    raise ValueError(f"VQA job {call_id} expired or not found")
                else:
                    response.raise_for_status()
```

### 6.3 Integration with Orchestrator Workflow

Update `app/orchestrator.py` nodes to use the new client methods:

```python
from app.services.modal_client import ModalServiceClient

modal_client = ModalServiceClient()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def call_vqa_node(state: AgentState) -> dict:
    """VQA service node with retry logic."""
    try:
        image_url = state["image_url"]
        query = state.get("user_query") or ""
        vqa_type = state.get("vqa_type")
        
        state["execution_log"].append(f"VQA ({vqa_type or 'general'}): Calling service...")
        
        # Use async client method
        result = await modal_client.call_vqa(
            image_url=image_url,
            query=query,
            vqa_type=vqa_type,
            max_tokens=128,
            temperature=0.0
        )
        
        answer = result.get("answer", "")
        state["execution_log"].append(f"VQA: Success ({answer[:50]}...)")
        
        # Append assistant message
        message_update = append_assistant_message(
            state,
            answer,
            "answer",
            metadata={"vqa_result": result, "vqa_type": vqa_type}
        )
        
        return {
            "vqa_result": answer,
            **message_update
        }
    
    except Exception as e:
        logger.error(f"VQA service failed: {e}")
        state["execution_log"].append(f"VQA: Failed - {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def call_captioning_node(state: AgentState) -> dict:
    """Captioning service node with retry logic."""
    try:
        image_url = state["image_url"]
        
        state["execution_log"].append("Captioning: Calling service...")
        
        # Use async client method
        result = await modal_client.call_captioning(
            image_url=image_url,
            prompt="Describe this satellite image in detail.",
            max_tokens=512,
            temperature=0.7
        )
        
        caption = result.get("caption", "")
        state["execution_log"].append(f"Captioning: Success ({caption[:50]}...)")
        
        # Append assistant message
        message_update = append_assistant_message(
            state,
            caption,
            "caption",
            metadata={"caption_result": result}
        )
        
        return {
            "caption_result": caption,
            **message_update
        }
    
    except Exception as e:
        logger.error(f"Captioning service failed: {e}")
        state["execution_log"].append(f"Captioning: Failed - {str(e)}")
        raise
```

### 6.4 Simple FastAPI Endpoint Example

```python
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()
VQA_BASE_URL = os.getenv("VQA_BASE_URL", "https://<org>--qwen3-vlm-vqa.modal.run")

@router.post("/api/vqa")
async def vqa_endpoint(image_url: str, question: str):
    """Simple VQA endpoint wrapper."""
    async with httpx.AsyncClient(timeout=120) as client:
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
        try:
            resp = await client.post(f"{VQA_BASE_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer}
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"VQA service error: {e.response.text}"
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="VQA service timeout")
```

### 6.5 Batch Processing with Async Endpoints

For processing multiple images concurrently:

```python
async def process_batch_vqa(
    client: ModalServiceClient,
    image_question_pairs: list[tuple[str, str]]
) -> dict[str, str]:
    """Process multiple VQA requests concurrently using async endpoints."""
    import asyncio
    
    # Submit all jobs
    call_ids = {}
    for image_url, question in image_question_pairs:
        call_id = await client.call_vqa_async(image_url, question)
        call_ids[call_id] = (image_url, question)
    
    # Poll all jobs concurrently
    results = {}
    pending = set(call_ids.keys())
    
    while pending:
        tasks = [
            client.poll_vqa_result(call_id, timeout=300)
            for call_id in pending
        ]
        
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        for call_id, result in zip(list(pending), completed):
            if isinstance(result, Exception):
                # Retry failed polls
                continue
            
            image_url, question = call_ids[call_id]
            results[image_url] = result.get("answer", "")
            pending.remove(call_id)
        
        if pending:
            await asyncio.sleep(2)  # Wait before next poll cycle
    
    return results
```

---

## 7. Error Handling

### 7.1 Error Response Format

All errors return JSON with `error` and optional `traceback` fields:

```json
{
  "error": "Error message description",
  "traceback": "Full stack trace (in development mode)"
}
```

### 7.2 HTTP Status Codes

- **200 OK**: Request successful
- **202 Accepted**: Async job submitted (pending)
- **400 Bad Request**: Invalid request format
- **404 Not Found**: 
  - Async job expired (older than 7 days)
  - Invalid `call_id`
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service temporarily unavailable

### 7.3 Error Handling Best Practices

```python
import httpx
from typing import Optional

async def safe_call_vqa(
    base_url: str,
    image_url: str,
    question: str,
    max_retries: int = 3
) -> Optional[str]:
    """Call VQA with retry logic and proper error handling."""
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "model": "qwen-vqa-special",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0
        }
        
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        error_data = response.json()
                        raise Exception(f"VQA service error: {error_data.get('error')}")
                
                else:
                    # Client error - don't retry
                    response.raise_for_status()
                    
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception("VQA service timeout after retries")
            
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception(f"VQA service request failed: {e}")
        
        return None
```

### 7.4 Async Error Handling

```python
async def safe_poll_result(
    base_url: str,
    call_id: str,
    max_wait: int = 300,
    poll_interval: float = 2.0
) -> str:
    """Poll async job with proper error handling."""
    import asyncio
    
    async with httpx.AsyncClient(timeout=10) as client:
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job {call_id} timed out after {max_wait}s")
            
            try:
                response = await client.get(
                    f"{base_url}/v1/chat/completions/status/{call_id}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                
                elif response.status_code == 202:
                    # Still pending
                    await asyncio.sleep(poll_interval)
                    continue
                
                elif response.status_code == 404:
                    error_data = response.json()
                    raise ValueError(
                        f"Job expired or not found: {error_data.get('error', 'Unknown')}"
                    )
                
                else:
                    response.raise_for_status()
                    
            except httpx.TimeoutException:
                # Retry on timeout
                await asyncio.sleep(poll_interval)
                continue
```

---

## 8. Migration Guide

### 8.1 Updating from Old API Format

If you're migrating from the old `/vqa` and `/captioning` endpoints:

**Old format:**
```python
response = requests.post(
    f"{base_url}/vqa",
    json={"image_url": image_url, "query": question}
)
answer = response.json()["answer"]
```

**New format:**
```python
response = requests.post(
    f"{base_url}/v1/chat/completions",
    json={
        "model": "qwen-vqa-special",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 128,
        "temperature": 0.0
    }
)
answer = response.json()["choices"][0]["message"]["content"]
```

### 8.2 Key Changes

1. **Endpoint paths**: `/vqa` → `/v1/chat/completions`
2. **Request format**: Simple dict → OpenAI Chat Completions format
3. **Response format**: `{"answer": "..."}` → `{"choices": [{"message": {"content": "..."}}]}`
4. **Model parameter**: Now required in request (`"model": "qwen-vqa-special"`)

---

## 9. Summary

- **VQA** and **Captioning** share the same API interface (OpenAI Chat format)
- Use **sync** endpoints (`/v1/chat/completions`) for low-latency, short requests
- Use **async** endpoints (`/v1/chat/completions/async` + `/status/{call_id}`) for batch processing and higher concurrency
- Always send **both** text and image in the `messages` array when doing multimodal inference
- Update `ModalServiceClient` to use the new OpenAI-compatible endpoints
- Configure environment variables (`VQA_BASE_URL`, `CAPTION_BASE_URL`) for your deployment
- Implement proper error handling with retry logic for production use

This guide provides everything needed to integrate these services into your backend pipeline.