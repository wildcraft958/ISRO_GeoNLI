# Qwen3-VL Async Inference API Documentation

This document describes how to integrate with the async job queue API for Qwen3-VL satellite imagery **Captioning** and **VQA** services.


**URLs:
```
CAPTION_BASE_URL=https://maximuspookus--qwen3-vlm-captioning-serve.modal.run
VQA_BASE_URL=https://maximuspookus--qwen3-vlm-vqa-serve.modal.run
```

---

## Overview

Both services expose identical API interfaces with two modes:

- **Synchronous**: `POST /v1/chat/completions` – Blocks until inference completes
- **Asynchronous**: `POST /v1/chat/completions/async` + `GET /v1/chat/completions/status/{call_id}` – Non-blocking job queue pattern

For production applications, use the async endpoints to maintain responsiveness and handle high concurrency.

---

## Model Information

| Property | Captioning Service | VQA Service |
|----------|-------------------|-------------|
| **Model Name** | `qwen-caption-special` | `qwen-vqa-special` |
| **Base Model** | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Instruct |
| **Fine-tuned for** | Satellite imagery captioning | Visual Question Answering |
| **GPU** | A100 (40GB) | A100 (40GB) |
| **Volume** | `vlm-weights-merged-caption-special` | `vlm-weights-merged-vqa-special` |
| **Max Tokens** | 512 (configurable) | 512 (configurable) |
| **Timeout** | 10 minutes | 10 minutes |
| **Scaledown Window** | 5 minutes | 5 minutes |

---

## Async Workflow

1. **Submit Job**: `POST /v1/chat/completions/async` → Receive `call_id`
2. **Poll Status**: `GET /v1/chat/completions/status/{call_id}` → Check job status
3. **Get Result**: When status is `200`, extract the generated response

---

## Endpoints

### 1. Submit Async Job

**Endpoint**: `POST /v1/chat/completions/async`

**Request Body** (Captioning example):
```json
{
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
            "url": "data:image/png;base64,iVBORw0KGgoAAAANS..."
          }
        }
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Request Body** (VQA example):
```json
{
  "model": "qwen-vqa-special",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "How many ships are visible in this harbor?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANS..."
          }
        }
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0.0
}
```

**Response** (200 OK):
```json
{
  "call_id": "fc-abc123xyz",
  "status": "pending",
  "model": "qwen-caption-special"
}
```

**Error Response** (500):
```json
{
  "error": "Error message",
  "traceback": "..."
}
```

---

### 2. Poll Job Status

**Endpoint**: `GET /v1/chat/completions/status/{call_id}`

**Path Parameters**:
- `call_id` (string): The job ID returned from `/async` endpoint

**Response Codes**:
- **200 OK**: Job completed, returns result
- **202 Accepted**: Job still processing
- **404 Not Found**: Job expired (older than 7 days) or invalid call_id
- **500 Internal Server Error**: Server error

**Response** (200 OK - Job Complete):
```json
{
  "id": "chatcmpl-fc-abc123xyz",
  "object": "chat.completion",
  "model": "qwen-caption-special",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This satellite image shows an airport with multiple runways..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 45,
    "total_tokens": 45
  }
}
```

**Response** (202 Accepted - Pending):
```json
{
  "status": "pending",
  "call_id": "fc-abc123xyz"
}
```

**Response** (404 Not Found - Expired):
```json
{
  "error": "Job result expired (older than 7 days)"
}
```

---

### 3. Synchronous Inference

**Endpoint**: `POST /v1/chat/completions`

Same request format as async, but blocks until inference completes. Use for:
- Testing and development
- Low-volume applications
- When you need immediate results

---

### 4. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "model": "qwen-caption-special"
}
```

---

## Python Integration Examples

### Basic Async Client

```python
import asyncio
import base64
import httpx
from pathlib import Path

# Choose your service
CAPTION_BASE_URL = "https://maximuspookus--qwen3-vlm-captioning-serve.modal.run"
VQA_BASE_URL = "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run"


async def submit_async_job(
    base_url: str,
    model: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Submit an async inference job."""
    # Load and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions/async",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["call_id"]


async def poll_job_status(
    base_url: str,
    call_id: str,
    max_wait: int = 300,
    poll_interval: float = 2.0
) -> str:
    """Poll for job result until complete or timeout."""
    async with httpx.AsyncClient() as client:
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job {call_id} did not complete within {max_wait}s")
            
            response = await client.get(
                f"{base_url}/v1/chat/completions/status/{call_id}",
                timeout=10.0
            )
            
            if response.status_code == 200:
                # Job complete
                result = response.json()
                return result["choices"][0]["message"]["content"]
            
            elif response.status_code == 202:
                # Still pending, wait and retry
                await asyncio.sleep(poll_interval)
                continue
            
            elif response.status_code == 404:
                # Job expired or invalid
                error_data = response.json()
                raise ValueError(f"Job expired or invalid: {error_data.get('error', 'Unknown error')}")
            
            else:
                # Other error
                response.raise_for_status()


# =============================================================================
# Captioning Example
# =============================================================================
async def caption_image(image_path: str) -> str:
    """Generate a caption for a satellite image."""
    call_id = await submit_async_job(
        base_url=CAPTION_BASE_URL,
        model="qwen-caption-special",
        image_path=image_path,
        prompt="Describe this satellite image in detail.",
        max_tokens=512,
        temperature=0.7
    )
    print(f"Captioning job submitted: {call_id}")
    
    result = await poll_job_status(CAPTION_BASE_URL, call_id)
    return result


# =============================================================================
# VQA Example
# =============================================================================
async def ask_question(image_path: str, question: str) -> str:
    """Ask a question about a satellite image."""
    call_id = await submit_async_job(
        base_url=VQA_BASE_URL,
        model="qwen-vqa-special",
        image_path=image_path,
        prompt=question,
        max_tokens=128,
        temperature=0.0  # Deterministic for factual answers
    )
    print(f"VQA job submitted: {call_id}")
    
    result = await poll_job_status(VQA_BASE_URL, call_id)
    return result


# =============================================================================
# Usage
# =============================================================================
async def main():
    image_path = "sample_satellite.png"
    
    # Captioning
    caption = await caption_image(image_path)
    print(f"Caption: {caption}")
    
    # VQA
    answer = await ask_question(image_path, "How many buildings are visible?")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Batch Processing Example

```python
import asyncio
import httpx
from typing import List, Dict, Tuple

async def process_batch_async(
    base_url: str,
    model: str,
    image_prompt_pairs: List[Tuple[str, str]],
    max_tokens: int = 256
) -> Dict[str, str]:
    """Process multiple images concurrently using async jobs."""
    
    # Submit all jobs
    call_ids = []
    for image_path, prompt in image_prompt_pairs:
        call_id = await submit_async_job(
            base_url=base_url,
            model=model,
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens
        )
        call_ids.append((image_path, call_id))
        print(f"Submitted: {image_path} -> {call_id}")
    
    # Poll all jobs (with exponential backoff)
    results = {}
    pending = {call_id: image_path for image_path, call_id in call_ids}
    poll_interval = 1.0
    
    async with httpx.AsyncClient() as client:
        while pending:
            tasks = [
                client.get(f"{base_url}/v1/chat/completions/status/{call_id}")
                for call_id in pending.keys()
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            completed = []
            for call_id, response in zip(list(pending.keys()), responses):
                if isinstance(response, Exception):
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    image_path = pending[call_id]
                    results[image_path] = result["choices"][0]["message"]["content"]
                    completed.append(call_id)
                elif response.status_code == 404:
                    image_path = pending[call_id]
                    results[image_path] = None  # Expired
                    completed.append(call_id)
            
            for call_id in completed:
                del pending[call_id]
            
            if pending:
                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 10.0)  # Exponential backoff
    
    return results


# Usage
async def batch_caption():
    images = [
        ("image1.png", "Describe this satellite image."),
        ("image2.png", "Describe this satellite image."),
        ("image3.png", "Describe this satellite image."),
    ]
    
    results = await process_batch_async(
        base_url=CAPTION_BASE_URL,
        model="qwen-caption-special",
        image_prompt_pairs=images
    )
    
    for path, caption in results.items():
        print(f"{path}: {caption[:100]}...")
```

---

### Error Handling Best Practices

```python
import httpx
from typing import Optional

async def safe_poll_job(
    base_url: str,
    call_id: str,
    max_retries: int = 3
) -> Optional[str]:
    """Poll job with retry logic and proper error handling."""
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(
                    f"{base_url}/v1/chat/completions/status/{call_id}",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                
                elif response.status_code == 202:
                    # Still processing, wait before retry
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                
                elif response.status_code == 404:
                    error_data = response.json()
                    raise ValueError(f"Job expired: {error_data.get('error')}")
                
                else:
                    response.raise_for_status()
                    
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    # Retry on server errors
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise
        
        raise TimeoutError(f"Job {call_id} did not complete after {max_retries} attempts")
```

---

## JavaScript/TypeScript Integration

```typescript
// Configuration
const CAPTION_BASE_URL = "https://maximuspookus--qwen3-vlm-captioning-serve.modal.run";
const VQA_BASE_URL = "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run";

interface AsyncJobResponse {
  call_id: string;
  status: "pending";
  model: string;
}

interface JobStatusResponse {
  id: string;
  object: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

async function submitAsyncJob(
  baseUrl: string,
  model: string,
  imageBase64: string,
  prompt: string,
  maxTokens: number = 512,
  temperature: number = 0.7
): Promise<string> {
  const response = await fetch(`${baseUrl}/v1/chat/completions/async`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: model,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: { url: `data:image/png;base64,${imageBase64}` },
            },
          ],
        },
      ],
      max_tokens: maxTokens,
      temperature: temperature,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to submit job: ${response.statusText}`);
  }

  const data: AsyncJobResponse = await response.json();
  return data.call_id;
}

async function pollJobStatus(
  baseUrl: string,
  callId: string,
  maxWait: number = 300,
  pollInterval: number = 2000
): Promise<string> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWait * 1000) {
    const response = await fetch(
      `${baseUrl}/v1/chat/completions/status/${callId}`
    );

    if (response.status === 200) {
      const data: JobStatusResponse = await response.json();
      return data.choices[0].message.content;
    } else if (response.status === 202) {
      // Still pending, wait and retry
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
      continue;
    } else if (response.status === 404) {
      throw new Error("Job expired or invalid");
    } else {
      throw new Error(`Unexpected status: ${response.status}`);
    }
  }

  throw new Error(`Job did not complete within ${maxWait}s`);
}

// =============================================================================
// Usage Examples
// =============================================================================

// Captioning
async function captionImage(imageBase64: string): Promise<string> {
  const callId = await submitAsyncJob(
    CAPTION_BASE_URL,
    "qwen-caption-special",
    imageBase64,
    "Describe this satellite image in detail."
  );
  console.log(`Captioning job submitted: ${callId}`);
  
  return await pollJobStatus(CAPTION_BASE_URL, callId);
}

// VQA
async function askQuestion(imageBase64: string, question: string): Promise<string> {
  const callId = await submitAsyncJob(
    VQA_BASE_URL,
    "qwen-vqa-special",
    imageBase64,
    question,
    128,  // Lower max_tokens for answers
    0.0   // Deterministic
  );
  console.log(`VQA job submitted: ${callId}`);
  
  return await pollJobStatus(VQA_BASE_URL, callId);
}
```

---

## cURL Examples

### Captioning - Submit Job
```bash
curl -X POST "https://maximuspookus--qwen3-vlm-captioning-serve.modal.run/v1/chat/completions/async" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-caption-special",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this satellite image."},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."}
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### VQA - Submit Job
```bash
curl -X POST "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run/v1/chat/completions/async" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vqa-special",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "How many ships are in this harbor?"},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."}
          }
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

### Poll Status
```bash
# Captioning
curl "https://maximuspookus--qwen3-vlm-captioning-serve.modal.run/v1/chat/completions/status/fc-abc123xyz"

# VQA
curl "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run/v1/chat/completions/status/fc-abc123xyz"
```

### Health Check
```bash
# Captioning
curl "https://maximuspookus--qwen3-vlm-captioning-serve.modal.run/health"

# VQA
curl "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run/health"
```

---

## Rate Limits & Best Practices

1. **Polling Interval**: Use 2-5 seconds between polls to avoid overwhelming the server
2. **Timeout**: Set reasonable timeouts (5-10 minutes for satellite images)
3. **Exponential Backoff**: Increase poll interval if job is taking longer
4. **Error Handling**: Always handle 404 (expired), 202 (pending), and 500 (server error)
5. **Concurrency**: Modal auto-scales, but batch submissions should be throttled client-side
6. **Job Retention**: Results are available for **7 days** after completion
7. **Cold Starts**: First request may take 10-30s if no containers are warm

---

## GPU Configuration

Both services are configured to stay within **2 A100 GPUs total**:

| Setting | Value | Description |
|---------|-------|-------------|
| GPU | A100 (40GB) | Sufficient for 8B models with FP16 |
| Memory Utilization | 88% | ~35GB used, leaves headroom |
| Tensor Parallel | 1 | Single GPU per request |
| Scaledown Window | 5 minutes | Container stays warm after request |
| Min Containers | 0 | No always-on containers (on-demand) |

---

## Troubleshooting

### Job Stuck in Pending
- Check Modal dashboard for container status
- Verify GPU availability (A100 40GB)
- Cold start can take 10-30s for first request
- Increase poll timeout

### 404 Expired Error
- Jobs expire after 7 days
- Re-submit the job if needed

### 500 Server Error
- Check Modal logs for stack trace
- Verify model is deployed correctly
- Ensure volume is mounted:
  - Captioning: `vlm-weights-merged-caption-special`
  - VQA: `vlm-weights-merged-vqa-special`

### Cold Start Delays
- First request after idle may take 10-30s
- Subsequent requests within 5 minutes are fast
- Consider a warm-up request before production traffic

---

## Deployment

Deploy using Modal CLI:

```bash
# Deploy Captioning service
modal deploy captioning.py

# Deploy VQA service
modal deploy vqa.py
```

Check deployment status:
```bash
modal app list
```

View logs:
```bash
modal app logs qwen3-vlm-captioning
modal app logs qwen3-vlm-vqa
```

---

## Backend Integration Guide

### Updating ModalServiceClient for Async Operations

Add async methods to `app/services/modal_client.py`:

```python
import httpx
import asyncio
from typing import Optional, Dict, List, Tuple

class ModalServiceClient:
    """Client for calling Modal-deployed VLM services."""
    
    def __init__(self):
        self.vqa_base_url = os.getenv("VQA_BASE_URL", "https://<org>--qwen3-vlm-vqa.modal.run")
        self.caption_base_url = os.getenv("CAPTION_BASE_URL", "https://<org>--qwen3-vlm-captioning.modal.run")
        self.timeout = int(os.getenv("VQA_TIMEOUT", "120"))
    
    # ... existing sync methods ...
    
    # =========================================================================
    # Async Job Methods
    # =========================================================================
    
    async def submit_vqa_async(
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
    
    async def submit_captioning_async(
        self,
        image_url: str,
        prompt: str = "Describe this satellite image in detail.",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Submit async captioning job and return call_id."""
        async with httpx.AsyncClient(timeout=30) as client:
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
                f"{self.caption_base_url}/v1/chat/completions/async",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["call_id"]
    
    async def poll_job_status(
        self,
        base_url: str,
        call_id: str,
        max_wait: int = 300,
        poll_interval: float = 2.0
    ) -> Dict[str, any]:
        """Poll for async job result until complete or timeout."""
        async with httpx.AsyncClient(timeout=10) as client:
            start_time = asyncio.get_event_loop().time()
            
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_wait:
                    raise TimeoutError(f"Job {call_id} timed out after {max_wait}s")
                
                response = await client.get(
                    f"{base_url}/v1/chat/completions/status/{call_id}"
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 202:
                    await asyncio.sleep(poll_interval)
                    continue
                
                elif response.status_code == 404:
                    error_data = response.json()
                    raise ValueError(
                        f"Job expired or invalid: {error_data.get('error', 'Unknown')}"
                    )
                
                else:
                    response.raise_for_status()
    
    async def poll_vqa_result(self, call_id: str, timeout: int = 300) -> str:
        """Poll VQA job and return answer."""
        result = await self.poll_job_status(
            self.vqa_base_url,
            call_id,
            max_wait=timeout
        )
        return result["choices"][0]["message"]["content"]
    
    async def poll_captioning_result(self, call_id: str, timeout: int = 300) -> str:
        """Poll captioning job and return caption."""
        result = await self.poll_job_status(
            self.caption_base_url,
            call_id,
            max_wait=timeout
        )
        return result["choices"][0]["message"]["content"]
```

### Integration with Orchestrator for Batch Processing

Update `app/orchestrator.py` to support async batch operations:

```python
from app.services.modal_client import ModalServiceClient
from typing import List, Dict

modal_client = ModalServiceClient()

async def process_batch_vqa_async(
    image_question_pairs: List[Tuple[str, str]],
    max_concurrent: int = 10
) -> Dict[str, str]:
    """
    Process multiple VQA requests concurrently using async endpoints.
    
    Args:
        image_question_pairs: List of (image_url, question) tuples
        max_concurrent: Maximum concurrent jobs to submit
    
    Returns:
        Dict mapping image_url to answer
    """
    import asyncio
    
    # Submit all jobs (with concurrency limit)
    semaphore = asyncio.Semaphore(max_concurrent)
    call_ids = {}
    
    async def submit_with_limit(image_url: str, question: str):
        async with semaphore:
            call_id = await modal_client.submit_vqa_async(image_url, question)
            call_ids[call_id] = image_url
    
    # Submit all jobs concurrently
    tasks = [
        submit_with_limit(image_url, question)
        for image_url, question in image_question_pairs
    ]
    await asyncio.gather(*tasks)
    
    # Poll all jobs concurrently
    results = {}
    pending = set(call_ids.keys())
    
    while pending:
        # Poll all pending jobs
        poll_tasks = [
            modal_client.poll_vqa_result(call_id, timeout=300)
            for call_id in pending
        ]
        
        completed = await asyncio.gather(*poll_tasks, return_exceptions=True)
        
        # Process results
        for call_id, result in zip(list(pending), completed):
            if isinstance(result, Exception):
                # Log error but continue polling others
                logger.error(f"VQA job {call_id} failed: {result}")
                image_url = call_ids[call_id]
                results[image_url] = None
            else:
                image_url = call_ids[call_id]
                results[image_url] = result
        
        # Remove completed jobs
        pending = {
            call_id for call_id, result in zip(list(call_ids.keys()), completed)
            if isinstance(result, Exception)  # Retry failed ones
        }
        
        if pending:
            await asyncio.sleep(2)  # Wait before next poll cycle
    
    return results
```

### Background Task Integration

Use FastAPI BackgroundTasks for fire-and-forget async jobs:

```python
from fastapi import BackgroundTasks, APIRouter
from app.services.modal_client import ModalServiceClient

router = APIRouter()
modal_client = ModalServiceClient()

@router.post("/api/vqa/async")
async def submit_vqa_async(
    image_url: str,
    question: str,
    background_tasks: BackgroundTasks
):
    """Submit VQA job and return call_id immediately."""
    call_id = await modal_client.submit_vqa_async(image_url, question)
    
    # Optionally: Store call_id in database for later retrieval
    # await store_job(db, call_id, "vqa", image_url, question)
    
    return {
        "call_id": call_id,
        "status": "pending",
        "poll_url": f"/api/vqa/status/{call_id}"
    }

@router.get("/api/vqa/status/{call_id}")
async def get_vqa_status(call_id: str):
    """Poll VQA job status."""
    try:
        answer = await modal_client.poll_vqa_result(call_id, timeout=0)  # Non-blocking
        return {
            "status": "completed",
            "answer": answer
        }
    except TimeoutError:
        return {
            "status": "pending",
            "call_id": call_id
        }
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e)
        }
```

### Error Handling in Production

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ModalServiceClient:
    # ... existing methods ...
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def submit_vqa_async_safe(
        self,
        image_url: str,
        query: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """Submit async VQA job with retry logic."""
        try:
            return await self.submit_vqa_async(
                image_url, query, max_tokens, temperature
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                # Server error - will retry
                raise
            else:
                # Client error - don't retry
                raise ValueError(f"Invalid request: {e.response.text}")
        except httpx.TimeoutException:
            raise TimeoutError("Request timeout")
    
    async def poll_job_status_safe(
        self,
        base_url: str,
        call_id: str,
        max_wait: int = 300,
        poll_interval: float = 2.0,
        max_retries: int = 3
    ) -> Dict[str, any]:
        """Poll job with retry logic for network errors."""
        for attempt in range(max_retries):
            try:
                return await self.poll_job_status(
                    base_url, call_id, max_wait, poll_interval
                )
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception(f"Network error: {e}")
```

---

## Support

For issues or questions:

1. **Modal Dashboard**: Check deployment logs and container status
2. **Volumes**: Verify model weights are present in the volumes
3. **GPU Availability**: Check Modal's GPU availability in your region
4. **Backend Integration**: See `Basic_vqa_caption.md` for sync endpoint integration

**Volumes to verify:**
- `vlm-weights-merged-caption-special` (Captioning)
- `vlm-weights-merged-vqa-special` (VQA)

**Environment Variables:**
```bash
VQA_BASE_URL=https://<org>--qwen3-vlm-vqa.modal.run
CAPTION_BASE_URL=https://<org>--qwen3-vlm-captioning.modal.run
VQA_TIMEOUT=120
CAPTION_TIMEOUT=120
```
