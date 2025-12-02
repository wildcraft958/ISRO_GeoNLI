# Qwen3-VL Async Inference API Documentation

This document describes how to integrate with the async job queue API for Qwen3-VL satellite imagery captioning.

## Base URL

```
https://maximuspookus--qwen3-vlm-caption-serve.modal.run
```

## Overview

The API supports two modes:
- **Synchronous**: `/v1/chat/completions` - Blocks until inference completes
- **Asynchronous**: `/v1/chat/completions/async` + `/v1/chat/completions/status/{call_id}` - Non-blocking job queue pattern

For production applications, use the async endpoints to maintain responsiveness and handle high concurrency.

---

## Async Workflow

1. **Submit Job**: POST to `/v1/chat/completions/async` → Receive `call_id`
2. **Poll Status**: GET `/v1/chat/completions/status/{call_id}` → Check job status
3. **Get Result**: When status is `200`, extract the generated caption

---

## Endpoints

### 1. Submit Async Job

**Endpoint**: `POST /v1/chat/completions/async`

**Request Body**:
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

## Python Integration Examples

### Basic Async Client

```python
import asyncio
import base64
import httpx
from pathlib import Path

API_BASE_URL = "https://maximuspookus--qwen3-vlm-caption-serve.modal.run"

async def submit_async_job(image_path: str, prompt: str = "Describe this satellite image in detail."):
    """Submit an async inference job."""
    # Load and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": "qwen-caption-special",
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
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/v1/chat/completions/async",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["call_id"]


async def poll_job_status(call_id: str, max_wait: int = 300, poll_interval: float = 2.0):
    """Poll for job result until complete or timeout."""
    async with httpx.AsyncClient() as client:
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job {call_id} did not complete within {max_wait}s")
            
            response = await client.get(
                f"{API_BASE_URL}/v1/chat/completions/status/{call_id}",
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


async def process_image_async(image_path: str, prompt: str = "Describe this satellite image in detail."):
    """Complete async workflow: submit job and wait for result."""
    call_id = await submit_async_job(image_path, prompt)
    print(f"Job submitted: {call_id}")
    
    result = await poll_job_status(call_id)
    return result


# Example usage
async def main():
    caption = await process_image_async("sample1.png")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Batch Processing Example

```python
import asyncio
import httpx
from typing import List, Dict

async def process_batch_async(image_paths: List[str], prompt: str = "Describe this satellite image."):
    """Process multiple images concurrently using async jobs."""
    # Submit all jobs
    call_ids = []
    async with httpx.AsyncClient() as client:
        for image_path in image_paths:
            # ... (submit job logic from above)
            call_id = await submit_async_job(image_path, prompt)
            call_ids.append((image_path, call_id))
    
    # Poll all jobs (with exponential backoff)
    results = {}
    async with httpx.AsyncClient() as client:
        pending = {call_id: image_path for image_path, call_id in call_ids}
        poll_interval = 1.0
        
        while pending:
            tasks = [
                client.get(f"{API_BASE_URL}/v1/chat/completions/status/{call_id}")
                for call_id in pending.keys()
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for call_id, response in zip(pending.keys(), responses):
                if isinstance(response, Exception):
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    image_path = pending.pop(call_id)
                    results[image_path] = result["choices"][0]["message"]["content"]
                elif response.status_code == 404:
                    image_path = pending.pop(call_id)
                    results[image_path] = None  # Expired
            
            if pending:
                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 10.0)  # Exponential backoff
    
    return results
```

---

### Error Handling Best Practices

```python
import httpx
from typing import Optional

async def safe_poll_job(call_id: str, max_retries: int = 3) -> Optional[str]:
    """Poll job with retry logic and proper error handling."""
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(
                    f"{API_BASE_URL}/v1/chat/completions/status/{call_id}",
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
const API_BASE_URL = "https://maximuspookus--qwen3-vlm-caption-serve.modal.run";

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
  imageBase64: string,
  prompt: string = "Describe this satellite image in detail."
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/v1/chat/completions/async`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "qwen-caption-special",
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
      max_tokens: 512,
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to submit job: ${response.statusText}`);
  }

  const data: AsyncJobResponse = await response.json();
  return data.call_id;
}

async function pollJobStatus(
  callId: string,
  maxWait: number = 300,
  pollInterval: number = 2000
): Promise<string> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWait * 1000) {
    const response = await fetch(
      `${API_BASE_URL}/v1/chat/completions/status/${callId}`
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

// Usage
async function processImage(imageBase64: string): Promise<string> {
  const callId = await submitAsyncJob(imageBase64);
  console.log(`Job submitted: ${callId}`);
  
  const result = await pollJobStatus(callId);
  return result;
}
```

---

## cURL Examples

### Submit Job
```bash
curl -X POST https://maximuspookus--qwen3-vlm-caption-serve.modal.run/v1/chat/completions/async \
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

### Poll Status
```bash
curl https://maximuspookus--qwen3-vlm-caption-serve.modal.run/v1/chat/completions/status/fc-abc123xyz
```

---

## Rate Limits & Best Practices

1. **Polling Interval**: Use 2-5 seconds between polls to avoid overwhelming the server
2. **Timeout**: Set reasonable timeouts (5-10 minutes for satellite images)
3. **Exponential Backoff**: Increase poll interval if job is taking longer
4. **Error Handling**: Always handle 404 (expired), 202 (pending), and 500 (server error)
5. **Concurrency**: Modal auto-scales, but batch submissions should be throttled client-side
6. **Job Retention**: Results are available for 7 days after completion

---

## Synchronous Endpoint (Legacy)

For backward compatibility, the synchronous endpoint is still available:

**Endpoint**: `POST /v1/chat/completions`

Same request format as async, but blocks until inference completes. Use only for:
- Testing
- Low-volume applications
- When you need immediate results and can afford blocking

---

## Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "model": "qwen-caption-special"
}
```

---

## Troubleshooting

### Job Stuck in Pending
- Check Modal dashboard for container status
- Verify GPU availability
- Increase poll timeout

### 404 Expired Error
- Jobs expire after 7 days
- Re-submit the job if needed

### 500 Server Error
- Check Modal logs
- Verify model is deployed correctly
- Ensure volume is mounted properly

---

## Model Information

- **Model Name**: `qwen-caption-special`
- **Base Model**: Qwen3-VL-8B-Instruct
- **Fine-tuned for**: Satellite imagery captioning
- **GPU**: A100-80GB
- **Max Image Size**: No hard limit (handles full-resolution satellite images)
- **Supported Formats**: PNG, JPEG (via base64 data URLs)

---

## Support

For issues or questions:
1. Check Modal dashboard: https://modal.com/apps/maximuspookus/main/deployed/qwen3-vlm-caption
2. Review deployment logs in Modal
3. Verify volume contains merged model: `vlm-weights-merged-caption-special`

