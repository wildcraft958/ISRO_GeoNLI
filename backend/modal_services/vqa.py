import modal

# --- CONFIGURATION ---
# Point to the folder where we saved the merged model
MERGED_MODEL_PATH = "/data/models/merged-qwen-vl-vqa"
MODEL_NAME = "qwen-vqa-special"  # The name clients will use

# Use the same volume as merge script
vol = modal.Volume.from_name("vlm-weights-merged-vqa-special")

# Standard vLLM image with latest version
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.4",
        "transformers>=4.45.0",  # For AutoProcessor
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "pydantic>=2.0",  # For request/response schemas
        "pillow",
        "httpx",
        "prometheus-fastapi-instrumentator"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen3-vlm-vqa")

# Global LLM instance (cached per container)
_llm_cache = None


@app.function(
    image=image,
    gpu="A100",  # 80GB A100 - sufficient for 8B models with FP16
    volumes={"/data/models": vol},
    timeout=600,  # 10 minutes per job
    scaledown_window=300,  # Keep container warm for 5 min
)
def process_inference_job(
    messages: list,
    image_bytes: bytes | None,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Process a single inference job. LLM is cached per container."""
    import io

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    global _llm_cache
    
    # Load LLM once per container (cached in GPU memory)
    if _llm_cache is None:
        print("🔄 Loading LLM in job container...")
        _llm_cache = LLM(
            model=MERGED_MODEL_PATH,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.88,  # Reduced for safety margin
            tensor_parallel_size=1,
            enforce_eager=True,
            max_num_seqs=4,
        )
        print("✅ LLM loaded in job container")
    
    # Load processor once per container
    if not hasattr(process_inference_job, '_processor'):
        print("🔄 Loading processor in job container...")
        process_inference_job._processor = AutoProcessor.from_pretrained(
            MERGED_MODEL_PATH, 
            trust_remote_code=True
        )
        print("✅ Processor loaded in job container")
    
    llm = _llm_cache
    processor = process_inference_job._processor

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Convert image_bytes back to PIL Image if present
    image = None
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes))

    # Format messages using processor's chat template
    try:
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback: format manually with image placeholder
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted_prompt += f"{role}: {content}\n"
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        formatted_prompt += f"{role}: {item.get('text', '')}\n"
                    elif item.get("type") == "image_url":
                        formatted_prompt += "<image>\n"  # Image placeholder for Qwen3-VL
        print(f"⚠️ Using fallback formatting: {e}")

    # vLLM expects prompt as dict with "prompt" and "multi_modal_data" keys for multimodal
    if image:
        outputs = llm.generate(
            [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling_params,
        )
    else:
        outputs = llm.generate(
            [formatted_prompt],
            sampling_params=sampling_params,
        )

    return outputs[0].outputs[0].text


@app.function(
    image=image,
    gpu="A100",  # 80GB A100 - sufficient for 8B models with FP16
    volumes={"/data/models": vol},
    scaledown_window=300,
    # min_containers removed to stay within 2 A100 limit (containers spin up on-demand)
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    import base64
    import io
    from typing import List, Literal, Optional, Union

    import fastapi
    import httpx
    from fastapi import HTTPException, Path
    from fastapi.responses import JSONResponse
    from PIL import Image
    from prometheus_fastapi_instrumentator import Instrumentator
    from pydantic import BaseModel, Field
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # =========================================================================
    # Pydantic Models for OpenAPI/Swagger Documentation
    # =========================================================================
    
    class ImageUrl(BaseModel):
        """Image URL object."""
        url: str = Field(..., description="Image URL or base64 data URI")

    class TextContent(BaseModel):
        """Text content item."""
        type: Literal["text"] = "text"
        text: str = Field(..., description="Text content")

    class ImageContent(BaseModel):
        """Image content item."""
        type: Literal["image_url"] = "image_url"
        image_url: ImageUrl = Field(..., description="Image URL object")

    class Message(BaseModel):
        """Chat message in OpenAI format."""
        role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
        content: Union[str, List[Union[TextContent, ImageContent]]] = Field(
            ..., 
            description="Message content (string or list of content items)"
        )

    class ChatRequest(BaseModel):
        """OpenAI-compatible chat completion request for VQA."""
        model: str = Field(default=MODEL_NAME, description="Model name")
        messages: List[Message] = Field(..., description="List of messages")
        max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
        temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0 for deterministic)")

        class Config:
            json_schema_extra = {
                "example": {
                    "model": "qwen-vqa-special",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "How many ships are visible in this harbor?"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                            ]
                        }
                    ],
                    "max_tokens": 128,
                    "temperature": 0.0
                }
            }

    class AssistantMessage(BaseModel):
        """Assistant response message."""
        role: Literal["assistant"] = "assistant"
        content: str = Field(..., description="Generated answer")

    class Choice(BaseModel):
        """Chat completion choice."""
        index: int = 0
        message: AssistantMessage
        finish_reason: Literal["stop", "length"] = "stop"

    class Usage(BaseModel):
        """Token usage statistics."""
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ChatCompletionResponse(BaseModel):
        """OpenAI-compatible chat completion response."""
        id: str = "chatcmpl-modal"
        object: Literal["chat.completion"] = "chat.completion"
        model: str
        choices: List[Choice]
        usage: Usage

    class AsyncJobResponse(BaseModel):
        """Async job submission response."""
        call_id: str = Field(..., description="Job ID for polling")
        status: Literal["pending"] = "pending"
        model: str

    class AsyncPendingResponse(BaseModel):
        """Async job pending status."""
        status: Literal["pending"] = "pending"
        call_id: str

    class ErrorResponse(BaseModel):
        """Error response."""
        error: str
        traceback: Optional[str] = None

    class HealthResponse(BaseModel):
        """Health check response."""
        status: Literal["ok"] = "ok"
        model: str

    # =========================================================================
    # Initialize vLLM
    # =========================================================================
    
    print("🚀 Initializing vLLM with Merged Vision Weights...")

    llm = LLM(
        model=MERGED_MODEL_PATH,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.88,  # Reduced for safety margin
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=4,
    )

    # Load processor for chat template formatting
    processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)

    print(f"✅ Engine ready! Serving as: {MODEL_NAME}")
    
    # =========================================================================
    # FastAPI App
    # =========================================================================
    
    app = fastapi.FastAPI(
        title="Qwen3-VL VQA API",
        description="OpenAI-compatible API for Visual Question Answering on satellite imagery using Qwen3-VL",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Instrument Prometheus metrics at /metrics
    Instrumentator().instrument(app).expose(app)

    http_client = httpx.AsyncClient(timeout=30)

    @app.on_event("shutdown")
    async def shutdown_http_client():
        await http_client.aclose()

    # =========================================================================
    # Helper Functions
    # =========================================================================

    async def extract_messages_and_image(messages: List[Message]):
        """Extract messages in OpenAI format and image as PIL Image."""
        processed_messages = []
        image = None

        for msg in messages:
            role = msg.role
            content = msg.content
            
            # Handle string content (text-only)
            if isinstance(content, str):
                processed_messages.append({
                    "role": role,
                    "content": content
                })
                continue

            # Handle list content (multimodal)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, TextContent):
                        new_content.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent) and image is None:
                        # Extract image
                        image_url = item.image_url.url
                        if not image_url:
                            continue
                        
                        if image_url.startswith("data:"):
                            # Base64 image
                            _, b64_data = image_url.split(",", 1)
                            image_bytes = base64.b64decode(b64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                        else:
                            # URL image
                            resp = await http_client.get(image_url)
                            resp.raise_for_status()
                            image = Image.open(io.BytesIO(resp.content))
                        
                        # Keep image_url in content for vLLM to process
                        new_content.append({"type": "image_url", "image_url": {"url": image_url}})
                    elif isinstance(item, dict):
                        # Handle raw dict format (backward compatibility)
                        if item.get("type") == "text":
                            new_content.append(item)
                        elif item.get("type") == "image_url" and image is None:
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url:
                                if image_url.startswith("data:"):
                                    _, b64_data = image_url.split(",", 1)
                                    image_bytes = base64.b64decode(b64_data)
                                    image = Image.open(io.BytesIO(image_bytes))
                                else:
                                    resp = await http_client.get(image_url)
                                    resp.raise_for_status()
                                    image = Image.open(io.BytesIO(resp.content))
                                new_content.append(item)
                
                processed_messages.append({
                    "role": role,
                    "content": new_content
                })
        
        return processed_messages, image

    # =========================================================================
    # API Endpoints
    # =========================================================================

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        responses={
            200: {"model": ChatCompletionResponse, "description": "Successful completion"},
            500: {"model": ErrorResponse, "description": "Server error"}
        },
        summary="Visual Question Answering",
        description="Answer questions about satellite imagery using OpenAI-compatible format."
    )
    async def chat(req: ChatRequest):
        """Synchronous VQA endpoint."""
        try:
            # Extract messages in OpenAI format and image
            processed_messages, image = await extract_messages_and_image(req.messages)

            sampling_params = SamplingParams(
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )

            # Format messages using processor's chat template
            try:
                formatted_prompt = processor.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback: format manually with image placeholder
                formatted_prompt = ""
                for msg in processed_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        formatted_prompt += f"{role}: {content}\n"
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                formatted_prompt += f"{role}: {item.get('text', '')}\n"
                            elif item.get("type") == "image_url":
                                formatted_prompt += "<image>\n"
                print(f"⚠️ Using fallback formatting: {e}")

            # Generate with vLLM
            if image:
                outputs = llm.generate(
                    [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
                    sampling_params=sampling_params,
                )
            else:
                outputs = llm.generate(
                    [formatted_prompt],
                    sampling_params=sampling_params,
                )

            generated_text = outputs[0].outputs[0].text

            # Estimate tokens
            prompt_text = ""
            for msg in processed_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt_text += content + " "
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            prompt_text += item.get("text", "") + " "
            
            prompt_tokens = len(prompt_text.split()) if prompt_text else 0
            completion_tokens = len(generated_text.split())

            return ChatCompletionResponse(
                id="chatcmpl-modal",
                model=req.model,
                choices=[
                    Choice(
                        index=0,
                        message=AssistantMessage(content=generated_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )

        except Exception as e:
            import traceback
            return JSONResponse(
                content={"error": str(e), "traceback": traceback.format_exc()},
                status_code=500
            )

    @app.post(
        "/v1/chat/completions/async",
        response_model=AsyncJobResponse,
        responses={
            200: {"model": AsyncJobResponse, "description": "Job submitted"},
            500: {"model": ErrorResponse, "description": "Server error"},
            503: {"model": ErrorResponse, "description": "Service unavailable"}
        },
        summary="Async VQA",
        description="Submit an async VQA job and get a call_id for polling."
    )
    async def chat_async(req: ChatRequest):
        """Submit an async inference job and return call_id for polling."""
        try:
            # Extract messages and image
            processed_messages, image = await extract_messages_and_image(req.messages)
            
            # Convert PIL Image to bytes for serialization
            image_bytes = None
            if image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()

            # Look up the deployed function
            try:
                inference_job = modal.Function.from_name(
                    "qwen3-vlm-vqa", "process_inference_job"
                )
            except Exception as e:
                return JSONResponse(
                    content={
                        "error": f"Function not found. Deploy with 'modal deploy vqa.py'. Error: {str(e)}"
                    },
                    status_code=503
                )

            # Spawn the job
            call = await inference_job.spawn.aio(
                messages=processed_messages,
                image_bytes=image_bytes,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )

            return AsyncJobResponse(
                call_id=call.object_id,
                status="pending",
                model=req.model
            )

        except Exception as e:
            import traceback
            return JSONResponse(
                content={"error": str(e), "traceback": traceback.format_exc()},
                status_code=500
            )

    @app.get(
        "/v1/chat/completions/status/{call_id}",
        responses={
            200: {"model": ChatCompletionResponse, "description": "Job completed"},
            202: {"model": AsyncPendingResponse, "description": "Job still processing"},
            404: {"model": ErrorResponse, "description": "Job expired or not found"},
            500: {"model": ErrorResponse, "description": "Server error"}
        },
        summary="Poll Job Status",
        description="Poll for async job result. Returns 202 if pending, 200 if complete."
    )
    async def get_job_status(call_id: str = Path(..., description="Job call ID for polling")):
        """Poll for job result. Returns 202 if pending, 200 if complete, 404 if expired."""
        try:
            function_call = modal.FunctionCall.from_id(call_id)

            try:
                # Poll non-blocking (timeout=0)
                result = await function_call.get.aio(timeout=0)
                generated_text = result

                return ChatCompletionResponse(
                    id=f"chatcmpl-{call_id}",
                    model=MODEL_NAME,
                    choices=[
                        Choice(
                            index=0,
                            message=AssistantMessage(content=generated_text),
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=0,
                        completion_tokens=len(generated_text.split()),
                        total_tokens=len(generated_text.split())
                    )
                )

            except modal.exception.OutputExpiredError:
                return JSONResponse(
                    content={"error": "Job result expired (older than 7 days)"},
                    status_code=404
                )
            except TimeoutError:
                # Job still pending
                return JSONResponse(
                    content={"status": "pending", "call_id": call_id},
                    status_code=202
                )

        except Exception as e:
            import traceback
            return JSONResponse(
                content={"error": str(e), "traceback": traceback.format_exc()},
                status_code=500
            )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        description="Check if the service is running."
    )
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", model=MODEL_NAME)

    return app
