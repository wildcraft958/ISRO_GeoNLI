import modal
import base64
import io
from PIL import Image

# --- CONFIGURATION ---
MERGED_MODEL_PATH = "/data/models/merged-qwen3vl-sar"
MODEL_NAME = "qwen3vl-sar"

vol = modal.Volume.from_name("vlm-weights-merge-qwen3vl-sar")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.4",
        "transformers>=4.45.0",
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "pillow",
        "prometheus-fastapi-instrumentator",
        "pydantic>=2.0.0",
        "httpx"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen3-vl-sar")

@app.function(
    image=image,
    gpu="A100",
    volumes={"/data/models": vol},
    scaledown_window=1200,
    min_containers=0,
)
@modal.asgi_app()
def serve():
    import fastapi
    from fastapi.responses import JSONResponse
    from prometheus_fastapi_instrumentator import Instrumentator
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from pydantic import BaseModel, Field
    from typing import Optional, List
    import httpx
    from urllib.parse import urlparse

    # Pydantic Models
    class PredictRequest(BaseModel):
        """Request model for single prediction."""
        system_prompt: Optional[str] = Field(
            default="You are a helpful vision assistant.",
            description="System prompt to set the assistant's behavior"
        )
        user_prompt: str = Field(
            ...,
            description="User's question or prompt",
            min_length=1
        )
        image: Optional[str] = Field(
            default=None,
            description=(
                "Image input in one of the following formats:\n"
                "- Base64 encoded string (e.g., 'iVBORw0KGgoAAAANSUhEUgAA...')\n"
                "- Data URL (e.g., 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...')\n"
                "- Image URL (e.g., 'https://example.com/image.jpg' or 'http://example.com/image.png')"
            )
        )
        max_tokens: int = Field(
            default=512,
            description="Maximum number of tokens to generate",
            ge=1,
            le=4096
        )
        temperature: float = Field(
            default=0.2,
            description="Sampling temperature (0.0 to 2.0)",
            ge=0.0,
            le=2.0
        )

        class Config:
            json_schema_extra = {
                "example": {
                    "system_prompt": "You are a helpful vision assistant.",
                    "user_prompt": "What is in this image?",
                    "image": "https://example.com/image.jpg",
                    "max_tokens": 512,
                    "temperature": 0.2
                },
                "examples": [
                    {
                        "system_prompt": "You are a helpful vision assistant.",
                        "user_prompt": "What is in this image?",
                        "image": "https://example.com/image.jpg",
                        "max_tokens": 512,
                        "temperature": 0.2
                    },
                    {
                        "system_prompt": "You are a helpful vision assistant.",
                        "user_prompt": "Describe this image.",
                        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                        "max_tokens": 1024,
                        "temperature": 0.2
                    }
                ]
            }

    class PredictResponse(BaseModel):
        """Response model for single prediction."""
        response: str = Field(..., description="Generated text response")
        model: str = Field(..., description="Model name used for generation")
        status: str = Field(..., description="Status of the request")

        class Config:
            json_schema_extra = {
                "example": {
                    "response": "This image shows...",
                    "model": "qwen3vl-sar",
                    "status": "success"
                }
            }

    class ErrorResponse(BaseModel):
        """Error response model."""
        error: str = Field(..., description="Error message")
        traceback: Optional[str] = Field(default=None, description="Error traceback")

    class BatchRequestItem(BaseModel):
        """Single item in batch prediction request."""
        system_prompt: Optional[str] = Field(
            default="You are a helpful vision assistant.",
            description="System prompt to set the assistant's behavior"
        )
        user_prompt: str = Field(
            ...,
            description="User's question or prompt",
            min_length=1
        )
        image: Optional[str] = Field(
            default=None,
            description=(
                "Image input in one of the following formats:\n"
                "- Base64 encoded string (e.g., 'iVBORw0KGgoAAAANSUhEUgAA...')\n"
                "- Data URL (e.g., 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...')\n"
                "- Image URL (e.g., 'https://example.com/image.jpg' or 'http://example.com/image.png')"
            )
        )
        max_tokens: int = Field(
            default=512,
            description="Maximum number of tokens to generate",
            ge=1,
            le=4096
        )
        temperature: float = Field(
            default=0.2,
            description="Sampling temperature (0.0 to 2.0)",
            ge=0.0,
            le=2.0
        )

    class BatchPredictRequest(BaseModel):
        """Request model for batch prediction."""
        requests: List[BatchRequestItem] = Field(
            ...,
            description="List of prediction requests",
            min_length=1,
            max_length=100
        )

        class Config:
            json_schema_extra = {
                "example": {
                    "requests": [
                        {
                            "system_prompt": "You are a helpful vision assistant.",
                            "user_prompt": "What is in this image?",
                            "image": "https://example.com/image.jpg",
                            "max_tokens": 512,
                            "temperature": 0.2
                        }
                    ]
                }
            }

    class BatchResultItem(BaseModel):
        """Single result item in batch prediction response."""
        index: int = Field(..., description="Index of the request in the batch")
        response: Optional[str] = Field(default=None, description="Generated text response")
        error: Optional[str] = Field(default=None, description="Error message if failed")
        status: str = Field(..., description="Status: 'success' or 'failed'")

    class BatchPredictResponse(BaseModel):
        """Response model for batch prediction."""
        results: List[BatchResultItem] = Field(..., description="List of prediction results")
        model: str = Field(..., description="Model name used for generation")

    class HealthResponse(BaseModel):
        """Health check response model."""
        status: str = Field(..., description="Service status")
        model: str = Field(..., description="Model name")

    def load_image_from_input(image_input: str) -> tuple[Image.Image, str]:
        """
        Load image from either base64 string or URL.
        Returns (PIL Image, base64_string_for_message_formatting)
        """
        # Check if it's a URL - check for http:// or https:// at the start
        is_url = image_input.strip().startswith(('http://', 'https://'))
        
        if is_url:
            # Download image from URL using httpx (better default headers than requests)
            try:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    response = client.get(image_input)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    # Convert to base64 for message formatting
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    b64_string = base64.b64encode(img_buffer.read()).decode('utf-8')
                    return image, b64_string
            except Exception as e:
                raise ValueError(f"Failed to download image from URL '{image_input[:100]}...': {str(e)}")
        else:
            # Try to decode as base64
            original_input = image_input
            base64_part = image_input
            
            # Remove data URL prefix if present (data:image/...;base64,)
            if image_input.startswith('data:image/') and ';base64,' in image_input:
                base64_part = image_input.split(';base64,', 1)[1]
            
            try:
                image_bytes = base64.b64decode(base64_part, validate=True)
                image = Image.open(io.BytesIO(image_bytes))
                # Return the base64 part (without data URL prefix) for message formatting
                return image, base64_part
            except Exception as e:
                # If base64 decode fails, check if it might be a URL without http:// prefix
                if any(char in original_input for char in ['/', '.', ':', '?', '&', '=']) and not original_input.startswith('data:'):
                    raise ValueError(
                        f"Failed to decode as base64. If this is a URL, ensure it starts with 'http://' or 'https://'. "
                        f"Original error: {str(e)}"
                    )
                raise ValueError(f"Failed to decode base64 image: {str(e)}")

    print("🚀 Initializing vLLM with Merged Vision Weights...")

    llm = LLM(
        model=MERGED_MODEL_PATH,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.94,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=100000,  # Limit sequence length to fit KV cache memory
    )

    processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    print(f"✅ Engine ready! Serving as: {MODEL_NAME}")
    
    app = fastapi.FastAPI(
        title=f"{MODEL_NAME} API",
        description=f"Vision Language Model API for {MODEL_NAME} using vLLM",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    Instrumentator().instrument(app).expose(app)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Bad request"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Single Prediction",
        description="Generate a response from the vision language model for a single request with optional image input."
    )
    async def predict(request: PredictRequest):
        """
        Generate a response from the vision language model.
        
        - **system_prompt**: Optional system prompt to set assistant behavior
        - **user_prompt**: Required user question or prompt
        - **image**: Optional image input. Supports:
          - Base64 encoded string (e.g., 'iVBORw0KGgoAAAANSUhEUgAA...')
          - Data URL format (e.g., 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...')
          - Image URL (e.g., 'https://example.com/image.jpg' or 'http://example.com/image.png')
        - **max_tokens**: Maximum tokens to generate (default: 512)
        - **temperature**: Sampling temperature (default: 0.2)
        """
        try:
            system_prompt = request.system_prompt or "You are a helpful vision assistant."
            user_prompt = request.user_prompt
            image_input = request.image
            max_tokens = request.max_tokens
            temperature = request.temperature

            # Load image from base64 string or URL if provided
            image = None
            b64_image = None
            if image_input:
                try:
                    image, b64_image = load_image_from_input(image_input)
                except ValueError as e:
                    return JSONResponse(
                        content=ErrorResponse(
                            error=str(e)
                        ).model_dump(),
                        status_code=400,
                    )

            # Build messages array
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message with image
            if image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": user_prompt
                })

            # Format prompt using processor's chat template
            try:
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback formatting
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
                                formatted_prompt += "<image>\n"
                print(f"⚠️ Using fallback formatting: {e}")

            # Generate response
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )

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

            return PredictResponse(
                response=generated_text,
                model=MODEL_NAME,
                status="success"
            )

        except Exception as e:
            import traceback
            return JSONResponse(
                content=ErrorResponse(
                    error=str(e),
                    traceback=traceback.format_exc()
                ).model_dump(),
                status_code=500,
            )

    @app.post(
        "/predict-batch",
        response_model=BatchPredictResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Bad request"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Batch Prediction",
        description="Process multiple prediction requests in a single batch. Returns results for all requests with individual status indicators."
    )
    async def predict_batch(request: BatchPredictRequest):
        """
        Process multiple prediction requests in a batch.
        
        - **requests**: List of prediction requests (max 100)
        - Each request follows the same structure as /predict endpoint
        - Image input supports base64 strings, data URLs, or image URLs (http:// or https://)
        """
        try:
            requests = request.requests

            results = []
            for idx, item in enumerate(requests):
                try:
                    system_prompt = item.system_prompt or "You are a helpful vision assistant."
                    user_prompt = item.user_prompt
                    image_input = item.image
                    max_tokens = item.max_tokens
                    temperature = item.temperature

                    # Load image from base64 string or URL if provided
                    image = None
                    b64_image = None
                    if image_input:
                        try:
                            image, b64_image = load_image_from_input(image_input)
                        except ValueError as e:
                            results.append(BatchResultItem(
                                index=idx,
                                error=f"Failed to load image: {str(e)}",
                                status="failed"
                            ))
                            continue

                    # Build messages
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    
                    if image:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                            ]
                        })
                    else:
                        messages.append({"role": "user", "content": user_prompt})

                    # Format and generate
                    formatted_prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    sampling_params = SamplingParams(
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    if image:
                        outputs = llm.generate(
                            [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
                            sampling_params=sampling_params,
                        )
                    else:
                        outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)

                    generated_text = outputs[0].outputs[0].text
                    results.append(BatchResultItem(
                        index=idx,
                        response=generated_text,
                        status="success"
                    ))

                except Exception as e:
                    results.append(BatchResultItem(
                        index=idx,
                        error=str(e),
                        status="failed"
                    ))

            return BatchPredictResponse(
                results=results,
                model=MODEL_NAME
            )

        except Exception as e:
            import traceback
            return JSONResponse(
                content=ErrorResponse(
                    error=str(e),
                    traceback=traceback.format_exc()
                ).model_dump(),
                status_code=500,
            )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        description="Check if the service is running and return model information."
    )
    async def health():
        """Health check endpoint to verify service status."""
        return HealthResponse(status="ok", model=MODEL_NAME)

    return app