# vllm_modal_deploy.py
#
# Deploy Qwen3-VL-32B-Thinking model with vLLM on Modal
#
# This script deploys a vLLM server with Qwen3-VL-32B-Thinking model,
# using Modal volumes to cache model weights for fast loading.
#
# Usage:
#   # First, download model to volume (one-time):
#   modal run vllm_modal_deploy.py::download_model_to_volume
#
#   # Then deploy the server:
#   modal deploy vllm_modal_deploy.py
#
# After deployment, the endpoint will be available at:
#   https://your-username--qwen3-vl-vllm-server.modal.run/v1
#
# Use this endpoint in SAM3 agent requests:
#   {
#     "llm_config": {
#       "base_url": "https://your-username--qwen3-vl-vllm-server.modal.run/v1",
#       "model": "Qwen/Qwen3-VL-32B-Thinking",
#       "api_key": "",
#       "name": "qwen3-vl-32b-thinking-modal"
#     }
#   }

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import modal
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Modal app + image
# ------------------------------------------------------------------------------

app = modal.App("qwen3-vl-vllm-server-30B")

# Create volume for model weights (~64GB for 32B model in bfloat16)
# This will cache the model weights to avoid re-downloading on each deployment
MODEL_VOLUME = modal.Volume.from_name("qwen3-vl-30b-instruct-weights", create_if_missing=True)

# Model configuration
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HF_CACHE_DIR = "/root/.cache/huggingface"  # Standard HuggingFace cache location

# GPU Configuration
# Available GPUs in Modal: "B200", "H200", "H100", "A100", "L40S", "A10", "L4", "T4"
# For 30B model: Use tensor parallelism (2-4 GPUs) to avoid OOM
# Note: B200 and H200 may not be available in all regions. If unavailable, use H100 or A100.
GPU_TYPE = "A100-80GB"  # A100 80GB recommended for 30B model
NUM_GPUS = 1  # Use 2 GPUs with tensor parallelism to avoid OOM (increase to 3-4 if still OOM)

# Build GPU string for Modal
if NUM_GPUS == 1:
    GPU_SPEC = GPU_TYPE
else:
    GPU_SPEC = f"{GPU_TYPE}:{NUM_GPUS}"

# Build custom image with vLLM and vision model support
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
    )
    # Install torch first (required for flash-attn)
    .pip_install(
        "torch>=2.1.0",
    )
    # Install flash-attn after torch (requires torch to be installed first)
    # .pip_install(
    #     "flash-attn>=2.5.0",  # Flash attention for faster inference
    # )
    # Install remaining dependencies
    .pip_install(
        "vllm==0.12.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "pillow>=10.0.0",
        "qwen-vl-utils>=0.0.14",  # Required for Qwen3-VL
        "fastapi>=0.100.0",
        "httpx>=0.24.0",
        "requests>=2.31.0",  # For health checks during startup
    )
    .env({"HF_HOME": HF_CACHE_DIR})
)

# ------------------------------------------------------------------------------
# Pydantic Models for API Documentation (must be at module level for Swagger)
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Server status", example="ok")
    model: str = Field(..., description="Model identifier", example=MODEL_ID)
    parser: str = Field(..., description="Parser type (automatic or manual)", example="automatic")
    error: Optional[str] = Field(None, description="Error message if status is error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model": MODEL_ID,
                "parser": "automatic"
            }
        }

class ModelInfo(BaseModel):
    """Model information"""
    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(default=0, description="Creation timestamp")
    owned_by: str = Field(default="vllm", description="Model owner")

class ModelListResponse(BaseModel):
    """List of available models"""
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": 0,
                        "owned_by": "vllm"
                    }
                ]
            }
        }

class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Message role (system, user, assistant, tool)", example="user")
    content: Union[str, List[Dict[str, Any]]] = Field(
        ..., 
        description="Message content. Can be a string (text) or a list of content parts for multimodal messages (text and images).",
        example="What is the capital of France?"
    )
    name: Optional[str] = Field(None, description="Name of the author (for tool calls)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID (for tool messages)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the capital of France?"
            }
        }

class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)"""
    model: str = Field(default=MODEL_ID, description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(8192, ge=1, le=8192, description="Maximum tokens to generate (capped at 8192)")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of available tools for function calling")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice strategy")
    max_thinking_tokens: Optional[int] = Field(2048, ge=0, le=2048, description="Maximum thinking tokens (capped at 2048)")
    
    class Config:
        # Allow extra fields for OpenAI compatibility (e.g., response_format, etc.)
        extra = "allow"
        json_schema_extra = {
            "example": {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 100,
                "max_thinking_tokens": 2048
            }
        }

class ChatChoice(BaseModel):
    """Chat completion choice"""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")

class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")

class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI-compatible)"""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 9,
                    "total_tokens": 18
                }
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(None, description="Error traceback (if available)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Request to vLLM server timed out"
            }
        }

# ------------------------------------------------------------------------------
# vLLM Server Deployment
# ------------------------------------------------------------------------------

# Feature flag: Use vLLM's automatic parser (True) or manual parsing (False)
USE_AUTOMATIC_PARSER = True  # Set to False to fallback to manual parsing

@app.function(
    gpu=GPU_SPEC,  # Use string format: "H100", "H100:2", "A100:4", etc.
    image=image,
    volumes={HF_CACHE_DIR: MODEL_VOLUME},  # Mount volume at HuggingFace cache location
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # For HF_TOKEN if model is gated
    ],
    timeout=7200,  # 2 hour timeout for model loading (32B model takes longer)
    scaledown_window=3600,  # Keep container alive for 1 hour after last request
    min_containers=1,  # Keep at least 1 container always running (persistent model)
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def vllm_server():
    """
    Deploy vLLM server with Qwen3-VL-32B-Thinking model.
    
    This function uses vLLM's built-in OpenAI server with automatic tool call parsing
    (when USE_AUTOMATIC_PARSER=True) or falls back to custom FastAPI endpoint with
    manual parsing (when USE_AUTOMATIC_PARSER=False).
    
    GPU Configuration:
    - Single GPU: Set NUM_GPUS=1, GPU_TYPE="A100" (A100 80GB recommended for 32B model)
    - Multiple GPUs: Set NUM_GPUS=2+ for tensor parallelism (if single GPU OOM)
    """
    import atexit
    import time
    import fastapi
    import httpx
    from fastapi.responses import JSONResponse
    from fastapi import Request
    from starlette.requests import ClientDisconnect
    
    print(f"🚀 Initializing vLLM server with {MODEL_ID}...")
    print(f"GPU Configuration: {GPU_SPEC} ({NUM_GPUS} GPU(s))")
    print(f"HuggingFace cache directory: {HF_CACHE_DIR}")
    print(f"Automatic parser: {'ENABLED' if USE_AUTOMATIC_PARSER else 'DISABLED (manual parsing)'}")
    
    # Set HuggingFace environment variables
    os.environ["HF_HOME"] = HF_CACHE_DIR
    model_cache_path = Path(HF_CACHE_DIR) / "hub"
    os.environ["TRANSFORMERS_CACHE"] = str(model_cache_path)
    os.environ["HF_HUB_CACHE"] = str(model_cache_path)
    
    # Create FastAPI app for proxy
    from fastapi.middleware.cors import CORSMiddleware
    app = fastapi.FastAPI(
        title="Qwen3-VL vLLM Server",
        description="OpenAI-compatible API server for Qwen3-VL-30B-A3B-Instruct model powered by vLLM",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware to allow frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for API access
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    http_client = httpx.AsyncClient(timeout=300.0)
    
    vllm_process = None
    vllm_port = 8000
    vllm_url = f"http://localhost:{vllm_port}"
    
    if USE_AUTOMATIC_PARSER:
        # Start vLLM's built-in OpenAI server with automatic tool call parsing
        print("🚀 Starting vLLM OpenAI server with automatic tool call parsing...")
        
        # Build vLLM serve command
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_ID,
            "--trust-remote-code",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.85",  # Reduced from 0.95 to avoid OOM
            "--max-model-len", "32768",  # Reduced from 60K to 32K to save memory (increase if needed)
            "--limit-mm-per-prompt", '{"image": 4}',  # Increased to 4 for examine_each_mask (needs 3 images: raw, masked, zoomed)
            "--enforce-eager",  # Disables CUDA graphs to save memory
            "--max-num-seqs", "2",  # Further reduced batch size for memory efficiency
            "--max-num-batched-tokens", "4096",  # Reduced to save memory
            "--swap-space", "4",  # Enable 4GB CPU swap space for overflow
            "--port", str(vllm_port),
            "--host", "0.0.0.0",
        ]
        
        # Add tensor parallelism if multiple GPUs (required for 30B model to avoid OOM)
        if NUM_GPUS > 1:
            vllm_cmd.extend(["--tensor-parallel-size", str(NUM_GPUS)])
            print(f"✓ Tensor parallelism enabled: {NUM_GPUS} GPUs (splits model across GPUs to avoid OOM)")
        else:
            print("⚠️  WARNING: Single GPU may cause OOM for 30B model. Consider setting NUM_GPUS=2 or higher.")
        
        # Enable flash attention (installed in image by default)
        # try:
        #     # import flash_attn
        #     vllm_cmd.append("--enable-flash-attn")
        #     print("✓ Flash attention enabled (default)")
        # except ImportError:
        #     print("⚠ Flash attention not available despite being in image - using default attention")
        #     print("   This may indicate an installation issue with flash-attn")
        
        print(f"📋 vLLM command: {' '.join(vllm_cmd)}")
        
        # Start vLLM server as subprocess
        try:
            # Collect all output for better error reporting
            output_lines = []
            
            vllm_process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Wait for server to be ready (check health endpoint)
            print("⏳ Waiting for vLLM server to start...")
            max_wait = 600  # 10 minutes max wait
            wait_interval = 2
            waited = 0
            
            # Use synchronous requests for health checks during startup
            import requests
            health_client = requests.Session()
            
            # Start reading output in background
            import threading
            def read_output():
                if vllm_process.stdout:
                    for line in iter(vllm_process.stdout.readline, ''):
                        if line:
                            output_lines.append(line)
                            # Print important lines in real-time
                            if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                                print(f"   vLLM: {line.rstrip()}")
            
            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()
            
            while waited < max_wait:
                # Check if process is still running
                if vllm_process.poll() is not None:
                    # Process died, wait a bit for output thread to finish
                    time.sleep(1)
                    # Try to get any remaining output
                    try:
                        remaining, _ = vllm_process.communicate(timeout=2)
                        if remaining:
                            output_lines.append(remaining)
                    except subprocess.TimeoutExpired:
                        vllm_process.kill()
                        remaining, _ = vllm_process.communicate()
                        if remaining:
                            output_lines.append(remaining)
                    
                    full_output = ''.join(output_lines)
                    print(f"❌ vLLM server process exited with code {vllm_process.returncode}")
                    print(f"   Full output ({len(full_output)} chars):")
                    print("=" * 80)
                    # Show last 5000 chars (more than before)
                    print(full_output[-5000:] if len(full_output) > 5000 else full_output)
                    print("=" * 80)
                    health_client.close()
                    raise RuntimeError(f"vLLM server failed to start. Exit code: {vllm_process.returncode}. See output above.")
                
                # Try to connect to health endpoint
                try:
                    resp = health_client.get(f"{vllm_url}/health", timeout=2)
                    if resp.status_code == 200:
                        print("✅ vLLM server is ready!")
                        health_client.close()
                        break
                except (requests.exceptions.RequestException, ConnectionError):
                    # Server not ready yet, wait and retry
                    time.sleep(wait_interval)
                    waited += wait_interval
                    if waited % 30 == 0:
                        print(f"   Still waiting... ({waited}s)")
            
            health_client.close()
            
            if waited >= max_wait:
                raise RuntimeError(f"vLLM server did not become ready within {max_wait} seconds")
                
        except Exception as e:
            import traceback
            print(f"❌ Failed to start vLLM server: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            if vllm_process:
                vllm_process.terminate()
            raise
    
    # Cleanup function
    def cleanup():
        if vllm_process:
            print("🛑 Shutting down vLLM server...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_process.kill()
    
    atexit.register(cleanup)
    
    @app.on_event("shutdown")
    async def shutdown():
        cleanup()
        await http_client.aclose()
    
    # Health endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Check the health status of the vLLM server and model",
        tags=["Health"]
    )
    async def health() -> HealthResponse:
        """
        Health check endpoint.
        
        Returns the status of the server, model identifier, and parser type.
        """
        if USE_AUTOMATIC_PARSER and vllm_process:
            # Check vLLM server health
            try:
                resp = await http_client.get(f"{vllm_url}/health", timeout=2)
                if resp.status_code == 200:
                    return HealthResponse(
                        status="ok",
                        model=MODEL_ID,
                        parser="automatic"
                    )
            except:
                return HealthResponse(
                    status="error",
                    model=MODEL_ID,
                    parser="automatic",
                    error="vLLM server not responding"
                )
        return HealthResponse(
            status="ok",
            model=MODEL_ID,
            parser="manual"
        )
    
    # Models endpoint
    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        summary="List models",
        description="List all available models",
        tags=["Models"]
    )
    async def list_models() -> ModelListResponse:
        """
        List all available models.
        
        Returns a list of models available on this server.
        """
        if USE_AUTOMATIC_PARSER:
            # Proxy to vLLM server
            try:
                resp = await http_client.get(f"{vllm_url}/v1/models", timeout=10)
                data = resp.json()
                # Convert to our response model if possible
                if "data" in data and isinstance(data["data"], list):
                    models = [ModelInfo(**item) for item in data["data"]]
                    return ModelListResponse(object=data.get("object", "list"), data=models)
                return ModelListResponse(**data)
            except Exception as e:
                print(f"❌ Error proxying to vLLM server: {e}")
                raise fastapi.HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to vLLM server: {e}"
                )
        else:
            # Manual response
            return ModelListResponse(
                object="list",
                data=[
                    ModelInfo(
                        id=MODEL_ID,
                        object="model",
                        created=0,
                        owned_by="vllm"
                    )
                ]
            )
    
    # Chat completions endpoint - proxy to vLLM server when using automatic parser
    @app.post(
        "/v1/chat/completions",
        summary="Create chat completion",
        description="Create a chat completion using the Qwen3-VL model. Supports text and multimodal inputs, as well as tool/function calling.",
        tags=["Chat"],
        response_model=Union[ChatCompletionResponse, Dict[str, Any]],
        responses={
            200: {"description": "Successful completion", "model": ChatCompletionResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            500: {"description": "Internal server error", "model": ErrorResponse},
            504: {"description": "Gateway timeout", "model": ErrorResponse}
        }
    )
    async def chat_completions(chat_request: ChatCompletionRequest):
        """
        Create a chat completion.
        
        This endpoint accepts OpenAI-compatible chat completion requests and proxies them
        to the underlying vLLM server. It supports:
        - Text and multimodal (image) inputs
        - Tool/function calling
        - Streaming responses
        - Custom temperature, top_p, and other parameters
        
        The request body follows the OpenAI Chat Completions API format.
        """
        if USE_AUTOMATIC_PARSER:
            # Proxy request to vLLM's built-in server (which handles tool calling automatically)
            try:
                # Convert Pydantic model to dict, including any extra fields
                # When extra="allow" is set in Config, model_dump() automatically includes extra fields
                req_body = chat_request.model_dump(exclude_none=True)
                
                print(f"📥 Proxying request to vLLM server: model={req_body.get('model', MODEL_ID)}, tools={len(req_body.get('tools', []))}")
                
                # Cap max_tokens to 8192 to allow longer reasoning outputs
                # For 64K context models, 8192 max_tokens leaves ~56K for input tokens
                if 'max_tokens' in req_body and req_body['max_tokens'] > 8192:
                    print(f"⚠️  Capping max_tokens from {req_body['max_tokens']} to 8192")
                    req_body['max_tokens'] = 8192
                
                # Set conservative max_thinking_tokens for fast responses (if not already set)
                # This ensures fast reasoning while maintaining tool call support
                if 'max_thinking_tokens' not in req_body:
                    req_body['max_thinking_tokens'] = 2048
                elif req_body.get('max_thinking_tokens', 0) > 2048:
                    print(f"⚠️  Capping max_thinking_tokens from {req_body['max_thinking_tokens']} to 2048 for fast responses")
                    req_body['max_thinking_tokens'] = 2048
                
                # Forward request to vLLM server
                resp = await http_client.post(
                    f"{vllm_url}/v1/chat/completions",
                    json=req_body,
                    timeout=300.0
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    # Count tool calls if present
                    tool_calls = result.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
                    tool_call_count = len(tool_calls) if tool_calls else 0
                    if tool_call_count > 0:
                        print(f"✅ vLLM server returned response with {tool_call_count} tool call(s)")
                    else:
                        print(f"✅ vLLM server returned text response")
                    return JSONResponse(result)
                
                # Normal error handling for non-200 responses
                # httpx.Response uses .text property (synchronous), not .atext() method
                # Note: resp.text is a property, not a method, so no await needed
                # For httpx.AsyncClient responses, the body is automatically read, so .text works directly
                try:
                    error_text = resp.text
                except Exception as text_error:
                    # Fallback: use content bytes and decode manually
                    try:
                        error_text = resp.content.decode('utf-8', errors='replace')
                    except Exception:
                        error_text = f"Failed to read error response (status {resp.status_code}): {text_error}"
                print(f"❌ vLLM server error: {resp.status_code} - {error_text[:500]}")
                error_response = ErrorResponse(error=error_text)
                return JSONResponse(
                    content=error_response.model_dump(),
                    status_code=resp.status_code
                )
            except httpx.TimeoutException:
                error_response = ErrorResponse(error="Request to vLLM server timed out")
                return JSONResponse(
                    content=error_response.model_dump(),
                    status_code=504
                )
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()
                print(f"❌ Error proxying to vLLM server: {error_msg}")
                print(f"   Traceback: {error_trace}")
                error_response = ErrorResponse(error=error_msg, traceback=error_trace)
                return JSONResponse(
                    content=error_response.model_dump(),
                    status_code=500
                )
        else:
            # Fallback to manual parsing (old implementation)
            # This code path is kept for safety/fallback
            error_response = ErrorResponse(
                error="Manual parsing mode is not implemented in this version. Please use automatic parser (USE_AUTOMATIC_PARSER=True)",
                traceback="Automatic parser provides better reliability and maintenance"
            )
            return JSONResponse(
                content=error_response.model_dump(),
                status_code=501
            )
    
    return app


# ------------------------------------------------------------------------------
# Helper function to download model to volume (run once)
# ------------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",  # Use A100 for download (or any available GPU - doesn't need to match server GPU)
    volumes={HF_CACHE_DIR: MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours for 32B model download (larger model requires more time)
)
def download_model_to_volume():
    """
    Download model weights to volume (run once to pre-populate cache).
    
    This function downloads the model weights to the Modal volume,
    which will persist across deployments for fast loading.
    
    Usage:
        modal run vllm_modal_deploy.py::download_model_to_volume
    
    Note: This is a one-time operation. After running, the model
    will be cached in the volume and subsequent deployments will
    load much faster.
    """
    from huggingface_hub import snapshot_download
    import os
    
    print(f"Downloading {MODEL_ID} to volume...")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print("This may take 15-30 minutes for a 32B model...")
    
    # Set HuggingFace environment variables
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_CACHE_DIR) / "hub")
    os.environ["HF_HUB_CACHE"] = str(Path(HF_CACHE_DIR) / "hub")
    
    # Download model (this will save to the mounted volume)
    print("Starting model download...")
    snapshot_download(
        MODEL_ID,
        cache_dir=Path(HF_CACHE_DIR) / "hub",
        local_files_only=False,
        force_download=True,  # Resume if interrupted
    )
    
    # Commit volume to persist the downloaded weights
    print("Committing model weights to volume...")
    MODEL_VOLUME.commit()
    
    # Check volume contents
    try:
        volume_files = MODEL_VOLUME.listdir("/", recursive=True)
        print(f"✓ Model downloaded and cached in volume")
        print(f"  Files cached: {len(volume_files)} files")
        
        # Estimate size from file entries
        total_size = 0
        for item in volume_files[:10]:  # Sample first 10 files
            if hasattr(item, 'size') and item.size:
                total_size += item.size
        if total_size > 0:
            print(f"  Estimated size: ~{total_size / (1024**3):.1f} GB (sampled)")
    except Exception as e:
        print(f"✓ Model downloaded and cached in volume")
        print(f"  (Could not list volume contents: {e})")

