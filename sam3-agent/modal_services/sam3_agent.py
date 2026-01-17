# modal_agent.py
#
# SAM3 Agent Microservice on Modal - LLM Provider Agnostic
#
# This microservice provides SAM3 agent functionality with complete LLM provider
# flexibility. All LLM configuration is passed in API requests - no hardcoded
# providers or required LLM secrets.
#
# HTTP API (after `modal deploy modal_agent.py`):
#
#   POST /sam3/segment
#   Body (JSON):
#   {
#     "prompt": "segment all ships",
#     "image_url": "https://example.com/image.jpg",   # or "image_b64": "..."
#     "llm_config": {                                 # Required: complete LLM config
#       "base_url": "https://api.example.com/v1",    # Any OpenAI-compatible API
#       "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",   # Any model name
#       "api_key": "",                                # API key (can be empty for some backends)
#       "name": "qwen3-vl",                           # Optional: for output files
#       "max_tokens": 4096                            # Optional: default 4096
#     },
#     "debug": true                                   # Optional: get visualization
#   }
#
#   Response (JSON):
#   {
#     "status": "success",
#     "summary": "...",
#     "regions": [...],
#     "debug_image_b64": "...",      # only if debug=true
#     "raw_sam3_json": {...},
#     "llm_config": {...}
#   }
#
# Supports any OpenAI-compatible LLM provider:
#   - Qwen3-VL (via vLLM)
#   - Anthropic (Claude)
#   - vLLM servers
#   - Custom OpenAI-compatible APIs

import os
import sys
import io
import base64
import json
import re
import tempfile
from functools import partial

import modal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal, Union

# Conditional import for pydantic - only needed in Modal container, not locally
try:
    from pydantic import BaseModel, Field, ValidationError, ConfigDict
except ImportError:
    # Stub classes for local parsing - Modal container will have real pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            pass
        model_config = None
    
    def Field(*args, **kwargs):
        return None
    
    class ValidationError(Exception):
        pass
    
    class ConfigDict(dict):
        pass

# ------------------------------------------------------------------------------
# Modal app + image
# ------------------------------------------------------------------------------

app = modal.App("sam3-agent-pyramidal-v2")

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SAM3_DIR = REPO_ROOT / "sam3"
LOCAL_ASSETS_DIR = REPO_ROOT / "assets"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "numpy<2.0",  # sam3 requires numpy<2.0
        "tqdm",
        "ftfy",
        "regex",
        "iopath",
        "typing_extensions",
        "huggingface_hub",
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "scikit-image",
        "openai",
        "pillow",
        "einops",
        "decord",
        "scikit-learn",
        "psutil",
        "pandas",
        "scipy",
        "fastapi",  # Required for fastapi_endpoint decorator
        "httpx",  # For downloading images from URLs (better headers, redirect handling)
        "pydantic",  # For VLM response validation
    )
    .env({"PYTHONPATH": "/root/sam3"})
    # use repo-relative paths so deploy works regardless of cwd
    .add_local_dir(str(LOCAL_SAM3_DIR), remote_path="/root/sam3/sam3")
    .add_local_dir(str(LOCAL_ASSETS_DIR), remote_path="/root/sam3/assets")
)

# ------------------------------------------------------------------------------
# VLM Interface for Qwen3-VL
# ------------------------------------------------------------------------------

class VLMInterface:
    """
    Interface for Qwen3-VL via vLLM API.
    
    Wraps vLLM API calls with image support for prompt refinement,
    detection verification, and rephrasing.
    """
    
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
    
    @staticmethod
    def encode_image_to_base64(image) -> str:
        """Encode PIL Image to base64 string"""
        from PIL import Image as PILImage
        buffered = io.BytesIO()
        if isinstance(image, PILImage.Image):
            # Convert to RGB if needed (JPEG doesn't support alpha)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
        else:
            # Assume it's already bytes
            buffered.write(image)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def query(self, image, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """
        Query the vLLM API with an image and text prompt.
        
        Args:
            image: PIL Image or image bytes
            prompt: Text prompt/question
            max_tokens: Maximum response length
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response or None on error
        """
        import httpx
        from PIL import Image
        
        # Handle image input
        if isinstance(image, bytes):
            image_base64 = base64.b64encode(image).decode('utf-8')
        else:
            image_base64 = self.encode_image_to_base64(image)
        
        # OpenAI-compatible message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(max_tokens, 8192),  # Server caps at 8192
            "temperature": temperature
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Validate response structure before accessing nested keys
                        if not isinstance(result, dict):
                            print(f"❌ vLLM API Error: Invalid response format (expected dict, got {type(result).__name__})")
                            return None
                        
                        choices = result.get("choices", [])
                        if not choices or not isinstance(choices, list) or len(choices) == 0:
                            print(f"❌ vLLM API Error: No choices in response")
                            print(f"Response: {response.text[:1000]}")
                            return None
                        
                        message = choices[0].get("message", {})
                        if not isinstance(message, dict):
                            print(f"❌ vLLM API Error: Invalid message format in response")
                            return None
                        
                        content = message.get("content")
                        if content is None:
                            print(f"❌ vLLM API Error: No content in message")
                            print(f"Response: {response.text[:1000]}")
                            return None
                        
                        return content
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"❌ vLLM API Error: Failed to parse response JSON: {e}")
                        print(f"Response: {response.text[:1000]}")
                        return None
                else:
                    # Enhanced error messages for different status codes
                    error_msg = f"❌ vLLM API Error: {response.status_code}"
                    if response.status_code == 404:
                        error_msg += " Response: " + response.text
                        print(error_msg)
                        print(f"   Endpoint: {self.endpoint}")
                        print(f"   Model: {self.model}")
                        print(f"   This may indicate the vLLM server is stopped or the endpoint is incorrect.")
                    else:
                        print(error_msg)
                        print(f"Response: {response.text[:1000]}")
                    return None
        
        except httpx.TimeoutException:
            print(f"❌ Request timeout after {self.timeout}s")
            print(f"   Endpoint: {self.endpoint}")
            return None
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Endpoint: {self.endpoint}")
            print(f"   This may indicate the vLLM server is unreachable or stopped.")
            return None
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP status error: {e}")
            print(f"   Endpoint: {self.endpoint}")
            if e.response is not None:
                print(f"   Response: {e.response.text[:1000]}")
            return None
        except httpx.RequestError as e:
            print(f"❌ Request error: {e}")
            print(f"   Endpoint: {self.endpoint}")
            return None
        except Exception as e:
            print(f"❌ API Exception: {e}")
            print(f"   Endpoint: {self.endpoint}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:500]}")
            return None


# ------------------------------------------------------------------------------
# Pydantic models for VLM response validation
# ------------------------------------------------------------------------------

class KeywordExtractionResponse(BaseModel):
    """Pydantic model for keyword extraction VLM responses"""
    keywords: List[str] = Field(..., min_length=1, max_length=4, description="List of 1-4 keywords for segmentation")
    primary_object: str = Field(..., description="Primary object type")
    reasoning: Optional[str] = Field(None, description="Optional reasoning for keyword selection")


class VerificationResponse(BaseModel):
    """Pydantic model for detection verification VLM responses"""
    is_valid: bool = Field(..., description="Whether the detection is valid")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = Field(None, description="Optional reasoning for validation")


# ------------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible Chat Completions API (Swagger docs)
# ------------------------------------------------------------------------------

class ImageUrl(BaseModel):
    """Image URL for multimodal messages"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "url": "https://example.com/image.jpg",
            "detail": "auto"
        }
    })
    
    url: str = Field(..., description="URL of the image or base64 data URI (data:image/jpeg;base64,...)")
    detail: Optional[str] = Field("auto", description="Image detail level: 'low', 'high', or 'auto'")


class ContentPartText(BaseModel):
    """Text content part for multimodal messages"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "text",
            "text": "What is in this image?"
        }
    })
    
    type: Literal["text"] = Field("text", description="Content type, must be 'text'")
    text: str = Field(..., description="Text content")


class ContentPartImage(BaseModel):
    """Image content part for multimodal messages"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg",
                "detail": "auto"
            }
        }
    })
    
    type: Literal["image_url"] = Field("image_url", description="Content type, must be 'image_url'")
    image_url: ImageUrl = Field(..., description="Image URL object")


class FunctionDefinition(BaseModel):
    """Function definition for tool calling"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    })
    
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(None, description="Function description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Function parameters (JSON Schema)")


class Tool(BaseModel):
    """Tool definition for function calling"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }
    })
    
    type: str = Field("function", description="Tool type, currently only 'function' is supported")
    function: FunctionDefinition = Field(..., description="Function definition")


class ToolCall(BaseModel):
    """Tool call made by the assistant"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            }
        }
    })
    
    id: str = Field(..., description="Unique identifier for the tool call")
    type: str = Field("function", description="Type of tool call")
    function: Dict[str, Any] = Field(..., description="Function call details with 'name' and 'arguments'")


class ChatMessage(BaseModel):
    """
    Chat message in conversation.
    
    Supports multiple roles:
    - system: System instructions
    - user: User messages (can include text and images)
    - assistant: Assistant responses (can include tool calls)
    - tool: Tool/function results
    """
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}
                    }
                ]
            }
        ]
    })
    
    role: str = Field(
        ..., 
        description="Message role: 'system', 'user', 'assistant', or 'tool'"
    )
    content: Optional[Any] = Field(
        None,
        description="Message content. String for text, or list of content parts for multimodal (text + images)"
    )
    name: Optional[str] = Field(None, description="Name of the author (for tool messages)")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID (required for 'tool' role messages)")


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    
    Use this to send messages to the LLM and receive completions.
    Supports text, multimodal (images), and tool/function calling.
    """
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for provider compatibility
        json_schema_extra={
            "example": {
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    )
    
    model: str = Field(
        ..., 
        description="Model identifier to use for completion"
    )
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of messages in the conversation",
        min_length=1
    )
    temperature: Optional[float] = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature (0-2). Higher = more random"
    )
    top_p: Optional[float] = Field(
        1.0, 
        ge=0.0, 
        le=1.0, 
        description="Nucleus sampling: only consider tokens with top_p probability mass"
    )
    n: Optional[int] = Field(
        1, 
        ge=1, 
        le=10, 
        description="Number of completions to generate"
    )
    stream: Optional[bool] = Field(
        False, 
        description="Whether to stream partial responses (SSE)"
    )
    stop: Optional[Any] = Field(
        None, 
        description="Stop sequences (string or list of strings)"
    )
    max_tokens: Optional[int] = Field(
        4096, 
        ge=1, 
        le=128000, 
        description="Maximum tokens to generate"
    )
    presence_penalty: Optional[float] = Field(
        0.0, 
        ge=-2.0, 
        le=2.0, 
        description="Presence penalty for new tokens (-2 to 2)"
    )
    frequency_penalty: Optional[float] = Field(
        0.0, 
        ge=-2.0, 
        le=2.0, 
        description="Frequency penalty for repeated tokens (-2 to 2)"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, 
        description="Token ID to bias value mapping"
    )
    user: Optional[str] = Field(
        None, 
        description="Unique user identifier for abuse monitoring"
    )
    tools: Optional[List[Tool]] = Field(
        None, 
        description="List of tools/functions available for the model to call"
    )
    tool_choice: Optional[Any] = Field(
        None, 
        description="Tool choice: 'none', 'auto', 'required', or specific tool"
    )
    response_format: Optional[Dict[str, Any]] = Field(
        None, 
        description="Response format: {'type': 'json_object'} for JSON mode"
    )
    seed: Optional[int] = Field(
        None, 
        description="Random seed for deterministic generation"
    )


class ChatChoice(BaseModel):
    """A single completion choice in the response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    })
    
    index: int = Field(..., description="Choice index (0-based)")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(
        None, 
        description="Reason generation stopped: 'stop', 'length', 'tool_calls', 'content_filter'"
    )
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities (if requested)")


class TokenUsage(BaseModel):
    """Token usage statistics for the completion"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt_tokens": 25,
            "completion_tokens": 10,
            "total_tokens": 35
        }
    })
    
    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int = Field(..., description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens (prompt + completion)")


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    
    Contains the generated message(s), token usage, and metadata.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
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
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35
            }
        }
    })
    
    id: str = Field(..., description="Unique completion identifier")
    object: str = Field("chat.completion", description="Object type (always 'chat.completion')")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatChoice] = Field(..., description="List of completion choices")
    usage: Optional[TokenUsage] = Field(None, description="Token usage statistics")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for reproducibility")


class ChatCompletionChunk(BaseModel):
    """
    Streaming chat completion chunk (SSE response).
    
    Sent when stream=true. Contains partial content.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "The"},
                    "finish_reason": None
                }
            ]
        }
    })
    
    id: str = Field(..., description="Completion identifier (same across chunks)")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="Delta choices with partial content")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")


class APIError(BaseModel):
    """API error response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "message": "Invalid API key provided",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
        }
    })
    
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(None, description="Parameter that caused the error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """
    OpenAI-compatible error response.
    
    Returned when an API request fails.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": {
                "message": "Invalid API key provided",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key"
            }
        }
    })
    
    error: APIError = Field(..., description="Error details")


# ------------------------------------------------------------------------------
# SAM3 API Request/Response Models (for Swagger docs)
# ------------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """LLM configuration for VLM-based processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "api_key": ""
        }
    })
    
    base_url: str = Field(..., description="Base URL of the LLM API (OpenAI-compatible)")
    model: str = Field(..., description="Model identifier")
    api_key: str = Field("", description="API key (can be empty for some backends)")
    name: Optional[str] = Field(None, description="Name for output files")
    max_tokens: Optional[int] = Field(4096, description="Maximum tokens for generation")


class PyramidalConfig(BaseModel):
    """Configuration for pyramidal batch processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5
        }
    })
    
    tile_size: Optional[int] = Field(512, description="Tile size for pyramidal processing")
    overlap_ratio: Optional[float] = Field(0.15, description="Overlap ratio between tiles")
    scales: Optional[List[float]] = Field([1.0, 0.5], description="Scale factors for multi-scale detection")
    batch_size: Optional[int] = Field(16, description="Batch size for inference")
    iou_threshold: Optional[float] = Field(0.5, description="IoU threshold for NMS")


class SAM3CountRequest(BaseModel):
    """Request model for SAM3 object counting"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "trees",
            "image_url": "https://example.com/aerial-image.jpg",
            "llm_config": {
                "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "api_key": ""
            },
            "confidence_threshold": 0.3,
            "max_retries": 2
        },
        "examples": [
            {
                "summary": "Using HTTP URL",
                "value": {
                    "prompt": "trees",
                    "image_url": "https://example.com/aerial-image.jpg",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "confidence_threshold": 0.3,
                    "max_retries": 2
                }
            },
            {
                "summary": "Using data URI (base64-encoded image)",
                "value": {
                    "prompt": "trees",
                    "image_url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "confidence_threshold": 0.3,
                    "max_retries": 2
                }
            }
        ]
    })
    
    prompt: str = Field(..., description="What objects to count (e.g., 'trees', 'cars', 'buildings')")
    image_url: Optional[str] = Field(None, description="Image URL (REQUIRED) - HTTP/HTTPS URL or data URI format: data:image/<type>;base64,<base64_string>. Supports image hosting services, but some may return 403 Forbidden errors.")
    llm_config: LLMConfig = Field(..., description="VLM configuration for prompt refinement")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_retries: Optional[int] = Field(2, ge=0, le=5, description="Maximum retry attempts for verification")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal processing configuration")
    include_obb: Optional[bool] = Field(False, description="Include oriented bounding boxes (OBB) in response")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon coordinates instead of parametric format [cx, cy, w, h, angle]")


class DetectionInfo(BaseModel):
    """Individual detection information"""
    id: int = Field(..., description="Detection ID")
    score: float = Field(..., description="Confidence score")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    pixel_area: Optional[int] = Field(None, description="Area in pixels")
    obb: Optional[List[float]] = Field(None, description="Oriented bounding box [cx, cy, w, h, angle] in degrees")
    obb_polygon: Optional[List[List[float]]] = Field(None, description="OBB as polygon coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")


class SAM3CountResponse(BaseModel):
    """Response model for SAM3 object counting"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "count": 47,
            "visual_prompt": "tree",
            "object_type": "tree",
            "detections": [
                {"id": 1, "score": 0.95, "box": [100, 200, 150, 250]}
            ]
        }
    })
    
    status: str = Field(..., description="Response status: 'success' or 'error'")
    count: Optional[int] = Field(None, description="Total count of detected objects")
    visual_prompt: Optional[str] = Field(None, description="Refined visual prompt used for detection")
    object_type: Optional[str] = Field(None, description="Detected object type")
    verification_info: Optional[Dict[str, Any]] = Field(None, description="Verification details")
    detections: Optional[List[Dict[str, Any]]] = Field(None, description="List of detections")
    pyramidal_stats: Optional[Dict[str, Any]] = Field(None, description="Pyramidal processing statistics")
    message: Optional[str] = Field(None, description="Error message if status is 'error'")
    traceback: Optional[str] = Field(None, description="Error traceback if status is 'error'")


class SAM3AreaRequest(BaseModel):
    """Request model for SAM3 area calculation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "solar panels",
            "image_url": "https://example.com/satellite-image.jpg",
            "llm_config": {
                "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "api_key": ""
            },
            "gsd": 0.5,
            "confidence_threshold": 0.3
        },
        "examples": [
            {
                "summary": "Using HTTP URL",
                "value": {
                    "prompt": "solar panels",
                    "image_url": "https://example.com/satellite-image.jpg",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "gsd": 0.5,
                    "confidence_threshold": 0.3
                }
            },
            {
                "summary": "Using data URI (base64-encoded image)",
                "value": {
                    "prompt": "solar panels",
                    "image_url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "gsd": 0.5,
                    "confidence_threshold": 0.3
                }
            }
        ]
    })
    
    prompt: str = Field(..., description="What objects to measure (e.g., 'solar panels', 'buildings')")
    image_url: Optional[str] = Field(None, description="Image URL (REQUIRED) - HTTP/HTTPS URL or data URI format: data:image/<type>;base64,<base64_string>. Supports image hosting services, but some may return 403 Forbidden errors.")
    llm_config: LLMConfig = Field(..., description="VLM configuration for prompt refinement")
    gsd: Optional[float] = Field(None, gt=0, description="Ground Sample Distance in meters/pixel")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_retries: Optional[int] = Field(2, ge=0, le=5, description="Maximum retry attempts for verification")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal processing configuration")
    include_obb: Optional[bool] = Field(False, description="Include oriented bounding boxes (OBB) in response")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon coordinates instead of parametric format [cx, cy, w, h, angle]")


class IndividualArea(BaseModel):
    """Individual object area information"""
    id: int = Field(..., description="Object ID")
    pixel_area: int = Field(..., description="Area in pixels")
    real_area_m2: Optional[float] = Field(None, description="Real area in square meters (if GSD provided)")
    score: float = Field(..., description="Confidence score")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    obb: Optional[List[float]] = Field(None, description="Oriented bounding box [cx, cy, w, h, angle] in degrees")
    obb_polygon: Optional[List[List[float]]] = Field(None, description="OBB as polygon coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")


class SAM3AreaResponse(BaseModel):
    """Response model for SAM3 area calculation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "object_count": 12,
            "total_pixel_area": 125000,
            "total_real_area_m2": 31250.0,
            "coverage_percentage": 12.5
        }
    })
    
    status: str = Field(..., description="Response status: 'success' or 'error'")
    object_count: Optional[int] = Field(None, description="Number of detected objects")
    total_pixel_area: Optional[int] = Field(None, description="Total area in pixels")
    total_real_area_m2: Optional[float] = Field(None, description="Total area in square meters")
    coverage_percentage: Optional[float] = Field(None, description="Percentage of image covered")
    individual_areas: Optional[List[Dict[str, Any]]] = Field(None, description="Individual object areas")
    visual_prompt: Optional[str] = Field(None, description="Refined visual prompt used")
    verification_info: Optional[Dict[str, Any]] = Field(None, description="Verification details")
    pyramidal_stats: Optional[Dict[str, Any]] = Field(None, description="Pyramidal processing statistics")
    message: Optional[str] = Field(None, description="Error message if status is 'error'")
    traceback: Optional[str] = Field(None, description="Error traceback if status is 'error'")


class SAM3SegmentRequest(BaseModel):
    """Request model for SAM3 segmentation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "segment all ships in the harbor",
            "image_url": "https://example.com/harbor-image.jpg",
            "llm_config": {
                "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "api_key": ""
            },
            "debug": True,
            "confidence_threshold": 0.3,
            "pyramidal_config": {
                "tile_size": 512,
                "overlap_ratio": 0.15,
                "scales": [1.0, 0.5],
                "batch_size": 16,
                "iou_threshold": 0.5
            }
        },
        "examples": [
            {
                "summary": "Using HTTP URL",
                "value": {
                    "prompt": "segment all ships in the harbor",
                    "image_url": "https://example.com/harbor-image.jpg",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "debug": True,
                    "confidence_threshold": 0.3,
                    "pyramidal_config": {
                        "tile_size": 512,
                        "overlap_ratio": 0.15,
                        "scales": [1.0, 0.5],
                        "batch_size": 16,
                        "iou_threshold": 0.5
                    }
                }
            },
            {
                "summary": "Using data URI (base64-encoded image)",
                "value": {
                    "prompt": "segment all ships in the harbor",
                    "image_url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "llm_config": {
                        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                        "api_key": ""
                    },
                    "debug": True,
                    "confidence_threshold": 0.3,
                    "pyramidal_config": {
                        "tile_size": 512,
                        "overlap_ratio": 0.15,
                        "scales": [1.0, 0.5],
                        "batch_size": 16,
                        "iou_threshold": 0.5
                    }
                }
            }
        ]
    })
    
    prompt: str = Field(..., description="Segmentation prompt (e.g., 'segment all ships')")
    image_url: Optional[str] = Field(None, description="Image URL (REQUIRED) - HTTP/HTTPS URL or data URI format: data:image/<type>;base64,<base64_string>. Supports image hosting services, but some may return 403 Forbidden errors.")
    llm_config: LLMConfig = Field(..., description="LLM/VLM configuration")
    debug: Optional[bool] = Field(False, description="Return debug visualization")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal processing configuration")
    include_obb: Optional[bool] = Field(False, description="Include oriented bounding boxes (OBB) in response")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon coordinates instead of parametric format [cx, cy, w, h, angle]")


class RegionInfo(BaseModel):
    """Segmented region information"""
    id: int = Field(..., description="Region ID")
    label: str = Field(..., description="Region label")
    score: float = Field(..., description="Confidence score")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    obb: Optional[List[float]] = Field(None, description="Oriented bounding box [cx, cy, w, h, angle] in degrees")
    obb_polygon: Optional[List[List[float]]] = Field(None, description="OBB as polygon coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")
    mask_rle: Optional[Dict[str, Any]] = Field(None, description="RLE-encoded mask")


class SAM3SegmentResponse(BaseModel):
    """Response model for SAM3 segmentation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "summary": "Detected 5 ships in the harbor",
            "regions": [
                {"id": 1, "label": "ship", "score": 0.95, "box": [100, 200, 300, 400]}
            ]
        }
    })
    
    status: str = Field(..., description="Response status: 'success' or 'error'")
    summary: Optional[str] = Field(None, description="Summary of segmentation results")
    regions: Optional[List[Dict[str, Any]]] = Field(None, description="Segmented regions")
    debug_image_b64: Optional[str] = Field(None, description="Debug visualization (if debug=true)")
    raw_sam3_json: Optional[Dict[str, Any]] = Field(None, description="Raw SAM3 output")
    llm_config: Optional[Dict[str, Any]] = Field(None, description="LLM config used")
    pyramidal_stats: Optional[Dict[str, Any]] = Field(None, description="Pyramidal processing statistics")
    message: Optional[str] = Field(None, description="Error message if status is 'error'")
    traceback: Optional[str] = Field(None, description="Error traceback if status is 'error'")


# ------------------------------------------------------------------------------
# LLM configuration helpers
# ------------------------------------------------------------------------------

def validate_llm_config(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize LLM config from request.
    
    Required fields:
    - base_url: API endpoint URL
    - model: Model name/identifier
    - api_key: API key (can be empty for some backends)
    
    Optional fields:
    - name: Name for output files (defaults to model name)
    - provider: Provider type (defaults to "openai-compatible")
    - max_tokens: Maximum tokens (defaults to 4096)
    """
    if not isinstance(llm_config, dict):
        raise ValueError("llm_config must be a dictionary")
    
    # Required fields
    if "base_url" not in llm_config:
        raise ValueError("llm_config must include 'base_url'")
    if "model" not in llm_config:
        raise ValueError("llm_config must include 'model'")
    
    # Set defaults
    normalized = {
        "base_url": str(llm_config["base_url"]),
        "model": str(llm_config["model"]),
        "api_key": str(llm_config.get("api_key", "")),
        "name": llm_config.get("name", llm_config["model"]),  # Default to model name
        "provider": llm_config.get("provider", "openai-compatible"),
        "max_tokens": int(llm_config.get("max_tokens", 4096)),
    }
    
    return normalized


# ------------------------------------------------------------------------------
# GPU-backed SAM3 model class
# ------------------------------------------------------------------------------

@app.cls(
    gpu="A100",
    timeout=600,
    image=image,
    scaledown_window=3600,  # Keep container alive for 1 hour after last request
    min_containers=1,  # Keep at least 1 container always running (always loaded)
    # Required secrets - SAM3 is a gated repository on HuggingFace:
    #   - "hf-token" containing key HF_TOKEN (REQUIRED - SAM3 model is gated)
    # To add secret: modal secret create hf-token HF_TOKEN=<your-token>
    # Note: LLM configuration is passed in API requests, no LLM secrets needed
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # REQUIRED - SAM3 is a gated repo
    ],
)
class SAM3Model:
    @modal.enter()
    def setup(self):
        """Runs once per container: load SAM3 model + processor into GPU."""
        # Disable PIL decompression bomb limit for large satellite/aerial imagery
        from PIL import Image as PILImageConfig
        PILImageConfig.MAX_IMAGE_PIXELS = None
        
        from huggingface_hub import login
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if "/root/sam3" not in sys.path:
            sys.path.append("/root/sam3")

        # HF_TOKEN is REQUIRED - SAM3 is a gated repository
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required but not set. "
                "SAM3 is a gated repository on HuggingFace. "
                "Please create the Modal secret: modal secret create hf-token HF_TOKEN=<your-token>"
            )
        
        # Authenticate with HuggingFace before attempting to load model
        try:
            login(token=hf_token)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            raise ValueError(
                f"Failed to authenticate with HuggingFace: {e}. "
                "Please verify your HF_TOKEN is valid and has access to facebook/sam3 repository."
            ) from e

        print("Loading SAM3 model...")
        bpe_path = "/root/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        # Use lower confidence threshold (0.25) to get more results, can be adjusted per request
        self.processor = Sam3Processor(self.model, confidence_threshold=0.3)
        print(f"SAM3 model loaded successfully with confidence threshold: {self.processor.confidence_threshold}")

    # --------------------------------------------------------------------------
    # VLM Helper Methods
    # --------------------------------------------------------------------------

    def _refine_prompt_with_vlm(self, image_bytes: bytes, user_query: str, vlm: VLMInterface) -> str:
        """
        Convert user query to SAM3-friendly visual prompt using VLM.
        
        Args:
            image_bytes: Raw image bytes
            user_query: Natural language query from user
            vlm: VLMInterface instance
            
        Returns:
            Visual prompt string (1-3 words) for SAM3 segmentation
        """
        from PIL import Image
        import io
        import re
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_type = image.mode  # Get image type (RGB, RGBA, etc.)
        
        REFINEMENT_TEMPLATE = """You are a visual descriptor for object segmentation on satellite images. You ONLY describe what objects look like, you NEVER count or answer questions. NEVER INCLUDE NUMBERS IN ANY FORM.

YOUR TASK:
Extract the OBJECT NAME from the user query and describe it for visual detection.
- Output: 1-3 words maximum
- Format: [object name] or [attribute + object name]
- Use singular form
- Focus on visual characteristics: color, shape, size

USER QUERY: "{query}"

OUTPUT ONLY the visual descriptor (1-3 words), nothing else. No explanation, no JSON, just the descriptor.

Examples:
- "How many planes on the tarmac?" → "white plane"
- "Count the storage tanks" → "circular tank"
- "How many buildings?" → "building"
- "Find all red cars" → "red car"

Now extract the visual descriptor:"""
        
        prompt = REFINEMENT_TEMPLATE.format(query=user_query)
        
        def _extract_meaningful_fallback(query: str) -> str:
            """Extract meaningful object name from query, avoiding generic words."""
            # Generic words to avoid
            generic_words = {'image', 'how', 'what', 'where', 'when', 'why', 'which', 'who', 
                           'many', 'much', 'count', 'find', 'show', 'detect', 'identify',
                           'are', 'is', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                           'of', 'with', 'from', 'by', 'this', 'that', 'these', 'those'}
            
            # Remove question words and common verbs
            query_lower = query.lower().strip()
            
            # Try to extract nouns (words that are capitalized or after common question patterns)
            # Pattern: "How many X" or "Count the X" or "Find all X"
            patterns = [
                r'(?:how many|count|find|detect|identify|show|locate)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+){0,2})',
                r'(?:what|which)\s+([a-z]+(?:\s+[a-z]+){0,2})',
                r'([a-z]+(?:\s+[a-z]+){0,2})\s+(?:in|on|at|from)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    candidate = match.group(1).strip()
                    words = candidate.split()
                    # Filter out generic words
                    meaningful_words = [w for w in words if w not in generic_words]
                    if meaningful_words:
                        return ' '.join(meaningful_words[:3])  # Max 3 words
            
            # Fallback: take all words except generic ones
            words = query_lower.split()
            meaningful_words = [w for w in words if w not in generic_words and len(w) > 2]
            if meaningful_words:
                return ' '.join(meaningful_words[:3])
            
            # Last resort: return "object"
            return "object"
        
        try:
            response = vlm.query(image, prompt, max_tokens=64, temperature=0.3)
            
            if response is None:
                print("⚠ VLM returned None for prompt refinement, using fallback")
                print(f"   Image Type: {image_type}")
                print(f"   Question: {user_query}")
                fallback = _extract_meaningful_fallback(user_query)
                print(f"   Using fallback prompt: '{fallback}'")
                return fallback
            
            # Extract the visual prompt (first line, strip whitespace)
            visual_prompt = response.strip().split('\n')[0].strip()
            # Remove quotes if present
            visual_prompt = visual_prompt.strip('"\'')
            
            # Validate that prompt is not generic
            generic_words = {'image', 'how', 'what', 'where', 'when', 'why', 'which', 'who'}
            if not visual_prompt or visual_prompt.lower() in generic_words:
                print("⚠ VLM response is generic or empty, using fallback")
                print(f"   Image Type: {image_type}")
                print(f"   Question: {user_query}")
                print(f"   VLM response: {response[:200]}")
                fallback = _extract_meaningful_fallback(user_query)
                print(f"   Using fallback prompt: '{fallback}'")
                return fallback
            
            print(f"✓ VLM refined prompt: '{user_query}' → '{visual_prompt}'")
            return visual_prompt
            
        except Exception as e:
            print(f"⚠ Error in prompt refinement: {e}")
            print(f"   Image Type: {image_type}")
            print(f"   Question: {user_query}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:500]}")
            fallback = _extract_meaningful_fallback(user_query)
            print(f"   Using fallback prompt: '{fallback}'")
            return fallback

    def _verify_detections_with_vlm(self, image, detections: List[Dict], 
                                     query: str, visual_prompt: str, vlm: VLMInterface) -> List[Dict]:
        """
        Verify each detection matches the query using VLM yes/no validation.
        
        Args:
            image: PIL Image of the full image
            detections: List of detection dicts with 'box', 'score', 'mask_rle'
            query: Original user query
            visual_prompt: Visual prompt used for segmentation
            vlm: VLMInterface instance
            
        Returns:
            List of verified detections (only those that pass VLM validation)
        """
        from PIL import Image
        
        VERIFICATION_TEMPLATE = """You are a visual verification assistant. 
Answer ONLY with 'yes' or 'no'. DEFAULT 'yes'

TASK:
1. Look only at the region inside the bounding box.
2. Check if the object CLEARLY matches the USER QUERY or VISUAL PROMPT.

USER QUERY: {query}
VISUAL PROMPT: {visual_prompt}

Answer 'yes' if:
- The object in the box matches the query/prompt
- It's a complete object (not partial/edge artifact)
- It has expected visual characteristics

Answer 'no' if:
- Wrong object type
- Partial/edge artifact
- Doesn't match visual characteristics

Answer ONLY: yes or no"""
        
        verified = []
        rejected = []
        
        for idx, det in enumerate(detections):
            try:
                box = det["box"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    rejected.append((idx, "Invalid bounding box"))
                    continue
                
                # Crop the detection region
                cropped = image.crop((x1, y1, x2, y2))
                
                prompt = VERIFICATION_TEMPLATE.format(
                    query=query,
                    visual_prompt=visual_prompt
                )
                
                response = vlm.query(cropped, prompt, max_tokens=16, temperature=0.1)
                
                if response is None:
                    # Default to yes if VLM fails (conservative)
                    print(f"  Detection {idx+1}: VLM verification failed (None response), accepting by default")
                    print(f"    Query: {query}")
                    print(f"    Visual prompt: {visual_prompt}")
                    verified.append(det)
                    continue
                
                # Extract yes/no from response
                response_lower = response.strip().lower()
                is_valid = "yes" in response_lower and "no" not in response_lower[:10]
                
                if is_valid:
                    verified.append(det)
                    print(f"  Detection {idx+1}: ✓ Verified by VLM")
                else:
                    rejected.append((idx, f"VLM rejected: {response[:50]}"))
                    print(f"  Detection {idx+1}: ✗ Rejected by VLM")
                    print(f"    VLM response: {response[:100]}")
                    
            except Exception as e:
                print(f"  Detection {idx+1}: Error in verification: {e}, accepting by default")
                print(f"    Query: {query}")
                print(f"    Visual prompt: {visual_prompt}")
                import traceback
                print(f"    Traceback: {traceback.format_exc()[:300]}")
                verified.append(det)  # Conservative: accept on error
                continue
        
        print(f"✓ VLM verification: {len(verified)}/{len(detections)} detections verified")
        return verified

    def _rephrase_prompt_with_vlm(self, image_bytes: bytes, original_query: str, 
                                   previous_prompt: str, vlm: VLMInterface, 
                                   attempt: int = 0, used_prompts: set = None) -> str:
        """
        Generate alternative prompt when no detections found.
        
        Args:
            image_bytes: Raw image bytes
            original_query: Original user query
            previous_prompt: Previous visual prompt that found nothing
            vlm: VLMInterface instance
            attempt: Current attempt number (for generating diverse fallbacks)
            used_prompts: Set of prompts already tried (to avoid duplicates)
            
        Returns:
            Alternative visual prompt (synonym/variation)
        """
        from PIL import Image
        import io
        import re
        
        if used_prompts is None:
            used_prompts = set()
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_type = image.mode  # Get image type (RGB, RGBA, etc.)
        
        REPHRASE_TEMPLATE = """You are a visual descriptor for object segmentation. Generate alternative keywords (max 2-3 words).

Original question: "{query}"
Previous visual prompt that found nothing: "{previous_prompt}"

USE CLOSELY RELATED SYNONYMS OF THE PREVIOUS PROMPT.
- Try true synonyms (same meaning)
- Try color variations
- Try singular/plural
- DO NOT generalize too much (avoid "object", "thing", "area")

Examples:
- "white plane" → "aircraft" or "airplane" or "gray plane"
- "circular tank" → "round container" or "cylindrical tank"
- "building" → "structure" or "house"

Output ONLY the alternative keyword (2-3 words max), nothing else:"""
        
        prompt = REPHRASE_TEMPLATE.format(
            query=original_query,
            previous_prompt=previous_prompt
        )
        
        def _generate_diverse_fallback(prev_prompt: str, orig_query: str, attempt_num: int) -> str:
            """Generate diverse fallback prompts to avoid infinite loops."""
            words = prev_prompt.split()
            
            # Strategy 1: Try variations based on attempt number
            if attempt_num == 0:
                # First fallback: use last word if multiple words
                if len(words) > 1:
                    return words[-1]
                else:
                    # Try to extract object from original query
                    query_lower = orig_query.lower()
                    # Look for patterns like "storage tanks" -> "tank"
                    patterns = [
                        r'(\w+)\s+tanks?',
                        r'(\w+)\s+buildings?',
                        r'(\w+)\s+planes?',
                        r'(\w+)\s+cars?',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, query_lower)
                        if match:
                            return match.group(1)
                    return prev_prompt + "s"  # Try plural
            elif attempt_num == 1:
                # Second fallback: try first word or add descriptor
                if len(words) > 1:
                    return words[0]
                else:
                    # Try to add a common descriptor
                    descriptors = ["large", "small", "circular", "rectangular"]
                    for desc in descriptors:
                        candidate = f"{desc} {prev_prompt}"
                        if candidate not in used_prompts:
                            return candidate
                    return prev_prompt
            else:
                # Third+ fallback: try removing modifiers or using base word
                if len(words) > 1:
                    # Remove first word (likely a modifier)
                    return ' '.join(words[1:])
                else:
                    # Last resort: return base word with variation
                    return prev_prompt
        
        try:
            response = vlm.query(image, prompt, max_tokens=64, temperature=0.5)
            
            if response is None:
                print("⚠ VLM returned None for rephrase, using fallback")
                print(f"   Image Type: {image_type}")
                print(f"   Question: {original_query}")
                print(f"   Previous prompt: {previous_prompt}")
                fallback = _generate_diverse_fallback(previous_prompt, original_query, attempt)
                # Ensure we don't return the same prompt
                if fallback.lower() == previous_prompt.lower() or fallback in used_prompts:
                    # Force a different variation
                    words = previous_prompt.split()
                    if len(words) > 1:
                        fallback = words[-1] if words[-1] != previous_prompt else words[0]
                    else:
                        fallback = previous_prompt + "s"
                used_prompts.add(fallback)
                print(f"   Using fallback prompt: '{fallback}'")
                return fallback
            
            # Extract the alternative prompt
            alternative = response.strip().split('\n')[0].strip()
            alternative = alternative.strip('"\'')
            
            # Validate that alternative is different from previous
            if not alternative or alternative.lower() == previous_prompt.lower():
                print("⚠ VLM rephrase same as previous, using variation")
                print(f"   Image Type: {image_type}")
                print(f"   Question: {original_query}")
                print(f"   Previous prompt: {previous_prompt}")
                print(f"   VLM response: {response[:200]}")
                alternative = _generate_diverse_fallback(previous_prompt, original_query, attempt)
                # Ensure uniqueness
                if alternative.lower() == previous_prompt.lower() or alternative in used_prompts:
                    words = previous_prompt.split()
                    if len(words) > 1:
                        alternative = words[-1] if words[-1] != previous_prompt else words[0]
                    else:
                        alternative = previous_prompt + "s"
            
            used_prompts.add(alternative)
            print(f"✓ VLM rephrased: '{previous_prompt}' → '{alternative}'")
            return alternative
            
        except Exception as e:
            print(f"⚠ Error in rephrase: {e}, using fallback")
            print(f"   Image Type: {image_type}")
            print(f"   Question: {original_query}")
            print(f"   Previous prompt: {previous_prompt}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:300]}")
            fallback = _generate_diverse_fallback(previous_prompt, original_query, attempt)
            # Ensure uniqueness
            if fallback.lower() == previous_prompt.lower() or fallback in used_prompts:
                words = previous_prompt.split()
                if len(words) > 1:
                    fallback = words[-1] if words[-1] != previous_prompt else words[0]
                else:
                    fallback = previous_prompt + "s"
            used_prompts.add(fallback)
            return fallback

    @staticmethod
    def _calculate_mask_area(mask_rle: Dict, gsd: float, image_shape: Tuple[int, int]) -> float:
        """
        Calculate area from RLE mask in square meters.
        
        Args:
            mask_rle: RLE-encoded mask dict with 'counts' and 'size'
            gsd: Ground Sample Distance (meters/pixel)
            image_shape: (height, width) of original image
            
        Returns:
            Area in square meters
        """
        import numpy as np
        import pycocotools.mask as mask_utils
        
        try:
            # Decode RLE to binary mask using pycocotools
            # mask_rle should be a dict with 'counts' and 'size' keys
            # Handle stringified counts (from JSON serialization fix)
            mask_rle_for_decode = mask_rle
            if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                counts = mask_rle['counts']
                if isinstance(counts, str):
                    mask_rle_for_decode = mask_rle.copy()
                    mask_rle_for_decode['counts'] = counts.encode('utf-8')
            mask_binary = mask_utils.decode(mask_rle_for_decode)
            
            # Handle mask dimensions
            if mask_binary.ndim > 2:
                mask_binary = mask_binary.squeeze()
            
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[:, :, 0] if mask_binary.shape[2] == 1 else mask_binary.max(axis=2)
            
            # Resize mask if needed to match image shape
            if mask_binary.shape[:2] != image_shape:
                from scipy.ndimage import zoom
                zoom_factors = (
                    image_shape[0] / mask_binary.shape[0],
                    image_shape[1] / mask_binary.shape[1]
                )
                mask_binary = zoom(mask_binary.astype(float), zoom_factors, order=0) > 0.5
            
            # Count pixels and calculate area
            pixel_count = np.sum(mask_binary > 0)
            area_m2 = pixel_count * (gsd ** 2)
            
            return area_m2
            
        except Exception as e:
            print(f"⚠ Error calculating mask area: {e}")
            return 0.0

    @staticmethod
    def _generate_obb_from_mask_rle(
        mask_rle: Dict,
        image_shape: Tuple[int, int],
        min_contour_area: int = 10,
        min_obb_size: float = 5.0
    ) -> Optional[List[float]]:
        """
        Generate OBB from RLE-encoded mask.
        
        Args:
            mask_rle: RLE-encoded mask dict with 'counts' and 'size'
            image_shape: (height, width) of original image
            min_contour_area: Minimum contour area in pixels
            min_obb_size: Minimum OBB width/height in pixels
            
        Returns:
            OBB as [cx, cy, w, h, angle] or None if invalid
            Format: [center_x, center_y, width, height, angle_in_degrees]
            Angle is normalized to 0-180 degrees
        """
        import numpy as np
        import cv2
        import pycocotools.mask as mask_utils
        
        try:
            # Decode RLE to binary mask
            mask_rle_for_decode = mask_rle
            if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                counts = mask_rle['counts']
                if isinstance(counts, str):
                    mask_rle_for_decode = mask_rle.copy()
                    mask_rle_for_decode['counts'] = counts.encode('utf-8')
            
            mask_binary = mask_utils.decode(mask_rle_for_decode)
            
            # Handle mask dimensions
            if mask_binary.ndim > 2:
                mask_binary = mask_binary.squeeze()
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[:, :, 0] if mask_binary.shape[2] == 1 else mask_binary.max(axis=2)
            
            # Resize mask if needed to match image shape
            if mask_binary.shape[:2] != image_shape:
                from scipy.ndimage import zoom
                zoom_factors = (
                    image_shape[0] / mask_binary.shape[0],
                    image_shape[1] / mask_binary.shape[1]
                )
                mask_binary = zoom(mask_binary.astype(float), zoom_factors, order=0) > 0.5
            
            # Binarize
            mask_binary = (mask_binary > 0.5).astype(np.uint8)
            
            if mask_binary.sum() == 0:
                return None
            
            # Find contours
            cnts, _ = cv2.findContours(
                mask_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(cnts) == 0:
                return None
            
            # Get largest contour
            cnt = max(cnts, key=cv2.contourArea)
            
            if cv2.contourArea(cnt) < min_contour_area:
                return None
            
            # Compute OBB
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # Normalize angle (OpenCV returns angle where width may be < height)
            if w < h:
                angle = angle + 90
                w, h = h, w
            
            angle = angle % 180
            
            # Validate OBB size
            if w < min_obb_size or h < min_obb_size:
                return None
            
            return [float(cx), float(cy), float(w), float(h), float(angle)]
            
        except Exception as e:
            print(f"⚠ Error generating OBB from mask: {e}")
            return None

    @staticmethod
    def _normalize_obb(obb: List[float], image_width: int, image_height: int) -> Optional[List[float]]:
        """
        Normalize OBB parametric representation with respect to image dimensions.
        
        Args:
            obb: [cx, cy, w, h, angle] in pixel coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Normalized OBB as [cx/width, cy/height, w/width, h/height, angle] or None if invalid
        """
        if obb is None or len(obb) != 5:
            return None
        
        if image_width <= 0 or image_height <= 0:
            return None
        
        try:
            cx, cy, w, h, angle = obb
            return [
                float(cx) / image_width,
                float(cy) / image_height,
                float(w) / image_width,
                float(h) / image_height,
                float(angle)  # Angle remains unchanged
            ]
        except Exception as e:
            print(f"⚠ Error normalizing OBB: {e}")
            return None

    @staticmethod
    def _obb_to_polygon(obb: List[float], image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[List[List[float]]]:
        """
        Convert OBB parametric representation to polygon coordinates.
        
        Args:
            obb: [cx, cy, w, h, angle] in degrees (normalized or pixel coordinates)
            image_width: Optional image width for normalization (if None, returns pixel coordinates)
            image_height: Optional image height for normalization (if None, returns pixel coordinates)
            
        Returns:
            Polygon as list of 4 corner points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            Coordinates are normalized to [0, 1] if image_width and image_height are provided,
            otherwise in pixel coordinates.
            Returns None if obb is None or invalid
        """
        import numpy as np
        
        if obb is None or len(obb) != 5:
            return None
        
        try:
            cx, cy, w, h, angle = obb
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Calculate half dimensions
            half_w = w / 2.0
            half_h = h / 2.0
            
            # Create box points in local coordinate system (centered at origin)
            corners = np.array([
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h]
            ])
            
            # Rotation matrix
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            # Rotate corners
            rotated_corners = corners @ rotation_matrix.T
            
            # Translate to center position
            polygon = rotated_corners + np.array([cx, cy])
            
            # Normalize polygon coordinates if image dimensions provided
            if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
                polygon[:, 0] = polygon[:, 0] / image_width
                polygon[:, 1] = polygon[:, 1] / image_height
            
            # Convert to list of lists
            return polygon.tolist()
            
        except Exception as e:
            print(f"⚠ Error converting OBB to polygon: {e}")
            return None

    @modal.method()
    def sam3_infer_only(
        self,
        image_bytes: bytes,
        text_prompt: str,
        confidence_threshold: float = None,
        pyramidal_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        SAM3 inference using pyramidal batch processing.
        Returns the same format as original sam3_inference function.
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt for segmentation
            confidence_threshold: Optional confidence threshold (0.0-1.0)
            pyramidal_config: Optional pyramidal configuration dict
        """
        # Validate confidence threshold before proceeding
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }
        
        # Save original threshold and restore in finally block
        original_threshold = self.processor.confidence_threshold
        try:
            # Set confidence threshold if provided
            if confidence_threshold is not None:
                self.processor.confidence_threshold = confidence_threshold
                print(f"✓ Using confidence threshold: {confidence_threshold}")
            
            # Set default pyramidal config
            config = {
                "tile_size": 512,
                "overlap_ratio": 0.15,
                "scales": [1.0, 0.5],
                "batch_size": 16,
                "iou_threshold": 0.5,
            }
            if pyramidal_config:
                config.update(pyramidal_config)
            
            # Run pyramidal inference
            result = self._sam3_pyramidal_infer_impl(
                image_bytes=image_bytes,
                text_prompt=text_prompt,
                tile_size=config["tile_size"],
                overlap_ratio=config["overlap_ratio"],
                scales=config["scales"],
                iou_threshold=config["iou_threshold"],
                confidence_threshold=self.processor.confidence_threshold,
                batch_size=config["batch_size"],
            )
            
            if result["status"] != "success":
                return result
            
            # Convert to expected output format (normalized boxes in xyxy)
            orig_w = result["orig_img_w"]
            orig_h = result["orig_img_h"]
            detections = result["detections"]
            
            pred_boxes = []
            pred_masks = []
            pred_scores = []
            
            for det in detections:
                box = det["box"]  # [x1, y1, x2, y2] in pixels
                x1, y1, x2, y2 = box
                # Normalize to [0, 1] keeping [x1, y1, x2, y2] format
                pred_boxes.append([x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h])
                pred_masks.append(det["mask_rle"])
                pred_scores.append(det["score"])
            
            return {
                "status": "success",
                "orig_img_h": orig_h,
                "orig_img_w": orig_w,
                "pred_boxes": pred_boxes,
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
                "pyramidal_stats": result.get("pyramidal_stats", {}),
            }
        finally:
            # Always restore original threshold
            self.processor.confidence_threshold = original_threshold

    # --------------------------------------------------------------------------
    # Pyramidal Tiling Helper Methods
    # --------------------------------------------------------------------------

    def _create_tiles(self, image, tile_size: int, overlap_ratio: float):
        """
        Generate overlapping tiles from PIL image.
        
        Args:
            image: PIL Image
            tile_size: Size of each tile (e.g., 512)
            overlap_ratio: Overlap between tiles (e.g., 0.15)
            
        Returns:
            List of (tile_image, (offset_x, offset_y))
        """
        img_width, img_height = image.size
        stride = int(tile_size * (1 - overlap_ratio))
        tiles = []
        
        # If image is smaller than tile size, return as single tile
        if img_width <= tile_size and img_height <= tile_size:
            return [(image, (0, 0))]
        
        for y in range(0, img_height, stride):
            for x in range(0, img_width, stride):
                x_end = min(x + tile_size, img_width)
                y_end = min(y + tile_size, img_height)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                
                tile = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((tile, (x_start, y_start)))
                
                if x_end >= img_width:
                    break
            if y_end >= img_height:
                break
        
        return tiles

    def _transform_box_to_original(self, box, tile_offset, scale: float, orig_size):
        """
        Transform box from tile coordinates to original image space.
        
        Args:
            box: Box coordinates [x1, y1, x2, y2] in tile space
            tile_offset: (offset_x, offset_y) of tile in scaled image
            scale: Scale factor used for this pyramid level
            orig_size: (width, height) of original image
            
        Returns:
            Transformed box in original image coordinates
        """
        import numpy as np
        import torch
        
        offset_x, offset_y = tile_offset
        orig_w, orig_h = orig_size
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        
        box = np.array(box).copy()
        
        # Add offset (tile position in scaled image)
        box[0] += offset_x
        box[1] += offset_y
        box[2] += offset_x
        box[3] += offset_y
        
        # Scale back to original resolution
        box = box / scale
        
        # Clip to image bounds
        box[0] = max(0, min(box[0], orig_w))
        box[1] = max(0, min(box[1], orig_h))
        box[2] = max(0, min(box[2], orig_w))
        box[3] = max(0, min(box[3], orig_h))
        
        return box

    def _transform_mask_to_original(self, mask_binary, tile_offset, scale: float, orig_size, tile_size):
        """
        Transform binary mask from tile coordinates to original image space.
        
        Args:
            mask_binary: Binary mask numpy array at tile resolution [H, W]
            tile_offset: (offset_x, offset_y) of tile in scaled image
            scale: Scale factor used for this pyramid level
            orig_size: (width, height) of original image
            tile_size: (width, height) of tile
            
        Returns:
            Binary mask numpy array at original image resolution [orig_h, orig_w]
        """
        import numpy as np
        from scipy import ndimage
        
        offset_x, offset_y = tile_offset
        orig_w, orig_h = orig_size
        tile_w, tile_h = tile_size
        
        # Create full-size mask at original resolution
        global_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # If scale != 1.0, we need to resize the mask to account for scaling
        if scale != 1.0:
            # Resize mask to original scale
            # The tile covers a region that was scaled down, so we need to scale it back up
            scale_factor = 1.0 / scale
            resized_h = int(tile_h * scale_factor)
            resized_w = int(tile_w * scale_factor)
            
            # Use scipy zoom for resizing (faster than PIL for binary masks)
            zoom_factors = (resized_h / mask_binary.shape[0], resized_w / mask_binary.shape[1])
            mask_resized = ndimage.zoom(mask_binary.astype(float), zoom_factors, order=0) > 0.5
            mask_resized = mask_resized.astype(np.uint8)
        else:
            mask_resized = mask_binary.astype(np.uint8)
            resized_h, resized_w = tile_h, tile_w
        
        # Calculate position in original image
        # The offset is in scaled image coordinates, so scale it back
        orig_offset_x = int(offset_x / scale)
        orig_offset_y = int(offset_y / scale)
        
        # Calculate placement bounds (clip to image bounds)
        y_start = max(0, orig_offset_y)
        y_end = min(orig_h, orig_offset_y + resized_h)
        x_start = max(0, orig_offset_x)
        x_end = min(orig_w, orig_offset_x + resized_w)
        
        # Calculate source bounds (for the mask)
        src_y_start = max(0, -orig_offset_y)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -orig_offset_x)
        src_x_end = src_x_start + (x_end - x_start)
        
        # Ensure we don't exceed mask bounds
        if src_y_end > mask_resized.shape[0]:
            src_y_end = mask_resized.shape[0]
            y_end = y_start + (src_y_end - src_y_start)
        if src_x_end > mask_resized.shape[1]:
            src_x_end = mask_resized.shape[1]
            x_end = x_start + (src_x_end - src_x_start)
        
        # Place the mask
        if y_end > y_start and x_end > x_start:
            global_mask[y_start:y_end, x_start:x_end] = mask_resized[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return global_mask

    def _calculate_iou(self, box1, box2) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _calculate_mask_iou(self, mask_rle1: Dict, mask_rle2: Dict) -> float:
        """
        Calculate Intersection over Union between two RLE-encoded masks.
        More accurate than box-based IoU for irregular shapes.
        
        Args:
            mask_rle1, mask_rle2: RLE-encoded mask dicts with 'counts' and 'size'
            
        Returns:
            IoU value (0-1), or -1 on error (caller should fallback to box IoU)
        """
        import numpy as np
        import pycocotools.mask as mask_utils
        
        try:
            # Prepare RLE for decoding (handle string counts from JSON serialization)
            def prepare_rle(rle):
                if isinstance(rle, dict) and 'counts' in rle:
                    rle_copy = rle.copy()
                    if isinstance(rle_copy['counts'], str):
                        rle_copy['counts'] = rle_copy['counts'].encode('utf-8')
                    return rle_copy
                return rle
            
            rle1 = prepare_rle(mask_rle1)
            rle2 = prepare_rle(mask_rle2)
            
            # Check size compatibility
            if rle1.get('size') != rle2.get('size'):
                return -1  # Signal caller to fallback to box IoU
            
            # Use pycocotools for efficient IoU calculation
            iou = mask_utils.iou([rle1], [rle2], [False])[0][0]
            return float(iou)
            
        except Exception as e:
            # Log error and signal caller to fallback
            print(f"⚠ Mask IoU calculation failed: {e}, falling back to box IoU")
            return -1

    def _apply_nms(self, detections, iou_threshold: float, use_mask_iou: bool = True):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        Prefers higher scores, then finer scales.
        
        Args:
            detections: List of detection dicts with 'box', 'score', 'scale', 'mask_rle'
            iou_threshold: IoU threshold for suppression
            use_mask_iou: If True, use mask-based IoU (more accurate for irregular shapes).
                          Falls back to box-based IoU if masks unavailable or on error.
                          Default: True for better accuracy.
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by score (descending), then by scale (ascending = finer first)
        detections = sorted(
            detections, 
            key=lambda d: (-d['score'], d.get('scale', 1.0))
        )
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Filter remaining detections based on IoU
            remaining = []
            for d in detections:
                iou = -1  # Default: will trigger box fallback
                
                # Try mask-based IoU if enabled and masks available
                if use_mask_iou and 'mask_rle' in current and 'mask_rle' in d:
                    iou = self._calculate_mask_iou(current['mask_rle'], d['mask_rle'])
                
                # Fallback to box-based IoU if mask IoU failed or not available
                if iou < 0:
                    iou = self._calculate_iou(current['box'], d['box'])
                
                # Keep detection if IoU is below threshold
                if iou < iou_threshold:
                    remaining.append(d)
            
            detections = remaining
        
        return keep

    def _extract_tile_backbone(self, backbone_out, tile_idx: int, batch_size: int):
        """
        Extract single tile's backbone features from batched output.
        
        Args:
            backbone_out: Batched backbone output dictionary
            tile_idx: Index of the tile to extract
            batch_size: Total batch size
            
        Returns:
            Dictionary with single-tile backbone features
        """
        import torch
        
        extracted = {}
        
        for key, value in backbone_out.items():
            if isinstance(value, torch.Tensor):
                # Extract single tile from batch dimension
                if value.shape[0] == batch_size:
                    extracted[key] = value[tile_idx:tile_idx+1]
                else:
                    extracted[key] = value
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                extracted[key] = self._extract_tile_backbone(value, tile_idx, batch_size)
            elif isinstance(value, list):
                # Check if list contains tensors that need batch extraction
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    # List of tensors (e.g., backbone_fpn, vision_pos_enc)
                    # Extract single tile from each tensor's batch dimension
                    extracted[key] = [
                        t[tile_idx:tile_idx+1] if t.shape[0] == batch_size else t
                        for t in value
                    ]
                elif len(value) == batch_size:
                    # List of items per batch (original logic)
                    extracted[key] = [value[tile_idx]]
                else:
                    extracted[key] = value
            else:
                extracted[key] = value
        
        return extracted

    def _sam3_pyramidal_infer_impl(
        self,
        image_bytes: bytes,
        text_prompt: str,
        tile_size: int = 512,
        overlap_ratio: float = 0.15,
        scales: Optional[List[float]] = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Internal implementation for pyramidal batch inference with text encoding cache.
        
        Optimizations:
        1. Text encoded ONCE via self.processor.model.backbone.forward_text()
        2. Images batch-encoded via self.processor.set_image_batch()
        3. Cached text features injected into each tile state
        4. GPU-accelerated NMS via torchvision.ops.batched_nms
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt for segmentation
            tile_size: Size of each tile (default: 512)
            overlap_ratio: Overlap between tiles (default: 0.15)
            scales: List of scales for pyramid (default: [1.0, 0.5])
            iou_threshold: IoU threshold for NMS (default: 0.5)
            confidence_threshold: Minimum confidence threshold (default: 0.3)
            batch_size: Batch size for processing tiles (default: 16)
            
        Returns:
            Dict with detections, count, and processing stats
        """
        from PIL import Image
        import io
        import torch
        import numpy as np
        from sam3.train.masks_ops import rle_encode
        
        # Set defaults
        scales = sorted(scales or [1.0, 0.5], reverse=True)
        
        # Validate scales to prevent division by zero
        if any(s <= 0 for s in scales):
            return {"status": "error", "message": "scales must contain positive values only (> 0)"}
        
        # Set processor confidence threshold
        # NOTE: We set a LOWER threshold in processor to get more candidates,
        # then filter by confidence_threshold in the detection loop.
        # This avoids double filtering (processor + loop) being too aggressive.
        original_threshold = self.processor.confidence_threshold
        # Use half the threshold in processor to get more candidates for NMS
        processor_threshold = max(0.1, confidence_threshold * 0.5)
        self.processor.confidence_threshold = processor_threshold
        print(f"✓ Confidence thresholds: processor={processor_threshold:.2f}, final={confidence_threshold:.2f}")
        
        # Use try-finally to ensure threshold is always restored
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            orig_w, orig_h = image.size
            print(f"✓ Image loaded: {orig_w}x{orig_h}")
            
            # ================================================================
            # OPTIMIZATION 1: Encode text ONCE
            # ================================================================
            text_outputs = self.processor.model.backbone.forward_text(
                [text_prompt],
                device=self.processor.device
            )
            print(f"✓ Text encoded once (cached for all tiles)")
            
            all_detections = []
            total_tiles = 0
            stats = {
                "scales": scales,
                "tile_size": tile_size,
                "overlap_ratio": overlap_ratio,
                "tiles_per_scale": {},
            }
            
            for scale in scales:
                # Scale image
                if scale != 1.0:
                    scaled_w = int(orig_w * scale)
                    scaled_h = int(orig_h * scale)
                    scaled_image = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                else:
                    scaled_image = image
                
                # Generate tiles for this scale
                tiles_with_offsets = self._create_tiles(scaled_image, tile_size, overlap_ratio)
                stats["tiles_per_scale"][str(scale)] = len(tiles_with_offsets)
                total_tiles += len(tiles_with_offsets)
                
                print(f"  Scale {scale}: {len(tiles_with_offsets)} tiles")
                
                # Process tiles in batches
                for batch_start in range(0, len(tiles_with_offsets), batch_size):
                    batch_end = min(batch_start + batch_size, len(tiles_with_offsets))
                    batch = tiles_with_offsets[batch_start:batch_end]
                    
                    tile_images = [t[0] for t in batch]
                    tile_offsets = [t[1] for t in batch]
                    
                    # ================================================================
                    # OPTIMIZATION 2: Batch encode images
                    # ================================================================
                    batch_state = self.processor.set_image_batch(tile_images)
                    actual_batch_size = len(tile_images)
                    
                    for i in range(len(tile_images)):
                        try:
                            # Extract per-tile backbone
                            extracted_backbone = self._extract_tile_backbone(
                                batch_state['backbone_out'], i, actual_batch_size
                            )
                            
                            # Debug: Verify backbone_fpn extraction
                            if 'backbone_fpn' in extracted_backbone:
                                fpn_list = extracted_backbone['backbone_fpn']
                                if isinstance(fpn_list, list) and len(fpn_list) > 0:
                                    first_fpn_shape = fpn_list[0].shape if hasattr(fpn_list[0], 'shape') else 'N/A'
                                    print(f"  ✓ Tile {i}: backbone_fpn extracted, first level shape: {first_fpn_shape}")
                            
                            tile_state = {
                                'original_height': batch_state['original_heights'][i],
                                'original_width': batch_state['original_widths'][i],
                                'backbone_out': extracted_backbone
                            }
                            
                            # ================================================================
                            # OPTIMIZATION 1 (cont): Inject cached text features
                            # ================================================================
                            tile_state['backbone_out'].update({
                                'language_features': text_outputs['language_features'],
                                'language_mask': text_outputs['language_mask'],
                                'language_embeds': text_outputs['language_embeds'],
                            })
                            
                            # Initialize geometric prompt
                            if 'geometric_prompt' not in tile_state:
                                tile_state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                            
                            # Run grounding (text already embedded)
                            tile_state = self.processor._forward_grounding(tile_state)
                            
                            # Collect detections
                            if 'boxes' in tile_state and len(tile_state['boxes']) > 0:
                                boxes = tile_state['boxes'].cpu().numpy()
                                masks = tile_state['masks'].cpu()
                                scores = tile_state['scores'].cpu().numpy()
                                
                                # Get tile dimensions for mask transformation
                                tile_h = tile_state['original_height']
                                tile_w = tile_state['original_width']
                                
                                for j in range(len(boxes)):
                                    if scores[j] >= confidence_threshold:
                                        # Transform box to original coords
                                        orig_box = self._transform_box_to_original(
                                            boxes[j], tile_offsets[i], scale, (orig_w, orig_h)
                                        )
                                        
                                        # Validate box is not degenerate after clipping
                                        if orig_box[2] <= orig_box[0] or orig_box[3] <= orig_box[1]:
                                            continue  # Skip degenerate boxes
                                        
                                        # Transform mask to original coords (CRITICAL: mask must match box coordinates)
                                        mask_binary_tile = masks[j].squeeze().numpy() > 0.5
                                        mask_binary_global = self._transform_mask_to_original(
                                            mask_binary_tile, 
                                            tile_offsets[i], 
                                            scale, 
                                            (orig_w, orig_h),
                                            (tile_w, tile_h)
                                        )
                                        
                                        # Encode mask as RLE at GLOBAL resolution
                                        mask_rle = rle_encode(torch.tensor(mask_binary_global, dtype=torch.bool).unsqueeze(0))[0]
                                        
                                        # CRITICAL FIX: Decode bytes to string for JSON serialization
                                        # pycocotools returns 'counts' as bytes which can't be serialized to JSON
                                        if "counts" in mask_rle and isinstance(mask_rle["counts"], bytes):
                                            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                                        
                                        # Validate mask-to-box alignment: RLE size must match original image
                                        if mask_rle.get("size") != [orig_h, orig_w]:
                                            print(f"⚠ Mask size mismatch: expected [{orig_h}, {orig_w}], got {mask_rle.get('size')}")
                                            continue  # Skip this detection
                                        
                                        # Calculate box area in original image pixels
                                        box_area_pixels = int((orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1]))
                                        
                                        all_detections.append({
                                            'box': orig_box.tolist(),
                                            'mask_rle': mask_rle,
                                            'score': float(scores[j]),
                                            'scale': scale,
                                            'box_area_pixels': box_area_pixels,
                                        })
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"⚠ Error processing tile {i} in batch at scale {scale}:")
                            print(f"   Tile offset: {tile_offsets[i]}")
                            print(f"   Error: {e}")
                            print(f"   Traceback: {error_details[:500]}")
                            stats.setdefault("tile_errors", []).append({
                                "tile_idx": i,
                                "scale": scale,
                                "offset": list(tile_offsets[i]),
                                "error": str(e),
                            })
                            continue
                    
                    # Free batch memory
                    del batch_state
                    if batch_start % (batch_size * 2) == 0:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            stats["total_tiles"] = total_tiles
            stats["successful_tiles"] = total_tiles - len(stats.get("tile_errors", []))
            print(f"✓ Processed {total_tiles} tiles across {len(scales)} scales")
            print(f"  Successful tiles: {stats['successful_tiles']}/{total_tiles}")
            if stats.get("tile_errors"):
                print(f"  ⚠ Failed tiles: {len(stats['tile_errors'])}")
            print(f"  Raw detections: {len(all_detections)}")
            
            # Apply NMS
            final_detections = self._apply_nms(all_detections, iou_threshold)
            print(f"✓ After NMS: {len(final_detections)} detections")
            
            # Provide diagnostic warnings for edge cases
            if stats["successful_tiles"] == 0:
                print(f"❌ WARNING: All {total_tiles} tiles failed to process!")
                print(f"   This may indicate a model loading issue or incompatible image format.")
            elif len(final_detections) == 0 and len(all_detections) > 0:
                print(f"⚠ WARNING: {len(all_detections)} raw detections reduced to 0 after NMS.")
                print(f"   Consider lowering iou_threshold (current: {iou_threshold})")
            elif len(final_detections) == 0:
                print(f"ℹ No objects detected for prompt '{text_prompt}'.")
                print(f"   Consider lowering confidence_threshold (current: {confidence_threshold})")
            
            return {
                "status": "success",
                "count": len(final_detections),
                "detections": final_detections,
                "orig_img_w": orig_w,
                "orig_img_h": orig_h,
                "pyramidal_stats": stats,
            }
        finally:
            # Always restore original confidence threshold, even if an exception occurs
            self.processor.confidence_threshold = original_threshold

    @modal.method()
    def sam3_pyramidal_infer(
        self,
        image_bytes: bytes,
        text_prompt: str,
        tile_size: int = 512,
        overlap_ratio: float = 0.15,
        scales: Optional[List[float]] = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Modal wrapper around the local pyramidal inference implementation."""
        return self._sam3_pyramidal_infer_impl(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            scales=scales,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
        )

    @modal.method()
    def sam3_count(
        self,
        image_bytes: bytes,
        text_prompt: str,
        llm_config: Dict[str, Any],
        confidence_threshold: float = 0.3,
        pyramidal_config: Dict[str, Any] = None,
        max_retries: int = 2,
        include_obb: bool = False,
        obb_as_polygon: bool = False,
    ) -> Dict[str, Any]:
        """
        Count objects using VLM-enhanced pyramidal SAM3 segmentation.
        
        Pipeline: VLM Prompt Refinement → SAM3 Segmentation → VLM Verification → Retry if needed
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt describing objects to count
            llm_config: LLM configuration dict (required):
                - base_url: API endpoint URL
                - model: Model name
                - api_key: API key (can be empty)
            confidence_threshold: Minimum confidence threshold (default: 0.3)
            pyramidal_config: Optional pyramidal configuration:
                - tile_size: Size of each tile (default: 512)
                - overlap_ratio: Overlap between tiles (default: 0.15)
                - scales: List of scales (default: [1.0, 0.5])
                - batch_size: Batch size (default: 16)
                - iou_threshold: NMS IoU threshold (default: 0.5)
            max_retries: Maximum retry attempts with rephrased prompts (default: 2)
            include_obb: Whether to include oriented bounding boxes in response (default: False)
            obb_as_polygon: If True and include_obb=True, return OBB as polygon coordinates instead of parametric format (default: False)
        
        Returns:
            Dict with count, visual_prompt, verification_info, detections, and processing stats
        """
        from PIL import Image
        import io
        
        # Validate LLM config
        try:
            llm_config = validate_llm_config(llm_config)
        except Exception as e:
            return {"status": "error", "message": f"Invalid llm_config: {str(e)}"}
        
        # Initialize VLM interface
        vlm = VLMInterface(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            timeout=120
        )
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        # Load image for VLM operations
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Track attempts for retry logic
        attempts = []
        visual_prompt = None
        verified_detections = []
        used_prompts = set()  # Track used prompts to avoid infinite loops
        
        # Step 1: Refine prompt with VLM
        print("Step 1: VLM Prompt Refinement")
        visual_prompt = self._refine_prompt_with_vlm(image_bytes, text_prompt, vlm)
        if visual_prompt:
            used_prompts.add(visual_prompt.lower())
        
        # Retry loop
        for attempt in range(max_retries + 1):
            print(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")
            print(f"Using visual prompt: '{visual_prompt}'")
            
            # Step 2: Run SAM3 segmentation
            result = self._sam3_pyramidal_infer_impl(
                image_bytes=image_bytes,
                text_prompt=visual_prompt,
                tile_size=config["tile_size"],
                overlap_ratio=config["overlap_ratio"],
                scales=config["scales"],
                iou_threshold=config["iou_threshold"],
                confidence_threshold=confidence_threshold,
                batch_size=config["batch_size"],
            )
            
            if result["status"] != "success":
                attempts.append({
                    "attempt": attempt + 1,
                    "visual_prompt": visual_prompt,
                    "status": "error",
                    "message": result.get("message", "Unknown error")
                })
                if attempt < max_retries:
                    # Try rephrasing
                    visual_prompt = self._rephrase_prompt_with_vlm(
                        image_bytes, text_prompt, visual_prompt, vlm, 
                        attempt=attempt, used_prompts=used_prompts
                    )
                    continue
                else:
                    return result
            
            detections = result["detections"]
            print(f"Found {len(detections)} raw detections")
            
            # Step 3: Verify detections with VLM
            if len(detections) > 0:
                print("Step 3: VLM Detection Verification")
                verified_detections = self._verify_detections_with_vlm(
                    image, detections, text_prompt, visual_prompt, vlm
                )
                
                attempts.append({
                    "attempt": attempt + 1,
                    "visual_prompt": visual_prompt,
                    "raw_count": len(detections),
                    "verified_count": len(verified_detections),
                    "rejected_count": len(detections) - len(verified_detections)
                })
                
                # If we have verified detections, we're done
                if len(verified_detections) > 0:
                    break
            else:
                attempts.append({
                    "attempt": attempt + 1,
                    "visual_prompt": visual_prompt,
                    "raw_count": 0,
                    "verified_count": 0
                })
            
            # If no detections and we have retries left, try rephrasing
            if len(verified_detections) == 0 and attempt < max_retries:
                print(f"No detections found, rephrasing prompt...")
                visual_prompt = self._rephrase_prompt_with_vlm(
                    image_bytes, text_prompt, visual_prompt, vlm,
                    attempt=attempt, used_prompts=used_prompts
                )
        
        # Prepare response
        verification_info = {
            "attempts": attempts,
            "verified_count": len(verified_detections),
            "rejected_count": sum(a.get("rejected_count", 0) for a in attempts),
            "total_attempts": len(attempts)
        }
        
        # Extract object type from prompt
        object_type = text_prompt.strip().lower()
        words = object_type.split()
        if words:
            object_type = words[-1].rstrip('s')
        
        # Generate OBBs if requested
        if include_obb and len(verified_detections) > 0:
            print("Generating OBBs from masks...")
            orig_h = result.get("orig_img_h", 0)
            orig_w = result.get("orig_img_w", 0)
            image_shape = (orig_h, orig_w)
            
            for det in verified_detections:
                if 'mask_rle' in det:
                    obb = self._generate_obb_from_mask_rle(det['mask_rle'], image_shape)
                    # Normalize OBB and polygon if requested
                    if obb_as_polygon and obb is not None:
                        # Normalize OBB parametric format
                        normalized_obb = self._normalize_obb(obb, orig_w, orig_h)
                        det['obb'] = normalized_obb
                        # Convert to normalized polygon
                        obb_polygon = self._obb_to_polygon(obb, orig_w, orig_h)
                        det['obb_polygon'] = obb_polygon
                    else:
                        det['obb'] = obb
        
        return {
            "status": "success",
            "count": len(verified_detections),
            "visual_prompt": visual_prompt,
            "object_type": object_type,
            "verification_info": verification_info,
            "detections": verified_detections,
            "pyramidal_stats": result.get("pyramidal_stats", {}),
            "orig_img_w": result.get("orig_img_w", 0),
            "orig_img_h": result.get("orig_img_h", 0),
        }

    @modal.method()
    def sam3_area(
        self,
        image_bytes: bytes,
        text_prompt: str,
        llm_config: Dict[str, Any],
        gsd: float = None,
        confidence_threshold: float = 0.3,
        pyramidal_config: Dict[str, Any] = None,
        max_retries: int = 2,
        include_obb: bool = False,
        obb_as_polygon: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate areas of detected objects using VLM-enhanced Pyramidal SAM3 segmentation.
        
        Pipeline: VLM Keyword Extraction → SAM3 Segmentation → VLM Validation → Mask-Based Area Calculation
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt describing objects to measure
            llm_config: LLM configuration dict (required):
                - base_url: API endpoint URL
                - model: Model name
                - api_key: API key (can be empty)
            gsd: Ground Sample Distance in meters/pixel (required for real-world area)
            confidence_threshold: Minimum confidence threshold (default: 0.3)
            pyramidal_config: Optional pyramidal configuration:
                - tile_size: Size of each tile (default: 512)
                - overlap_ratio: Overlap between tiles (default: 0.15)
                - scales: List of scales (default: [1.0, 0.5])
                - batch_size: Batch size (default: 16)
                - iou_threshold: NMS IoU threshold (default: 0.5)
            max_retries: Maximum retry attempts with rephrased prompts (default: 2)
            include_obb: Whether to include oriented bounding boxes in response (default: False)
            obb_as_polygon: If True and include_obb=True, return OBB as polygon coordinates instead of parametric format (default: False)
        
        Returns:
            Dict with object count, total area (mask-based), coverage percentage, and per-object areas
        """
        from PIL import Image
        import io
        
        # Validate LLM config
        try:
            llm_config = validate_llm_config(llm_config)
        except Exception as e:
            return {"status": "error", "message": f"Invalid llm_config: {str(e)}"}
        
        # Initialize VLM interface
        vlm = VLMInterface(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            timeout=120
        )
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        # Load image for VLM operations
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = image.size
        image_shape = (orig_h, orig_w)
        total_image_pixels = orig_w * orig_h
        
        # Step 1: Extract keywords with VLM (for area, we use multi-keyword approach)
        print("Step 1: VLM Keyword Extraction")
        keywords = self._extract_keywords_with_vlm(image_bytes, text_prompt, vlm)
        
        # Track attempts
        attempts = []
        all_verified_detections = []
        last_result = None
        used_prompts = set()  # Track used prompts across all keywords to avoid infinite loops
        
        # Step 2: Segment with each keyword and combine
        print("\nStep 2: Multi-keyword Segmentation")
        for keyword_idx, keyword in enumerate(keywords):
            print(f"\n--- Keyword {keyword_idx + 1}/{len(keywords)}: '{keyword}' ---")
            
            visual_prompt = keyword
            if visual_prompt:
                used_prompts.add(visual_prompt.lower())
            verified_detections = []
            
            # Retry loop for this keyword
            for attempt in range(max_retries + 1):
                print(f"  Attempt {attempt + 1}/{max_retries + 1} with prompt: '{visual_prompt}'")
                
                # Run SAM3 segmentation
                result = self._sam3_pyramidal_infer_impl(
                    image_bytes=image_bytes,
                    text_prompt=visual_prompt,
                    tile_size=config["tile_size"],
                    overlap_ratio=config["overlap_ratio"],
                    scales=config["scales"],
                    iou_threshold=config["iou_threshold"],
                    confidence_threshold=confidence_threshold,
                    batch_size=config["batch_size"],
                )
                
                last_result = result  # Keep last result for stats
                
                if result["status"] != "success":
                    if attempt < max_retries:
                        visual_prompt = self._rephrase_prompt_with_vlm(
                            image_bytes, text_prompt, visual_prompt, vlm,
                            attempt=attempt, used_prompts=used_prompts
                        )
                        continue
                    else:
                        break
                
                detections = result["detections"]
                print(f"  Found {len(detections)} raw detections")
                
                # Step 3: Verify detections with VLM
                if len(detections) > 0:
                    print("  Step 3: VLM Mask Validation")
                    verified_detections = self._verify_detections_with_vlm(
                        image, detections, text_prompt, visual_prompt, vlm
                    )
                    
                    if len(verified_detections) > 0:
                        # Deduplicate against existing detections
                        for new_det in verified_detections:
                            is_duplicate = False
                            for existing_det in all_verified_detections:
                                iou = self._calculate_iou(new_det["box"], existing_det["box"])
                                if iou > 0.5:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                all_verified_detections.append(new_det)
                        break
                
                # Try rephrasing if no detections
                if len(verified_detections) == 0 and attempt < max_retries:
                    visual_prompt = self._rephrase_prompt_with_vlm(
                        image_bytes, text_prompt, visual_prompt, vlm,
                        attempt=attempt, used_prompts=used_prompts
                    )
        
        print(f"\nTotal verified detections: {len(all_verified_detections)}")
        
        # Step 4: Calculate mask-based areas
        print("Step 4: Mask-Based Area Calculation")
        if gsd is None or gsd <= 0:
            return {
                "status": "error",
                "message": "gsd is required for area calculation. Provide gsd in meters/pixel."
            }
        
        individual_areas = []
        total_pixel_area = 0
        
        for idx, det in enumerate(all_verified_detections):
            try:
                # Calculate area from mask (mask-based, more accurate)
                mask_rle = det.get("mask_rle")
                if mask_rle:
                    area_m2 = self._calculate_mask_area(mask_rle, gsd, image_shape)
                    pixel_area = int(area_m2 / (gsd ** 2))  # Convert back to pixels for reporting
                else:
                    # Fallback to box area if mask not available
                    box = det["box"]
                    pixel_area = int((box[2] - box[0]) * (box[3] - box[1]))
                    area_m2 = pixel_area * (gsd ** 2)
                
                if area_m2 <= 0:
                    continue
                
                total_pixel_area += pixel_area
                
                area_info = {
                    "id": idx + 1,
                    "pixel_area": pixel_area,
                    "real_area_m2": round(area_m2, 4),
                    "score": det["score"],
                    "box": det["box"],
                }
                
                # Generate OBB if requested
                if include_obb and mask_rle:
                    obb = self._generate_obb_from_mask_rle(mask_rle, image_shape)
                    # Normalize OBB and polygon if requested
                    if obb_as_polygon and obb is not None:
                        # Normalize OBB parametric format
                        normalized_obb = self._normalize_obb(obb, orig_w, orig_h)
                        area_info["obb"] = normalized_obb
                        # Convert to normalized polygon
                        obb_polygon = self._obb_to_polygon(obb, orig_w, orig_h)
                        area_info["obb_polygon"] = obb_polygon
                    else:
                        area_info["obb"] = obb
                
                individual_areas.append(area_info)
                print(f"  Object {idx+1}: {area_m2:.2f} m² ({pixel_area} pixels)")
                
            except Exception as e:
                print(f"⚠ Error calculating area for detection {idx}: {e}")
                continue
        
        # Calculate coverage percentage
        coverage_percentage = (total_pixel_area / total_image_pixels * 100) if total_image_pixels > 0 else 0
        total_real_area_m2 = total_pixel_area * (gsd ** 2)
        
        # Prepare response
        response = {
            "status": "success",
            "object_count": len(individual_areas),
            "total_pixel_area": total_pixel_area,
            "total_real_area_m2": round(total_real_area_m2, 4),
            "coverage_percentage": round(coverage_percentage, 4),
            "individual_areas": individual_areas,
            "visual_prompt": ", ".join(keywords),
            "verification_info": {
                "keywords": keywords,
                "verified_count": len(all_verified_detections)
            },
            "orig_img_w": orig_w,
            "orig_img_h": orig_h,
            "gsd": gsd,
            "pyramidal_stats": last_result.get("pyramidal_stats", {}) if last_result else {},
        }
        
        return response

    def _extract_keywords_with_vlm(self, image_bytes: bytes, user_query: str, vlm: VLMInterface) -> List[str]:
        """
        Extract multiple keywords from user query using VLM (for area calculation).
        
        Args:
            image_bytes: Raw image bytes
            user_query: Natural language query
            vlm: VLMInterface instance
            
        Returns:
            List of 2-4 keywords for segmentation
        """
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        KEYWORD_EXTRACTION_TEMPLATE = """Analyze this satellite/aerial image and the user's query to extract segmentation keywords.

User Query: "{query}"

Your task:
1. Identify the PRIMARY object(s) the user wants to measure area for
2. Generate 2-4 SHORT keywords/phrases (2-4 words each) that describe these objects visually
3. Focus on: color, shape, texture, context (e.g., "white circular tanks", "rectangular buildings")
4. DO NOT include numbers or counts
5. DO NOT answer the query, just extract visual descriptors

Output format (JSON):
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "primary_object": "main object type",
  "reasoning": "brief explanation"
}}

Now analyze the image and query."""
        
        prompt = KEYWORD_EXTRACTION_TEMPLATE.format(query=user_query)
        
        try:
            response = vlm.query(image, prompt, max_tokens=256, temperature=0.7)
            
            if response is None:
                print("⚠ VLM returned None for keyword extraction, using fallback")
                print(f"   Original query: {user_query}")
                return [user_query]
            
            # Try to extract JSON from response with improved patterns
            # First, try to remove markdown code blocks if present
            cleaned_response = response.strip()
            # Remove markdown code blocks
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
            
            # Try multiple JSON extraction patterns
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
                r'\{.*?"keywords".*?\}',  # JSON with keywords field
                r'\{[^}]*\}',  # Simple JSON object
            ]
            
            result_dict = None
            for pattern in json_patterns:
                json_match = re.search(pattern, cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        result_dict = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if result_dict:
                try:
                    # Validate with Pydantic
                    validated_result = KeywordExtractionResponse(**result_dict)
                    keywords = validated_result.keywords
                    primary_object = validated_result.primary_object
                    reasoning = validated_result.reasoning
                    
                    # Validate that keywords are meaningful (not empty, not generic)
                    if keywords and isinstance(keywords, list) and len(keywords) > 0:
                        # Filter out empty or generic keywords
                        meaningful_keywords = [k for k in keywords if k and isinstance(k, str) and len(k.strip()) > 0]
                        if meaningful_keywords:
                            print(f"\n=== Keyword Extraction ===")
                            print(f"Primary Object: {primary_object}")
                            print(f"Keywords: {meaningful_keywords}")
                            if reasoning:
                                print(f"Reasoning: {reasoning}")
                            print(f"===========================\n")
                            return meaningful_keywords
                    
                    print(f"⚠ No meaningful keywords in validated response, using fallback")
                    print(f"   Extracted keywords: {keywords}")
                    print(f"   Original query: {user_query}")
                    return [user_query]
                        
                except ValidationError as ve:
                    print(f"❌ VLM response validation failed: {ve}")
                    print(f"   Full VLM response: {response[:1000]}")
                    print(f"   Original query: {user_query}")
                    print(f"   Using fallback: {user_query}")
                    return [user_query]
            else:
                print(f"❌ Could not find or parse JSON in VLM response")
                print(f"   Full VLM response: {response[:1000]}")
                print(f"   Original query: {user_query}")
                print(f"   Using fallback: {user_query}")
                return [user_query]
                
        except Exception as e:
            print(f"❌ Error in keyword extraction: {e}")
            print(f"   Original query: {user_query}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:500]}")
            return [user_query]

    def call_sam_service_pyramidal(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str = "sam3_output",
        pyramidal_config: Dict[str, Any] = None,
    ) -> str:
        """
        Pyramidal version of call_sam_service - drop-in replacement.
        
        Uses batch pyramidal SAM3 instead of raw SAM3, then formats output
        to match the expected format for agent_inference.
        """
        import os as _os
        import json as _json
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        print(f"📞 Loading image '{image_path}' for pyramidal segmentation with prompt '{text_prompt}'...")
        
        # Load image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Run pyramidal inference
        result = self._sam3_pyramidal_infer_impl(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=config["tile_size"],
            overlap_ratio=config["overlap_ratio"],
            scales=config["scales"],
            iou_threshold=config["iou_threshold"],
            confidence_threshold=self.processor.confidence_threshold,
            batch_size=config["batch_size"],
        )
        
        if result["status"] != "success":
            # Return empty results on error
            outputs = {
                "original_image_path": image_path,
                "output_image_path": "",
                "orig_img_h": 0,
                "orig_img_w": 0,
                "pred_boxes": [],
                "pred_masks": [],
                "pred_scores": [],
            }
        else:
            orig_w = result["orig_img_w"]
            orig_h = result["orig_img_h"]
            detections = result["detections"]
            
            # Convert detections to expected format (normalized boxes in xyxy)
            pred_boxes = []
            pred_masks = []
            pred_scores = []
            
            for det in detections:
                box = det["box"]  # [x1, y1, x2, y2] in pixels
                x1, y1, x2, y2 = box
                # Normalize to [0, 1] keeping [x1, y1, x2, y2] format
                pred_boxes.append([x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h])
                
                # Mask is already in RLE format
                pred_masks.append(det["mask_rle"])
                pred_scores.append(det["score"])
            
            outputs = {
                "original_image_path": image_path,
                "output_image_path": "",  # set after path construction
                "orig_img_h": orig_h,
                "orig_img_w": orig_w,
                "pred_boxes": pred_boxes,
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
                "pyramidal_stats": result.get("pyramidal_stats", {}),
            }
        
        # Save to JSON (same as original call_sam_service)
        import hashlib
        
        image_basename = _os.path.basename(image_path)
        image_basename_no_ext = _os.path.splitext(image_basename)[0]
        safe_dir_name = image_basename_no_ext.replace("/", "_").replace("\\", "_")
        
        # Fallback for empty directory name (e.g., if filename starts with dot)
        if not safe_dir_name or safe_dir_name.startswith("-"):
            safe_dir_name = "output" if not safe_dir_name else "image_" + safe_dir_name
        
        # Truncate/hash long prompts to avoid "Filename too long" OS error (255 byte limit)
        prompt_hash = hashlib.md5(text_prompt.encode()).hexdigest()[:8]
        safe_prompt_slug = text_prompt.replace("/", "_").replace("\\", "_")
        if len(safe_prompt_slug) > 50:
            safe_prompt_slug = safe_prompt_slug[:50] + f"_{prompt_hash}"
        
        output_dir = _os.path.join(output_folder_path, safe_dir_name, safe_prompt_slug)
        _os.makedirs(output_dir, exist_ok=True)
        
        json_path = _os.path.join(output_dir, "sam3_output.json")
        output_image_path = _os.path.join(output_dir, "sam3_output.png")
        outputs["output_image_path"] = output_image_path
        
        # Save outputs JSON
        with open(json_path, "w") as f:
            _json.dump(outputs, f, indent=2)
        print(f"✓ Pyramidal SAM3 found {len(outputs['pred_boxes'])} objects")
        
        # Render visualization; fall back to copying the raw image if rendering fails
        try:
            from sam3.agent.viz import visualize
            viz_img = visualize(outputs)
            viz_img.save(output_image_path)
        except Exception as e:
            print(f"⚠ Warning: Failed to render pyramidal visualization: {e}")
            try:
                import shutil as _shutil
                _shutil.copy(image_path, output_image_path)
            except Exception as copy_e:
                print(f"⚠ Warning: Failed to copy raw image as fallback: {copy_e}")
        
        return json_path

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        prompt: str,
        llm_config: Dict[str, Any],
        debug: bool = False,
        confidence_threshold: float = None,
        pyramidal_config: Dict[str, Any] = None,
        include_obb: bool = False,
        obb_as_polygon: bool = False,
    ) -> Dict[str, Any]:
        """
        Core GPU inference method - LLM provider agnostic.
        
        This method accepts any LLM configuration and uses it to call the LLM API.
        No hardcoded provider assumptions - works with any OpenAI-compatible API.
        
        Args:
            image_bytes: raw bytes of the input image
            prompt: natural language query for segmentation
            llm_config: Complete LLM configuration dict (provider-agnostic):
                - base_url: API endpoint URL (required) - any OpenAI-compatible endpoint
                - model: Model name/identifier (required) - any model name
                - api_key: API key (required, can be empty string for backends without auth)
                - name: Name for output files (optional, defaults to model name)
                - max_tokens: Maximum tokens (optional, defaults to 4096)
            debug: whether to return a visualization image (base64)
            confidence_threshold: Optional confidence threshold (0.0-1.0). If None, uses processor's default (0.3)
            pyramidal_config: Optional pyramidal processing configuration dict:
                - tile_size: Tile size for pyramidal processing (default: 512)
                - overlap_ratio: Overlap ratio between tiles (default: 0.15)
                - scales: Scale factors for multi-scale detection (default: [1.0, 0.5])
                - batch_size: Batch size for inference (default: 16)
                - iou_threshold: IoU threshold for NMS deduplication (default: 0.5)
            include_obb: Whether to include oriented bounding boxes in response (default: False)
            obb_as_polygon: If True and include_obb=True, return OBB as polygon coordinates instead of parametric format (default: False)
        
        Returns:
            Dict with status, regions, summary, and optional debug visualization
        """
        # Ensure io module is available locally (avoids scoping issues)
        import io as _io
        
        try:
            from sam3.agent.client_llm import (
                send_generate_request as send_generate_request_orig,
            )
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import SAM3 agent modules: {e}",
            }

        # Validate and normalize LLM config
        try:
            llm_config = validate_llm_config(llm_config)
        except Exception as e:
            return {"status": "error", "message": f"Invalid llm_config: {str(e)}"}

        # Validate confidence threshold before proceeding
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }

        # Save original threshold and restore in finally block
        original_threshold = self.processor.confidence_threshold
        try:
            # Set confidence threshold if provided
            if confidence_threshold is not None:
                self.processor.confidence_threshold = confidence_threshold
                print(f"✓ Using confidence threshold: {confidence_threshold}")

            # Cap max_tokens to 4096 to allow longer reasoning outputs for 32B Thinking model
            requested_max_tokens = llm_config["max_tokens"]
            safe_max_tokens = min(requested_max_tokens, 4096)
            
            send_generate_request = partial(
                send_generate_request_orig,
                server_url=llm_config["base_url"],
                model=llm_config["model"],
                api_key=llm_config["api_key"],
                max_tokens=safe_max_tokens,
            )

            # Use pyramidal batch SAM3 instead of raw SAM3
            # Use provided config or sensible defaults
            effective_pyramidal_config = pyramidal_config or {
                "tile_size": 512,
                "overlap_ratio": 0.15,
                "scales": [1.0, 0.5],
                "batch_size": 16,
                "iou_threshold": 0.5,
            }
            call_sam_service = partial(
                self.call_sam_service_pyramidal,
                pyramidal_config=effective_pyramidal_config,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert image to JPEG for reliable reading by cv2/PIL
                from PIL import Image as PILImage
                
                try:
                    # Open image and convert to RGB if needed
                    pil_img = PILImage.open(_io.BytesIO(image_bytes))
                    
                    # Convert to RGB if needed (handles RGBA, P, LA modes)
                    if pil_img.mode in ('RGBA', 'LA', 'P'):
                        pil_img = pil_img.convert('RGB')
                    elif pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    # Save once as JPEG for consistency
                    image_path = os.path.join(temp_dir, "input_image.jpg")
                    pil_img.save(image_path, format="JPEG", quality=95)
                    
                except Exception as img_error:
                    print(f"⚠️ Image format detection failed: {img_error}, trying raw save")
                    image_path = os.path.join(temp_dir, "input_image.jpg")
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                try:
                    # Call agent_inference directly to get data from return values
                    from sam3.agent.agent_core import agent_inference
                    
                    agent_history, final_output_dict, rendered_final_output = agent_inference(
                        image_path,
                        prompt,
                        send_generate_request=send_generate_request,
                        call_sam_service=call_sam_service,
                        debug=debug,
                        output_dir=output_dir,
                        max_generations=10,  # Limit LLM API calls to 10
                    )

                    if not final_output_dict:
                        return {
                            "status": "error",
                            "message": "No output generated by SAM3.",
                        }

                    # Debug visualization image - convert PIL Image to base64
                    debug_image_b64 = None
                    if debug and rendered_final_output:
                        buffer = _io.BytesIO()
                        rendered_final_output.save(buffer, format="PNG")
                        debug_image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

                    # Extract data from final_output_dict (which contains the SAM3 results)
                    # The final_output_dict should have pred_boxes, pred_masks, pred_scores, etc.
                    raw_json = final_output_dict.copy()
                    
                    # Remove file paths that aren't needed in response
                    raw_json.pop("original_image_path", None)
                    raw_json.pop("output_image_path", None)
                    raw_json.pop("text_prompt", None)
                    raw_json.pop("image_path", None)

                    # Helper to make objects JSON-serializable (handles bytes from RLE masks)
                    def make_json_serializable(obj):
                        """Recursively convert bytes and other non-serializable types to JSON-safe values."""
                        if isinstance(obj, bytes):
                            return obj.decode("utf-8")
                        if isinstance(obj, dict):
                            return {k: make_json_serializable(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [make_json_serializable(i) for i in obj]
                        if isinstance(obj, (int, float, str, bool, type(None))):
                            return obj
                        # Handle numpy types
                        if hasattr(obj, 'item'):  # numpy scalar
                            return obj.item()
                        if hasattr(obj, 'tolist'):  # numpy array
                            return obj.tolist()
                        return str(obj)  # Fallback to string representation

                    # Try to normalize "regions" field if present, otherwise construct from pred_boxes/pred_masks
                    regions = (
                        raw_json.get("regions")
                        or raw_json.get("objects")
                        or raw_json.get("instances")
                        or []
                    )
                    
                    # If no regions found, construct from pred_boxes and pred_masks
                    if not regions and "pred_boxes" in raw_json and "pred_masks" in raw_json:
                        pred_boxes = raw_json.get("pred_boxes", [])
                        pred_masks = raw_json.get("pred_masks", [])
                        pred_scores = raw_json.get("pred_scores", [])
                        regions = [
                            {
                                "bbox": box,
                                "mask": mask,
                                "score": pred_scores[i] if i < len(pred_scores) else None,
                            }
                            for i, (box, mask) in enumerate(zip(pred_boxes, pred_masks))
                        ]

                    # Generate OBBs if requested
                    if include_obb and len(regions) > 0:
                        print("Generating OBBs from masks...")
                        orig_h = raw_json.get("orig_img_h", 0)
                        orig_w = raw_json.get("orig_img_w", 0)
                        image_shape = (orig_h, orig_w)
                        
                        for region in regions:
                            # Extract mask_rle from region (may be in "mask" or "mask_rle" field)
                            mask_rle = region.get("mask") or region.get("mask_rle")
                            if mask_rle:
                                obb = self._generate_obb_from_mask_rle(mask_rle, image_shape)
                                # Normalize OBB and polygon if requested
                                if obb_as_polygon and obb is not None:
                                    # Normalize OBB parametric format
                                    normalized_obb = self._normalize_obb(obb, orig_w, orig_h)
                                    region["obb"] = normalized_obb
                                    # Convert to normalized polygon
                                    obb_polygon = self._obb_to_polygon(obb, orig_w, orig_h)
                                    region["obb_polygon"] = obb_polygon
                                else:
                                    region["obb"] = obb

                    # Ensure all data is JSON-serializable before returning
                    raw_json = make_json_serializable(raw_json)
                    regions = make_json_serializable(regions)

                    summary = (
                        f"SAM3 returned {len(regions)} regions for prompt: {prompt}"
                    )

                    return {
                        "status": "success",
                        "summary": summary,
                        "regions": regions,
                        "debug_image_b64": debug_image_b64,
                        "raw_sam3_json": raw_json,
                        # "agent_history": agent_history,  # Include agent history for debugging
                        # "llm_config": {
                        #     "name": llm_config["name"],
                        #     "model": llm_config["model"],
                        #     "base_url": llm_config["base_url"],
                        # },
                    }

                except Exception as e:
                    import traceback

                    return {
                        "status": "error",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "llm_config": {
                            "name": llm_config.get("name", "unknown"),
                            "model": llm_config.get("model", "unknown"),
                        },
                    }
        finally:
            # Always restore original threshold
            self.processor.confidence_threshold = original_threshold


# ------------------------------------------------------------------------------
# FastAPI ASGI App with Swagger Documentation
# ------------------------------------------------------------------------------

@app.function(timeout=900, image=image)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI ASGI application with full Swagger documentation.
    
    Provides endpoints for:
    - POST /sam3/count - Count objects in images
    - POST /sam3/area - Calculate object areas
    - POST /sam3/segment - Full segmentation with LLM
    
    Access documentation at:
    - /docs - Swagger UI
    - /redoc - ReDoc
    - /openapi.json - OpenAPI schema
    """
    import io
    import fastapi
    from fastapi import HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import traceback as tb
    
    # Create FastAPI app with documentation
    api = fastapi.FastAPI(
        title="SAM3 Agent API",
        description="""
## SAM3 Agent API - Pyramidal Image Segmentation

This API provides VLM-enhanced image segmentation, counting, and area calculation
using SAM3 (Segment Anything Model 3) with pyramidal batch processing.

### Features
- **Object Counting**: Count specific objects in images with VLM verification
- **Area Calculation**: Measure object areas with optional GSD for real-world units
- **Full Segmentation**: Complete image segmentation with prompt refinement

### Authentication
All endpoints require an `llm_config` object with your VLM provider credentials.
Supports any OpenAI-compatible API (OpenAI, vLLM, Anthropic, etc.)

### Example VLM Config
```json
{
  "base_url": "https://your-vllm-server.modal.run/v1",
  "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "api_key": ""
}
```
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "Counting", "description": "Object counting endpoints"},
            {"name": "Area", "description": "Area calculation endpoints"},
            {"name": "Segmentation", "description": "Full segmentation endpoints"},
            {"name": "Health", "description": "Health check endpoints"},
        ]
    )
    
    # Add CORS middleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Helper function to get image bytes
    def get_image_bytes(image_url: Optional[str]) -> bytes:
        """Get image bytes from image_url (supports HTTP/HTTPS URLs and data URIs) with validation"""
        import httpx
        from PIL import Image as PILImage
        
        if not image_url:
            raise HTTPException(status_code=400, detail="'image_url' is required")
        
        image_bytes = None
        
        # Check if it's a data URI (data:image/...;base64,...)
        if image_url.startswith('data:'):
            try:
                # Extract base64 part after the comma
                if ',' in image_url:
                    base64_part = image_url.split(',', 1)[1]
                else:
                    raise HTTPException(status_code=400, detail="Invalid data URI format: missing base64 data")
                
                image_bytes = base64.b64decode(base64_part)
                print(f"✓ Decoded base64 image from data URI: {len(image_bytes)} bytes")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                raise HTTPException(status_code=400, detail=f"Invalid base64 data in 'image_url': {e}")
        else:
            # Assume it's an HTTP/HTTPS URL
            try:
                print(f"📥 Downloading image from: {image_url}")
                with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                    resp = client.get(image_url)
                    
                    # Check status code directly instead of using raise_for_status()
                    if resp.status_code == 403:
                        raise HTTPException(
                            status_code=403,
                            detail=f"Forbidden: Image hosting service blocked the request. URL: {image_url}. Solution: Use data URI format (data:image/...;base64,...) instead."
                        )
                    elif resp.status_code >= 400:
                        raise HTTPException(
                            status_code=resp.status_code,
                            detail=f"Failed to download image from URL (HTTP {resp.status_code}): {resp.text[:200]}"
                        )
                    
                    image_bytes = resp.content
                print(f"✓ Downloaded image: {len(image_bytes)} bytes")
            except HTTPException:
                # Re-raise HTTPExceptions (403, 408, etc.) as-is
                raise
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=408,
                    detail=f"Request timeout: Failed to download image from URL within 30 seconds: {image_url}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image from URL: {e}"
                )
        
        # Validate image can be opened
        if image_bytes:
            try:
                img = PILImage.open(io.BytesIO(image_bytes))
                img_format = img.format  # Capture format BEFORE verify() closes the image
                img.verify()  # Verify it's a valid image
                print(f"✓ Valid image: {img_format} format")
            except Exception as e:
                # Log first 100 bytes for debugging
                preview = image_bytes[:100] if len(image_bytes) > 100 else image_bytes
                print(f"❌ Invalid image data. First 100 bytes: {preview}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid image data: {e}. Make sure you're providing a valid image file."
                )
        
        return image_bytes
    
    # Health check endpoint
    @api.get(
        "/health",
        tags=["Health"],
        summary="Health check",
        description="Check if the API is running"
    )
    async def health():
        return {"status": "ok", "service": "sam3-agent"}
    
    # Count endpoint
    @api.post(
        "/sam3/count",
        response_model=SAM3CountResponse,
        tags=["Counting"],
        summary="Count objects in an image",
        description="""
Count specific objects in an image using VLM-enhanced pyramidal SAM3 segmentation.

The VLM refines your prompt into optimal visual keywords, then SAM3 detects and counts
all matching objects with verification to reduce false positives.

**Image URL Requirements:**
- `image_url` is required (despite being marked Optional for backwards compatibility)
- Supports HTTP/HTTPS URLs and data URIs (data:image/...;base64,...)
- Some image hosting services may return 403 Forbidden - use data URIs as fallback
- Image must be a valid image format (JPEG, PNG, etc.)

**Example prompts:**
- "trees" - counts all trees
- "cars in the parking lot" - counts parked cars
- "people wearing red shirts" - counts specific people
        """,
        responses={
            200: {"description": "Successful count", "model": SAM3CountResponse},
            400: {"description": "Bad request - Missing image_url, invalid image URL format, invalid image data, base64 decode errors, or missing required fields"},
            403: {"description": "Forbidden - Image hosting service blocked the request (403 Client Error). Solution: Use data URI format (data:image/...;base64,...) instead of HTTP URL"},
            408: {"description": "Request timeout - Failed to download image from URL within 30 seconds"},
            500: {"description": "Server error - Internal processing error during segmentation or VLM verification"}
        }
    )
    async def count_objects(request: SAM3CountRequest):
        try:
            # Get image bytes
            image_bytes = get_image_bytes(request.image_url)
            
            # Convert Pydantic models to dicts
            llm_config_dict = request.llm_config.model_dump()
            pyramidal_config_dict = request.pyramidal_config.model_dump() if request.pyramidal_config else None
            
            print(f"📞 Calling sam3_count with prompt: '{request.prompt}'")
            result = SAM3Model().sam3_count.remote(
                image_bytes=image_bytes,
                text_prompt=request.prompt,
                llm_config=llm_config_dict,
                confidence_threshold=request.confidence_threshold,
                pyramidal_config=pyramidal_config_dict,
                max_retries=request.max_retries,
                include_obb=request.include_obb,
                obb_as_polygon=request.obb_as_polygon,
            )
            print(f"✓ sam3_count returned count: {result.get('count', 0)}")
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            traceback_str = tb.format_exc()
            print(f"❌ Error in sam3_count: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": error_msg, "traceback": traceback_str}
            )
    
    # Area endpoint
    @api.post(
        "/sam3/area",
        response_model=SAM3AreaResponse,
        tags=["Area"],
        summary="Calculate object areas in an image",
        description="""
Calculate the total and individual areas of objects in an image using VLM-enhanced
pyramidal SAM3 segmentation.

Optionally provide a Ground Sample Distance (GSD) in meters/pixel to get real-world
area measurements in square meters.

**Image URL Requirements:**
- `image_url` is required (despite being marked Optional for backwards compatibility)
- Supports HTTP/HTTPS URLs and data URIs (data:image/...;base64,...)
- Some image hosting services may return 403 Forbidden - use data URIs as fallback
- Image must be a valid image format (JPEG, PNG, etc.)

**Example prompts:**
- "solar panels" - measures solar panel coverage
- "buildings" - measures building footprints
- "agricultural fields" - measures crop areas
        """,
        responses={
            200: {"description": "Successful area calculation", "model": SAM3AreaResponse},
            400: {"description": "Bad request - Missing image_url, invalid image URL format, invalid image data, base64 decode errors, missing required fields, or invalid GSD (must be > 0)"},
            403: {"description": "Forbidden - Image hosting service blocked the request (403 Client Error). Solution: Use data URI format (data:image/...;base64,...) instead of HTTP URL"},
            408: {"description": "Request timeout - Failed to download image from URL within 30 seconds"},
            500: {"description": "Server error - Internal processing error during segmentation, VLM verification, or area calculation"}
        }
    )
    async def calculate_area(request: SAM3AreaRequest):
        try:
            # Get image bytes
            image_bytes = get_image_bytes(request.image_url)
            
            # Convert Pydantic models to dicts
            llm_config_dict = request.llm_config.model_dump()
            pyramidal_config_dict = request.pyramidal_config.model_dump() if request.pyramidal_config else None
            
            print(f"📞 Calling sam3_area with prompt: '{request.prompt}'")
            result = SAM3Model().sam3_area.remote(
                image_bytes=image_bytes,
                text_prompt=request.prompt,
                llm_config=llm_config_dict,
                gsd=request.gsd,
                confidence_threshold=request.confidence_threshold,
                pyramidal_config=pyramidal_config_dict,
                max_retries=request.max_retries,
                include_obb=request.include_obb,
                obb_as_polygon=request.obb_as_polygon,
            )
            print(f"✓ sam3_area returned {result.get('object_count', 0)} objects")
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            traceback_str = tb.format_exc()
            print(f"❌ Error in sam3_area: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": error_msg, "traceback": traceback_str}
            )
    
    # Segment endpoint
    @api.post(
        "/sam3/segment",
        response_model=SAM3SegmentResponse,
        tags=["Segmentation"],
        summary="Full image segmentation with LLM",
        description="""
Perform full image segmentation using LLM-guided SAM3 agent with pyramidal batch processing.

The LLM analyzes your prompt to understand what objects to segment, then guides SAM3
to produce accurate segmentation masks for all matching objects.

**Image URL Requirements:**
- `image_url` is required (despite being marked Optional for backwards compatibility)
- Supports HTTP/HTTPS URLs and data URIs (data:image/...;base64,...)
- Some image hosting services may return 403 Forbidden - use data URIs as fallback
- Image must be a valid image format (JPEG, PNG, etc.)

**Example prompts:**
- "segment all ships in the harbor"
- "find and segment all vehicles on the road"
- "segment the buildings in this aerial image"

Set `debug=true` to receive a visualization image in the response.
        """,
        responses={
            200: {"description": "Successful segmentation", "model": SAM3SegmentResponse},
            400: {"description": "Bad request - Missing image_url, invalid image URL format, invalid image data, base64 decode errors, or missing required fields"},
            403: {"description": "Forbidden - Image hosting service blocked the request (403 Client Error). Solution: Use data URI format (data:image/...;base64,...) instead of HTTP URL"},
            408: {"description": "Request timeout - Failed to download image from URL within 30 seconds"},
            500: {"description": "Server error - Internal processing error during segmentation or LLM agent execution"}
        }
    )
    async def segment_image(request: SAM3SegmentRequest):
        try:
            # Get image bytes
            image_bytes = get_image_bytes(request.image_url)
            
            # Convert Pydantic models to dicts
            llm_config_dict = request.llm_config.model_dump()
            pyramidal_config_dict = request.pyramidal_config.model_dump() if request.pyramidal_config else None
            
            print(f"📞 Calling sam3_segment with Qwen3-VL and prompt: '{request.prompt}'")
            result = SAM3Model().infer.remote(
                image_bytes=image_bytes,
                prompt=request.prompt,
                llm_config=llm_config_dict,
                debug=request.debug,
                confidence_threshold=request.confidence_threshold,
                pyramidal_config=pyramidal_config_dict,
                include_obb=True,
                obb_as_polygon=True,
            )
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            traceback_str = tb.format_exc()
            print(f"❌ Error in sam3_segment: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": error_msg, "traceback": traceback_str}
            )
    
    return api


# ------------------------------------------------------------------------------
# Optional: local quick test via `modal run modal_agent.py::local_test`
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def local_test():
    """
    Quick sanity check: runs SAM3 on a dummy red image, no HTTP involved.

    Run:
      modal run modal_agent.py::local_test
    """
    from PIL import Image
    from io import BytesIO

    img = Image.new("RGB", (128, 128), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    prompt = "find the red region"
    llm_config = {
        "base_url": "https://rockstar4119--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "api_key": "",
        "name": "qwen3-vl",
    }
    model = SAM3Model()
    result = model.infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=True,
    )

    print(json.dumps(result, indent=2)[:2000])
