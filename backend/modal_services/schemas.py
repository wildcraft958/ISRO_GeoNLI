"""
Pydantic schemas for VLM Modal services (Captioning & VQA).

These schemas define the OpenAI-compatible request/response format
and enable automatic Swagger documentation generation.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class ImageUrl(BaseModel):
    """Image URL object for multimodal content."""
    url: str = Field(..., description="Image URL (http/https) or base64 data URI")


class TextContent(BaseModel):
    """Text content item in a message."""
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ImageContent(BaseModel):
    """Image content item in a message."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl = Field(..., description="Image URL object")


# Union type for content items
ContentItem = Union[TextContent, ImageContent]


class Message(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant"] = Field(
        ..., 
        description="Role of the message author"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="Message content - either a string or list of content items (text/image)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "role": "user",
                    "content": "Describe this satellite image."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What objects are visible?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                    ]
                }
            ]
        }


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(
        default="qwen-vlm",
        description="Model name to use for inference"
    )
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen-caption-special",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this satellite image in detail."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw..."}}
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.7
            }
        }


# =============================================================================
# Response Models
# =============================================================================

class AssistantMessage(BaseModel):
    """Assistant message in response."""
    role: Literal["assistant"] = "assistant"
    content: str = Field(..., description="Generated text content")


class Choice(BaseModel):
    """Single choice in chat completion response."""
    index: int = Field(default=0, description="Index of this choice")
    message: AssistantMessage = Field(..., description="Generated message")
    finish_reason: Literal["stop", "length", "error"] = Field(
        default="stop",
        description="Reason for stopping generation"
    )


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default="chatcmpl-modal", description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    model: str = Field(..., description="Model used for generation")
    choices: List[Choice] = Field(..., description="List of generated choices")
    usage: Usage = Field(default_factory=Usage, description="Token usage statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-modal",
                "object": "chat.completion",
                "model": "qwen-caption-special",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This satellite image shows a coastal urban area..."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 85,
                    "total_tokens": 235
                }
            }
        }


# =============================================================================
# Async Job Models
# =============================================================================

class AsyncJobSubmitResponse(BaseModel):
    """Response when submitting an async job."""
    call_id: str = Field(..., description="Unique job ID for polling")
    status: Literal["pending"] = "pending"
    model: str = Field(..., description="Model used for the job")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "fc-abc123xyz",
                "status": "pending",
                "model": "qwen-caption-special"
            }
        }


class AsyncJobPendingResponse(BaseModel):
    """Response when job is still processing."""
    status: Literal["pending"] = "pending"
    call_id: str = Field(..., description="Job ID")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "pending",
                "call_id": "fc-abc123xyz"
            }
        }


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response with optional traceback."""
    error: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(None, description="Full traceback (if available)")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Image processing failed",
                "traceback": "Traceback (most recent call last):\n..."
            }
        }


class JobExpiredResponse(BaseModel):
    """Response when job has expired."""
    error: str = Field(
        default="Job result expired (older than 7 days)",
        description="Error message"
    )


# =============================================================================
# Health Check Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok"] = "ok"
    model: str = Field(..., description="Model name being served")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model": "qwen-caption-special"
            }
        }

