from typing import TypedDict, List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """Complete state for the multimodal chatbot workflow"""
    session_id: str
    image_url: str
    user_query: Optional[str]
    mode: Literal["auto", "grounding", "vqa", "captioning"]
    
    # Modality detection
    modality_detection_enabled: bool  # Whether to auto-detect modality
    detected_modality: Optional[str]  # "rgb", "infrared", "sar", "unknown"
    modality_diagnostics: Optional[Dict[str, Any]]  # Detection diagnostics
    
    # IR2RGB preprocessing (optional)
    needs_ir2rgb: bool  # Whether image needs IR2RGB conversion
    ir2rgb_channels: Optional[List[str]]  # e.g., ["NIR", "R", "G"]
    ir2rgb_synthesize: Optional[Literal["R", "G", "B"]]  # Which channel to synthesize
    original_image_url: Optional[str]  # Original URL before IR2RGB conversion
    
    # Service output (only one populated per invoke)
    caption_result: Optional[str]
    vqa_result: Optional[str]
    grounding_result: Optional[dict]  # {bboxes: [{x, y, w, h, label, conf}]}
    
    # Session memory
    messages: List[dict]  # [{role, content, source, timestamp}]
    session_context: dict  # {conversation_summary, user_prefs, execution_stats}
    
    # Debugging
    execution_log: List[str]


class ChatRequest(BaseModel):
    """Request model for chat orchestrator endpoint"""
    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        None,
        description="User identifier for user memory integration (optional)"
    )
    image_url: str
    query: Optional[str] = None
    mode: Literal["auto", "grounding", "vqa", "captioning"] = "auto"
    
    # Modality detection (optional)
    modality_detection_enabled: bool = Field(
        default=True,
        description="Whether to auto-detect image modality (rgb/infrared/sar)"
    )
    
    # IR2RGB preprocessing parameters (optional)
    needs_ir2rgb: bool = Field(
        default=False,
        description="Whether to convert multispectral/FCC image to RGB before processing"
    )
    ir2rgb_channels: Optional[List[str]] = Field(
        default=None,
        description="Channel order of input image, e.g., ['NIR', 'R', 'G']"
    )
    ir2rgb_synthesize: Optional[Literal["R", "G", "B"]] = Field(
        default="B",
        description="Which RGB channel to synthesize from the NIR band"
    )


class ChatResponse(BaseModel):
    """Response model for chat orchestrator endpoint"""
    session_id: str
    response_type: str  # "caption", "answer", "boxes"
    content: Any
    execution_log: List[str]
    message_id: Optional[str] = Field(
        default=None,
        description="ID of the persisted assistant message"
    )
    
    # Modality detection result
    detected_modality: Optional[str] = Field(
        default=None,
        description="Detected image modality: rgb, infrared, sar, or unknown"
    )
    
    # IR2RGB conversion result (if applied)
    converted_image_url: Optional[str] = Field(
        default=None,
        description="Data URI of converted RGB image (if IR2RGB was applied)"
    )
    original_image_url: Optional[str] = Field(
        default=None,
        description="Original image URL (if IR2RGB was applied)"
    )


class IR2RGBRequest(BaseModel):
    """Standalone request model for IR2RGB conversion endpoint"""
    image_url: str = Field(..., description="URL of multispectral image to convert")
    channels: List[str] = Field(
        ..., 
        min_length=3, 
        max_length=3,
        description="Channel order, e.g., ['NIR', 'R', 'G']"
    )
    synthesize_channel: Literal["R", "G", "B"] = Field(
        default="B",
        description="Which RGB channel to synthesize"
    )


class IR2RGBResponse(BaseModel):
    """Response model for IR2RGB conversion endpoint"""
    success: bool
    rgb_image_url: Optional[str] = Field(
        default=None,
        description="Data URI of converted RGB image"
    )
    rgb_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded RGB image"
    )
    format: Optional[str] = Field(default=None, description="Image format")
    dimensions: Optional[dict] = Field(default=None, description="Image dimensions")
    size_bytes: Optional[int] = Field(default=None, description="Image size in bytes")
    error: Optional[str] = Field(default=None, description="Error message if failed")
