from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# ------------------------------------------------------------------------------
# VLM Response Models
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
# SAM3 API Request/Response Models
# ------------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """LLM configuration for VLM-based processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
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
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": ""
            },
            "confidence_threshold": 0.3,
            "max_retries": 2
        }
    })
    
    prompt: str = Field(..., description="What objects to count (e.g., 'trees', 'cars', 'buildings')")
    image_url: Optional[str] = Field(None, description="Image URL (REQUIRED) - HTTP/HTTPS URL or data URI format")
    llm_config: LLMConfig = Field(..., description="VLM configuration for prompt refinement")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_retries: Optional[int] = Field(2, ge=0, le=5, description="Maximum retry attempts for verification")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal processing configuration")
    include_obb: Optional[bool] = Field(False, description="Include oriented bounding boxes (OBB) in response")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon coordinates instead of parametric format")


class DetectionInfo(BaseModel):
    """Individual detection information"""
    id: int = Field(..., description="Detection ID")
    score: float = Field(..., description="Confidence score")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    pixel_area: Optional[int] = Field(None, description="Area in pixels")
    obb: Optional[List[float]] = Field(None, description="Oriented bounding box [cx, cy, w, h, angle] in degrees")
    obb_polygon: Optional[List[List[float]]] = Field(None, description="OBB as polygon coordinates")


class SAM3CountResponse(BaseModel):
    """Response model for SAM3 object counting"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "count": 47,
            "visual_prompt": "tree",
            "object_type": "tree",
            "detections": []
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
            "gsd": 0.5,
            "llm_config": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": ""
            }
        }
    })
    
    prompt: str = Field(..., description="What objects to measure")
    image_url: Optional[str] = Field(None, description="Image URL")
    llm_config: LLMConfig = Field(..., description="VLM configuration")
    gsd: Optional[float] = Field(None, gt=0, description="Ground Sample Distance in meters/pixel")
    confidence_threshold: Optional[float] = Field(0.3, description="Minimum confidence threshold")
    max_retries: Optional[int] = Field(2, description="Maximum retry attempts")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal processing configuration")
    include_obb: Optional[bool] = Field(False, description="Include OBB")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon")


class SAM3AreaResponse(BaseModel):
    """Response model for SAM3 area calculation"""
    status: str = Field(..., description="Response status")
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
            "prompt": "segment all ships",
            "image_url": "https://example.com/image.jpg",
            "llm_config": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": ""
            },
            "debug": True
        }
    })
    
    prompt: str = Field(..., description="Segmentation prompt")
    image_url: Optional[str] = Field(None, description="Image URL")
    llm_config: LLMConfig = Field(..., description="LLM configuration")
    debug: Optional[bool] = Field(False, description="Return debug visualization")
    confidence_threshold: Optional[float] = Field(0.3, description="Minimum confidence threshold")
    pyramidal_config: Optional[PyramidalConfig] = Field(None, description="Pyramidal configuration")
    include_obb: Optional[bool] = Field(False, description="Include OBB")
    obb_as_polygon: Optional[bool] = Field(False, description="Return OBB as polygon")


class SAM3SegmentResponse(BaseModel):
    """Response model for SAM3 segmentation"""
    status: str = Field(..., description="Response status")
    summary: Optional[str] = Field(None, description="Summary of results")
    regions: Optional[List[Dict[str, Any]]] = Field(None, description="Segmented regions")
    debug_image_b64: Optional[str] = Field(None, description="Debug visualization")
    raw_sam3_json: Optional[Dict[str, Any]] = Field(None, description="Raw SAM3 output")
    llm_config: Optional[Dict[str, Any]] = Field(None, description="LLM config used")
    pyramidal_stats: Optional[Dict[str, Any]] = Field(None, description="Pyramidal stats")
    message: Optional[str] = Field(None, description="Error message if status is 'error'")
    traceback: Optional[str] = Field(None, description="Error traceback")
