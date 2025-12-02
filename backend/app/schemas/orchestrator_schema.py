from typing import TypedDict, List, Optional, Literal, Any
from pydantic import BaseModel


class AgentState(TypedDict):
    """Complete state for the multimodal chatbot workflow"""
    session_id: str
    image_url: str
    user_query: Optional[str]
    mode: Literal["auto", "grounding", "vqa", "captioning"]
    
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
    image_url: str
    query: Optional[str] = None
    mode: Literal["auto", "grounding", "vqa", "captioning"] = "auto"


class ChatResponse(BaseModel):
    """Response model for chat orchestrator endpoint"""
    session_id: str
    response_type: str  # "caption", "answer", "boxes"
    content: Any
    execution_log: List[str]

