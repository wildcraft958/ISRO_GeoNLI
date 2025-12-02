"""
LangGraph Orchestrator for Multimodal Chatbot.

This module defines the workflow for routing queries to specialized VLM services
(Grounding, VQA, Captioning) with optional IR2RGB preprocessing for
multispectral satellite images.

Workflow:
    START -> [IR2RGB Preprocessing (optional)] -> Router -> Service -> END
"""

import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas.orchestrator_schema import AgentState
from app.services.modal_client import ModalServiceClient
from app.services.ir2rgb_service import get_ir2rgb_service, is_ir2rgb_available
from app.core.checkpoint import MongoDBCheckpointer

logger = logging.getLogger(__name__)

# Initialize Modal client
modal_client = ModalServiceClient()


# =============================================================================
# IR2RGB Preprocessing Node
# =============================================================================

def preprocess_ir2rgb_node(state: AgentState) -> dict:
    """
    Preprocessing node: Convert IR/multispectral image to RGB if needed.
    
    This node runs before routing to any service if needs_ir2rgb is True.
    It converts FCC (False Color Composite) images to standard RGB.
    """
    if not state.get("needs_ir2rgb", False):
        state["execution_log"].append("IR2RGB: Skipped (not needed)")
        return {}
    
    if not is_ir2rgb_available():
        state["execution_log"].append("IR2RGB: Skipped (model weights not available)")
        return {}
    
    try:
        state["execution_log"].append("IR2RGB: Starting conversion...")
        
        ir2rgb_service = get_ir2rgb_service()
        channels = state.get("ir2rgb_channels", ["NIR", "R", "G"])
        synthesize = state.get("ir2rgb_synthesize", "B")
        original_url = state["image_url"]
        
        # Convert image
        result = ir2rgb_service.convert_from_url(
            state["image_url"],
            channels,
            synthesize
        )
        
        if result.get("success"):
            converted_url = result["rgb_image_url"]
            dimensions = result.get("dimensions", {})
            state["execution_log"].append(
                f"IR2RGB: Conversion successful "
                f"(synthesized {synthesize}, {dimensions.get('width')}x{dimensions.get('height')})"
            )
            
            return {
                "image_url": converted_url,
                "original_image_url": original_url,
            }
        else:
            error = result.get("error", "Unknown error")
            state["execution_log"].append(f"IR2RGB: Conversion failed - {error}")
            # Continue with original image if conversion fails
            return {"original_image_url": original_url}
    
    except Exception as e:
        logger.error(f"IR2RGB preprocessing failed: {e}")
        state["execution_log"].append(f"IR2RGB: Failed - {str(e)}")
        return {"original_image_url": state["image_url"]}


# =============================================================================
# Router Functions
# =============================================================================

def auto_router_func(state: AgentState) -> str:
    """
    Classifies intent from user query using LLM.
    
    Returns:
        One of: "call_captioning", "call_vqa", or "call_grounding"
    """
    query = state.get("user_query", "")
    query = query.strip() if query else ""
    
    # Rule 1: No query -> Captioning
    if not query:
        state["execution_log"].append("Auto Router: No query detected -> Captioning")
        return "call_captioning"
    
    # Rule 2: LLM Classification
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        classification_prompt = (
            f"Classify this user intent:\n\n"
            f"Query: \"{query}\"\n\n"
            f"Return ONLY ONE word:\n"
            f"- 'LOCATE' if user wants bounding boxes, object detection, or spatial info\n"
            f"- 'QA' if user asks a question (what, why, how, describe, explain, etc)\n"
            f"- 'DESCRIBE' if user wants general description\n\n"
            f"Response:"
        )
        response = llm.invoke(classification_prompt)
        intent = response.content.strip().upper()
        
        state["execution_log"].append(f"Auto Router: Classified as {intent}")
        
        # Map to service
        intent_map = {
            "LOCATE": "call_grounding",
            "QA": "call_vqa",
            "DESCRIBE": "call_vqa",  # Default QA for open-ended
        }
        return intent_map.get(intent, "call_vqa")
    
    except Exception as e:
        # Fallback to VQA on LLM error
        logger.warning(f"Auto router LLM classification failed: {e}")
        state["execution_log"].append(
            f"Auto Router: LLM classification failed, defaulting to VQA - {str(e)}"
        )
        return "call_vqa"


def conditional_router(state: AgentState) -> str:
    """
    Top-level router for modes.
    Returns single node name (no parallel execution).
    """
    mode = state["mode"]
    
    if mode == "auto":
        state["execution_log"].append("Router: Auto mode -> classifying intent")
        return auto_router_func(state)
    else:
        # Explicit mode (captioning, vqa, grounding)
        node = f"call_{mode}"
        state["execution_log"].append(f"Router: Explicit mode -> {node}")
        return node


def preprocessing_router(state: AgentState) -> str:
    """
    Router to determine if IR2RGB preprocessing is needed.
    Returns: "preprocess_ir2rgb" or "route_to_service"
    """
    if state.get("needs_ir2rgb", False):
        return "preprocess_ir2rgb"
    return "route_to_service"


# =============================================================================
# Service Nodes
# =============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_grounding_node(state: AgentState) -> dict:
    """Grounding service node with retry logic."""
    try:
        image_url = state["image_url"]
        query = state.get("user_query") or ""
        
        state["execution_log"].append(f"Grounding: Calling service...")
        result = modal_client.call_grounding(image_url, query)
        
        bbox_count = len(result.get("bboxes", []))
        state["execution_log"].append(f"Grounding: Success ({bbox_count} boxes detected)")
        
        return {"grounding_result": result}
    
    except Exception as e:
        logger.error(f"Grounding service failed: {e}")
        state["execution_log"].append(f"Grounding: Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_node(state: AgentState) -> dict:
    """VQA service node with retry logic."""
    try:
        image_url = state["image_url"]
        query = state.get("user_query") or ""
        
        state["execution_log"].append(f"VQA: Calling service...")
        result = modal_client.call_vqa(image_url, query)
        
        answer = result.get("answer", "")
        answer_preview = answer[:50] + "..." if len(answer) > 50 else answer
        state["execution_log"].append(f"VQA: Success ({answer_preview})")
        
        return {"vqa_result": answer}
    
    except Exception as e:
        logger.error(f"VQA service failed: {e}")
        state["execution_log"].append(f"VQA: Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_captioning_node(state: AgentState) -> dict:
    """Captioning service node with retry logic."""
    try:
        image_url = state["image_url"]
        
        state["execution_log"].append(f"Captioning: Calling service...")
        result = modal_client.call_captioning(image_url)
        
        caption = result.get("caption", "")
        caption_preview = caption[:50] + "..." if len(caption) > 50 else caption
        state["execution_log"].append(f"Captioning: Success ({caption_preview})")
        
        return {"caption_result": caption}
    
    except Exception as e:
        logger.error(f"Captioning service failed: {e}")
        state["execution_log"].append(f"Captioning: Failed - {str(e)}")
        raise


# =============================================================================
# Passthrough Node
# =============================================================================

def route_to_service_node(state: AgentState) -> dict:
    """
    Passthrough node for cases when no preprocessing is needed.
    This node does nothing but allows proper edge routing.
    """
    return {}


# =============================================================================
# Workflow Builder
# =============================================================================

def build_workflow():
    """
    Construct the multimodal chatbot workflow.
    
    Graph structure:
        START -> preprocessing_router
            -> preprocess_ir2rgb -> conditional_router -> service -> END
            -> route_to_service -> conditional_router -> service -> END
    """
    workflow = StateGraph(AgentState)
    
    # === ADD NODES ===
    workflow.add_node("preprocess_ir2rgb", preprocess_ir2rgb_node)
    workflow.add_node("route_to_service", route_to_service_node)
    workflow.add_node("call_grounding", call_grounding_node)
    workflow.add_node("call_vqa", call_vqa_node)
    workflow.add_node("call_captioning", call_captioning_node)
    
    # === ADD EDGES ===
    
    # Entry: START -> preprocessing router
    workflow.add_conditional_edges(
        START,
        preprocessing_router,
        {
            "preprocess_ir2rgb": "preprocess_ir2rgb",
            "route_to_service": "route_to_service",
        }
    )
    
    # After preprocessing, route to service
    workflow.add_conditional_edges(
        "preprocess_ir2rgb",
        conditional_router,
        {
            "call_grounding": "call_grounding",
            "call_vqa": "call_vqa",
            "call_captioning": "call_captioning",
        }
    )
    
    # Direct routing if no preprocessing needed
    workflow.add_conditional_edges(
        "route_to_service",
        conditional_router,
        {
            "call_grounding": "call_grounding",
            "call_vqa": "call_vqa",
            "call_captioning": "call_captioning",
        }
    )
    
    # All service nodes end the workflow
    workflow.add_edge("call_grounding", END)
    workflow.add_edge("call_vqa", END)
    workflow.add_edge("call_captioning", END)
    
    # === CHECKPOINTING ===
    checkpointer = MongoDBCheckpointer()
    
    logger.info("Building LangGraph workflow with IR2RGB preprocessing support")
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Global Workflow Instance
# =============================================================================

# Instantiate globally for reuse
app = build_workflow()
