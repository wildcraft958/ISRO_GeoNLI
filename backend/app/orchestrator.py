"""
LangGraph Orchestrator for Multimodal Chatbot.

This module defines the workflow for routing queries to specialized VLM services
(Grounding, VQA, Captioning) with automatic modality detection, optional
IR2RGB preprocessing, and VQA sub-classification.

Workflow:
    START -> detect_modality -> preprocessing_router
        -> (preprocess_ir2rgb OR route_to_service)
            -> conditional_router (VQA/Grounding/Captioning)
                -> IF VQA: vqa_subclassify_router
                    -> call_vqa_yesno OR call_vqa_general OR call_vqa_counting OR call_vqa_area
                -> IF Grounding: call_grounding
                -> IF Captioning: call_captioning

VQA Sub-Types:
    - yesno: Questions answerable with yes/no
    - general: Open-ended questions (what, why, how, describe)
    - counting: Count objects/features (how many)
    - area: Area/measurement calculations
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import END, START, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.checkpoint import MongoDBCheckpointer
from app.schemas.orchestrator_schema import AgentState
from app.services.ir2rgb_service import get_ir2rgb_service, is_ir2rgb_available
from app.services.modal_client import ModalServiceClient
from app.services.modality_router import (
    get_modality_router_service,
    is_modality_detection_available,
)
from app.services.resnet_classifier_client import (
    ResNetClassifierError,
    get_resnet_classifier_client,
    is_resnet_classifier_available,
)

logger = logging.getLogger(__name__)

# Initialize Modal client
modal_client = ModalServiceClient()


# =============================================================================
# Memory Helper Functions
# =============================================================================

def append_user_message(state: AgentState) -> dict:
    """Append user message to state.messages."""
    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    image_url = state.get("image_url", "")
    
    user_message = {
        "role": "user",
        "content": user_query if user_query else "[Image only]",
        "image_url": image_url,
        "timestamp": datetime.now(timezone.utc),
    }
    
    messages.append(user_message)
    return {"messages": messages}


def append_assistant_message(
    state: AgentState,
    content: str,
    response_type: str,
    metadata: Optional[dict] = None
) -> dict:
    """Append assistant message to state.messages."""
    messages = state.get("messages", [])
    
    assistant_message = {
        "role": "assistant",
        "content": content,
        "response_type": response_type,
        "image_url": state.get("image_url"),
        "timestamp": datetime.now(timezone.utc),
        "metadata": metadata or {},
    }
    
    messages.append(assistant_message)
    return {"messages": messages}


def integrate_user_memory(state: AgentState, user_memory: Optional[dict]) -> dict:
    """
    Integrate user memory into session context.
    
    Merges user preferences and conversation summaries into session_context.
    """
    if not user_memory:
        return {}
    
    session_context = state.get("session_context", {})
    user_preferences = user_memory.get("preferences", {})
    
    # Merge user preferences into session context
    if "user_preferences" not in session_context:
        session_context["user_preferences"] = {}
    session_context["user_preferences"].update(user_preferences)
    
    # Add conversation summaries to context
    conversation_summaries = user_memory.get("conversation_summaries", [])
    if conversation_summaries:
        session_context["previous_summaries"] = conversation_summaries[-3:]  # Last 3 summaries
    
    return {"session_context": session_context}


# =============================================================================
# Modality Detection Node
# =============================================================================

def detect_modality_node(state: AgentState) -> dict:
    """
    Detect image modality and configure preprocessing flags.
    
    This node runs first in the workflow to automatically detect
    whether the image is RGB, infrared, SAR, or unknown.
    
    Detection strategy:
    1. First, run statistical detection (fast, local)
    2. If result is SAR or uncertain, use ResNet classifier as fallback
    3. For infrared images, auto-enable IR2RGB preprocessing
    
    Returns:
        State updates with detected_modality, modality_confidence,
        resnet_classification_used, and potentially needs_ir2rgb.
    """
    update: dict = {
        "resnet_classification_used": False,
        "modality_confidence": None,
    }
    
    # Check if modality detection is enabled
    if not state.get("modality_detection_enabled", True):
        state["execution_log"].append("Modality Detection: Disabled by request")
        update["detected_modality"] = None
        update["modality_diagnostics"] = {"reason": "disabled"}
        return update
    
    image_url = state["image_url"]
    statistical_modality = None
    statistical_diagnostics = {}
    
    # ==========================================================================
    # Step 1: Statistical Detection (fast, local)
    # ==========================================================================
    if is_modality_detection_available():
        try:
            state["execution_log"].append(
                "Modality Detection: Running statistical analysis..."
            )
            
            modality_service = get_modality_router_service()
            statistical_modality, statistical_diagnostics = (
                modality_service.detect_modality_from_url(
                    image_url,
                    metadata_priority=True
                )
            )
            
            reason = statistical_diagnostics.get("reason", "unknown")
            state["execution_log"].append(
                f"Modality Detection (Statistical): '{statistical_modality}' "
                f"(reason: {reason})"
            )
            
        except Exception as e:
            logger.warning(f"Statistical modality detection failed: {e}")
            state["execution_log"].append(
                f"Modality Detection (Statistical): Failed - {str(e)}"
            )
            statistical_modality = "unknown"
            statistical_diagnostics = {"error": str(e)}
    else:
        state["execution_log"].append(
            "Modality Detection (Statistical): Skipped (dependencies unavailable)"
        )
        statistical_modality = "unknown"
    
    # ==========================================================================
    # Step 2: ResNet Fallback for SAR or Uncertain Cases
    # ==========================================================================
    # Use ResNet when:
    # - Statistical detection says SAR (needs confirmation)
    # - Statistical detection is uncertain/unknown
    # - Statistical detection failed
    needs_resnet_fallback = statistical_modality in ("sar", "unknown")
    
    final_modality = statistical_modality
    final_confidence = None
    
    if needs_resnet_fallback and is_resnet_classifier_available():
        try:
            state["execution_log"].append(
                "Modality Detection: Using ResNet classifier for confirmation..."
            )
            
            resnet_client = get_resnet_classifier_client()
            resnet_modality, resnet_confidence, probabilities = (
                resnet_client.classify_from_url(image_url)
            )
            
            update["resnet_classification_used"] = True
            final_modality = resnet_modality
            final_confidence = resnet_confidence
            
            state["execution_log"].append(
                f"Modality Detection (ResNet): '{resnet_modality}' "
                f"(confidence: {resnet_confidence:.3f})"
            )
            
            # Update diagnostics with ResNet results
            statistical_diagnostics["resnet_result"] = {
                "modality": resnet_modality,
                "confidence": resnet_confidence,
                "probabilities": probabilities,
            }
            
        except ResNetClassifierError as e:
            logger.warning(f"ResNet classification failed: {e}")
            state["execution_log"].append(
                f"Modality Detection (ResNet): Failed - {str(e)}. "
                f"Using statistical result."
            )
            # Keep statistical result as fallback
            
        except Exception as e:
            logger.error(f"Unexpected ResNet error: {e}", exc_info=True)
            state["execution_log"].append(
                f"Modality Detection (ResNet): Unexpected error - {str(e)}"
            )
    
    elif needs_resnet_fallback:
        state["execution_log"].append(
            "Modality Detection (ResNet): Skipped (service unavailable)"
        )
    
    # ==========================================================================
    # Step 3: Set Final Results and Configure Preprocessing
    # ==========================================================================
    update["detected_modality"] = final_modality
    update["modality_confidence"] = final_confidence
    update["modality_diagnostics"] = statistical_diagnostics
    
    # Log final detection result
    confidence_str = f", confidence: {final_confidence:.3f}" if final_confidence else ""
    resnet_str = " (ResNet)" if update["resnet_classification_used"] else " (Statistical)"
    state["execution_log"].append(
        f"Modality Detection: Final result '{final_modality}'{confidence_str}{resnet_str}"
    )
    
    # Auto-enable IR2RGB for infrared images
    if final_modality == "infrared":
        if not state.get("needs_ir2rgb", False):
            update["needs_ir2rgb"] = True
            state["execution_log"].append(
                "Modality Detection: Auto-enabled IR2RGB for infrared image"
            )
            
            # Set default IR2RGB channels if not provided
            if not state.get("ir2rgb_channels"):
                update["ir2rgb_channels"] = ["NIR", "R", "G"]
                state["execution_log"].append(
                    "Modality Detection: Using default channels ['NIR', 'R', 'G']"
                )
    
    # Log warning for SAR images
    elif final_modality == "sar":
        state["execution_log"].append(
            "Modality Detection: WARNING - SAR image detected. "
            "VLM processing may produce suboptimal results."
        )
        logger.warning(
            f"SAR image detected for URL: {image_url[:100]}... "
            "Consider specialized SAR processing."
        )
    
    return update


# =============================================================================
# IR2RGB Preprocessing Node
# =============================================================================

def preprocess_ir2rgb_node(state: AgentState) -> dict:
    """
    Preprocessing node: Convert IR/multispectral image to RGB if needed.
    
    This node runs before routing to any service if needs_ir2rgb is True.
    It converts FCC (False Color Composite) images to standard RGB.
    Also appends user message if not already present.
    """
    # Append user message first if needed
    update = {}
    messages = state.get("messages", [])
    if not messages or messages[-1].get("role") != "user":
        update = append_user_message(state)
    
    if not state.get("needs_ir2rgb", False):
        state["execution_log"].append("IR2RGB: Skipped (not needed)")
        return update
    
    if not is_ir2rgb_available():
        state["execution_log"].append("IR2RGB: Skipped (model weights not available)")
        return update
    
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
                **update,
                "image_url": converted_url,
                "original_image_url": original_url,
            }
        else:
            error = result.get("error", "Unknown error")
            state["execution_log"].append(f"IR2RGB: Conversion failed - {error}")
            # Continue with original image if conversion fails
            return {
                **update,
                "original_image_url": original_url,
            }
    
    except Exception as e:
        logger.error(f"IR2RGB preprocessing failed: {e}")
        state["execution_log"].append(f"IR2RGB: Failed - {str(e)}")
        return {
            **update,
            "original_image_url": state["image_url"],
        }


# =============================================================================
# Task Router Functions
# =============================================================================

def auto_router_func(state: AgentState) -> str:
    """
    Classifies intent from user query using Modal-deployed LLM.
    
    Uses the general LLM service with a task classification system prompt
    to determine whether the query should be routed to:
    - VQA (visual question answering)
    - Grounding (object detection/localization)
    - Captioning (image description)
    
    Returns:
        One of: "call_captioning", "vqa_subclassify", or "call_grounding"
    """
    query = state.get("user_query", "")
    query = query.strip() if query else ""
    
    # Rule 1: No query -> Captioning
    if not query:
        state["execution_log"].append("Auto Router: No query detected -> Captioning")
        return "call_captioning"
    
    # Rule 2: Use Modal LLM for classification
    try:
        detected_modality = state.get("detected_modality")
        has_image = bool(state.get("image_url"))
        
        state["execution_log"].append(
            f"Auto Router: Calling Modal LLM router (query: '{query[:50]}...', "
            f"modality: {detected_modality})"
        )
        
        result = modal_client.call_task_router(
            query=query,
            detected_modality=detected_modality,
            has_image=has_image
        )
        
        task = result.get("task", "vqa")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        state["execution_log"].append(
            f"Auto Router: Classified as '{task}' (confidence: {confidence:.2f}, "
            f"reasoning: {reasoning})"
        )
        
        # Map task to node name
        # VQA goes to vqa_subclassify for sub-classification
        task_map = {
            "vqa": "vqa_subclassify",
            "grounding": "call_grounding",
            "captioning": "call_captioning",
        }
        
        return task_map.get(task, "vqa_subclassify")
    
    except Exception as e:
        # Fallback to VQA on router error
        logger.warning(f"Auto router Modal LLM classification failed: {e}")
        state["execution_log"].append(
            f"Auto Router: Modal LLM classification failed, defaulting to VQA - {str(e)}"
        )
        return "vqa_subclassify"


def conditional_router(state: AgentState) -> str:
    """
    Top-level router for modes.
    
    Routes based on explicit mode or uses auto_router_func for auto mode.
    VQA requests are routed to vqa_subclassify for sub-classification.
    
    Returns:
        One of: "call_captioning", "vqa_subclassify", or "call_grounding"
    """
    mode = state["mode"]
    
    if mode == "auto":
        state["execution_log"].append("Router: Auto mode -> classifying intent")
        return auto_router_func(state)
    elif mode == "vqa":
        # VQA goes to sub-classification
        state["execution_log"].append("Router: Explicit VQA mode -> sub-classifying")
        return "vqa_subclassify"
    else:
        # Explicit mode (captioning, grounding)
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
# VQA Sub-Classification Router
# =============================================================================

def vqa_subclassify_node(state: AgentState) -> dict:
    """
    Classify VQA query into subtypes using Modal LLM.
    
    This node determines the specific type of VQA question:
    - yesno: Questions answerable with yes/no
    - general: Open-ended questions
    - counting: Count objects/features
    - area: Area/measurement calculations
    
    Returns:
        State updates with vqa_type, vqa_type_confidence, vqa_type_reasoning
    """
    query = state.get("user_query", "")
    detected_modality = state.get("detected_modality")
    
    try:
        state["execution_log"].append(
            f"VQA Subclassify: Classifying query type..."
        )
        
        result = modal_client.call_vqa_subclassifier(
            query=query,
            detected_modality=detected_modality
        )
        
        vqa_type = result.get("vqa_type", "general")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        state["execution_log"].append(
            f"VQA Subclassify: Type '{vqa_type}' (confidence: {confidence:.2f}, "
            f"reasoning: {reasoning})"
        )
        
        return {
            "vqa_type": vqa_type,
            "vqa_type_confidence": confidence,
            "vqa_type_reasoning": reasoning,
        }
    
    except Exception as e:
        logger.warning(f"VQA subclassification failed: {e}")
        state["execution_log"].append(
            f"VQA Subclassify: Failed - {str(e)}, defaulting to 'general'"
        )
        return {
            "vqa_type": "general",
            "vqa_type_confidence": 0.0,
            "vqa_type_reasoning": f"Classification failed: {str(e)}",
        }


def vqa_subtype_router(state: AgentState) -> str:
    """
    Route to appropriate VQA sub-type node based on classification.
    
    Returns:
        One of: "call_vqa_yesno", "call_vqa_general", 
                "call_vqa_counting", or "call_vqa_area"
    """
    vqa_type = state.get("vqa_type", "general")
    
    type_map = {
        "yesno": "call_vqa_yesno",
        "general": "call_vqa_general",
        "counting": "call_vqa_counting",
        "area": "call_vqa_area",
    }
    
    node = type_map.get(vqa_type, "call_vqa_general")
    state["execution_log"].append(f"VQA Router: Type '{vqa_type}' -> {node}")
    return node


# =============================================================================
# Service Nodes
# =============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_grounding_node(state: AgentState) -> dict:
    """Grounding service node with retry logic."""
    try:
        image_url = state["image_url"]
        query = state.get("user_query") or ""
        
        state["execution_log"].append("Grounding: Calling service...")
        result = modal_client.call_grounding(image_url, query)
        
        bbox_count = len(result.get("bboxes", []))
        state["execution_log"].append(f"Grounding: Success ({bbox_count} boxes detected)")
        
        # Append assistant message
        content = f"Found {bbox_count} bounding box(es)" if bbox_count > 0 else "No objects detected"
        message_update = append_assistant_message(
            state,
            content,
            "boxes",
            metadata={"grounding_result": result, "execution_log": state.get("execution_log", [])}
        )
        
        return {
            "grounding_result": result,
            **message_update
        }
    
    except Exception as e:
        logger.error(f"Grounding service failed: {e}")
        state["execution_log"].append(f"Grounding: Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_captioning_node(state: AgentState) -> dict:
    """Captioning service node with retry logic."""
    try:
        image_url = state["image_url"]
        
        state["execution_log"].append("Captioning: Calling service...")
        result = modal_client.call_captioning(image_url)
        
        caption = result.get("caption", "")
        caption_preview = caption[:50] + "..." if len(caption) > 50 else caption
        state["execution_log"].append(f"Captioning: Success ({caption_preview})")
        
        # Append assistant message
        message_update = append_assistant_message(
            state,
            caption,
            "caption",
            metadata={"caption_result": result, "execution_log": state.get("execution_log", [])}
        )
        
        return {
            "caption_result": caption,
            **message_update
        }
    
    except Exception as e:
        logger.error(f"Captioning service failed: {e}")
        state["execution_log"].append(f"Captioning: Failed - {str(e)}")
        raise


# =============================================================================
# VQA Sub-Type Service Nodes
# =============================================================================

def _call_vqa_with_type(state: AgentState, vqa_type: str) -> dict:
    """
    Common VQA call logic for all sub-types.
    
    Args:
        state: Current agent state
        vqa_type: VQA sub-type ("yesno", "general", "counting", "area")
    
    Returns:
        State updates with vqa_result and messages
    """
    image_url = state["image_url"]
    query = state.get("user_query") or ""
    
    state["execution_log"].append(f"VQA ({vqa_type}): Calling service...")
    result = modal_client.call_vqa(image_url, query, vqa_type=vqa_type)
    
    answer = result.get("answer", "")
    answer_preview = answer[:50] + "..." if len(answer) > 50 else answer
    state["execution_log"].append(f"VQA ({vqa_type}): Success ({answer_preview})")
    
    # Append assistant message with vqa_type in metadata
    metadata = {
        "vqa_result": result, 
        "vqa_type": vqa_type,
        "execution_log": state.get("execution_log", [])
    }
    
    message_update = append_assistant_message(
        state,
        answer,
        "answer",
        metadata=metadata
    )
    
    return {
        "vqa_result": answer,
        **message_update
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_yesno_node(state: AgentState) -> dict:
    """
    VQA service node for yes/no questions.
    
    Handles questions that can be answered with yes/no:
    - "Is there a building in the image?"
    - "Does this area contain water?"
    - "Are there any vehicles visible?"
    """
    try:
        return _call_vqa_with_type(state, "yesno")
    except Exception as e:
        logger.error(f"VQA (yesno) service failed: {e}")
        state["execution_log"].append(f"VQA (yesno): Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_general_node(state: AgentState) -> dict:
    """
    VQA service node for general/open-ended questions.
    
    Handles open-ended questions:
    - "What type of land cover is shown?"
    - "Describe the urban development in this area"
    - "What can you identify in this satellite image?"
    """
    try:
        return _call_vqa_with_type(state, "general")
    except Exception as e:
        logger.error(f"VQA (general) service failed: {e}")
        state["execution_log"].append(f"VQA (general): Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_counting_node(state: AgentState) -> dict:
    """
    VQA service node for counting questions.
    
    Handles questions asking to count objects/features:
    - "How many buildings are in this image?"
    - "Count the number of vehicles"
    - "How many ships can you see?"
    """
    try:
        return _call_vqa_with_type(state, "counting")
    except Exception as e:
        logger.error(f"VQA (counting) service failed: {e}")
        state["execution_log"].append(f"VQA (counting): Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_area_node(state: AgentState) -> dict:
    """
    VQA service node for area/measurement questions.
    
    Handles questions asking for area calculations:
    - "What is the area of the forest region?"
    - "Calculate the size of the water body"
    - "Estimate the area covered by buildings"
    """
    try:
        return _call_vqa_with_type(state, "area")
    except Exception as e:
        logger.error(f"VQA (area) service failed: {e}")
        state["execution_log"].append(f"VQA (area): Failed - {str(e)}")
        raise


# =============================================================================
# Passthrough Node
# =============================================================================

def route_to_service_node(state: AgentState) -> dict:
    """
    Passthrough node for cases when no preprocessing is needed.
    Appends user message to state if not already present.
    """
    # Append user message if this is the first time through
    messages = state.get("messages", [])
    if not messages or messages[-1].get("role") != "user":
        return append_user_message(state)
    return {}


# =============================================================================
# Workflow Builder
# =============================================================================

def build_workflow():
    """
    Construct the multimodal chatbot workflow with VQA sub-classification.
    
    Graph structure:
        START -> detect_modality -> preprocessing_router
            -> preprocess_ir2rgb -> conditional_router
            -> route_to_service -> conditional_router
                -> call_grounding -> END
                -> call_captioning -> END
                -> vqa_subclassify -> vqa_subtype_router
                    -> call_vqa_yesno -> END
                    -> call_vqa_general -> END
                    -> call_vqa_counting -> END
                    -> call_vqa_area -> END
    
    The detect_modality node runs first to classify the image and
    auto-configure IR2RGB preprocessing for infrared images.
    
    VQA queries go through sub-classification to determine the
    specific type (yesno, general, counting, area).
    """
    workflow = StateGraph(AgentState)
    
    # === ADD NODES ===
    # Modality detection and preprocessing
    workflow.add_node("detect_modality", detect_modality_node)
    workflow.add_node("preprocess_ir2rgb", preprocess_ir2rgb_node)
    workflow.add_node("route_to_service", route_to_service_node)
    
    # Non-VQA service nodes
    workflow.add_node("call_grounding", call_grounding_node)
    workflow.add_node("call_captioning", call_captioning_node)
    
    # VQA sub-classification and service nodes
    workflow.add_node("vqa_subclassify", vqa_subclassify_node)
    workflow.add_node("call_vqa_yesno", call_vqa_yesno_node)
    workflow.add_node("call_vqa_general", call_vqa_general_node)
    workflow.add_node("call_vqa_counting", call_vqa_counting_node)
    workflow.add_node("call_vqa_area", call_vqa_area_node)
    
    # === ADD EDGES ===
    
    # Entry: START -> detect_modality
    workflow.add_edge(START, "detect_modality")
    
    # After modality detection, route to preprocessing
    workflow.add_conditional_edges(
        "detect_modality",
        preprocessing_router,
        {
            "preprocess_ir2rgb": "preprocess_ir2rgb",
            "route_to_service": "route_to_service",
        }
    )
    
    # After preprocessing, route to service (VQA goes to sub-classify)
    workflow.add_conditional_edges(
        "preprocess_ir2rgb",
        conditional_router,
        {
            "call_grounding": "call_grounding",
            "call_captioning": "call_captioning",
            "vqa_subclassify": "vqa_subclassify",
        }
    )
    
    # Direct routing if no preprocessing needed
    workflow.add_conditional_edges(
        "route_to_service",
        conditional_router,
        {
            "call_grounding": "call_grounding",
            "call_captioning": "call_captioning",
            "vqa_subclassify": "vqa_subclassify",
        }
    )
    
    # After VQA sub-classification, route to appropriate VQA node
    workflow.add_conditional_edges(
        "vqa_subclassify",
        vqa_subtype_router,
        {
            "call_vqa_yesno": "call_vqa_yesno",
            "call_vqa_general": "call_vqa_general",
            "call_vqa_counting": "call_vqa_counting",
            "call_vqa_area": "call_vqa_area",
        }
    )
    
    # All service nodes end the workflow
    workflow.add_edge("call_grounding", END)
    workflow.add_edge("call_captioning", END)
    workflow.add_edge("call_vqa_yesno", END)
    workflow.add_edge("call_vqa_general", END)
    workflow.add_edge("call_vqa_counting", END)
    workflow.add_edge("call_vqa_area", END)
    
    # === CHECKPOINTING ===
    checkpointer = MongoDBCheckpointer()
    
    logger.info(
        "Building LangGraph workflow with modality detection, IR2RGB preprocessing, "
        "and VQA sub-classification"
    )
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Global Workflow Instance
# =============================================================================

# Instantiate globally for reuse
app = build_workflow()
