import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schemas.orchestrator_schema import AgentState
from app.services.modal_client import ModalServiceClient
from app.core.checkpoint import MongoDBCheckpointer

# Initialize Modal client
modal_client = ModalServiceClient()


def auto_router_func(state: AgentState) -> str:
    """
    Classifies intent from user query.
    Returns: "call_captioning", "call_vqa", or "call_grounding"
    """
    query = state.get("user_query", "").strip()
    
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
        state["execution_log"].append(f"Auto Router: LLM classification failed, defaulting to VQA - {str(e)}")
        return "call_vqa"


def conditional_router(state: AgentState) -> str:
    """
    Top-level router for modes.
    Returns single node name (no parallel execution needed).
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_grounding_node(state: AgentState) -> dict:
    """Grounding service node with retry"""
    try:
        result = modal_client.call_grounding(state["image_url"], state["user_query"] or "")
        state["execution_log"].append("Grounding service: Success")
        return {"grounding_result": result}
    except Exception as e:
        state["execution_log"].append(f"Grounding service: Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_node(state: AgentState) -> dict:
    """VQA service node with retry"""
    try:
        result = modal_client.call_vqa(state["image_url"], state["user_query"] or "")
        state["execution_log"].append("VQA service: Success")
        return {"vqa_result": result.get("answer", "")}
    except Exception as e:
        state["execution_log"].append(f"VQA service: Failed - {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_captioning_node(state: AgentState) -> dict:
    """Captioning service node with retry"""
    try:
        result = modal_client.call_captioning(state["image_url"])
        state["execution_log"].append("Captioning service: Success")
        return {"caption_result": result.get("caption", "")}
    except Exception as e:
        state["execution_log"].append(f"Captioning service: Failed - {str(e)}")
        raise


def build_workflow():
    """Construct the multimodal chatbot workflow"""
    
    workflow = StateGraph(AgentState)
    
    # === ADD NODES ===
    workflow.add_node("call_grounding", call_grounding_node)
    workflow.add_node("call_vqa", call_vqa_node)
    workflow.add_node("call_captioning", call_captioning_node)
    
    # === ADD EDGES ===
    
    # Entry: START -> conditional router
    workflow.add_conditional_edges(
        START,
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
    # Use MongoDB checkpointer
    checkpointer = MongoDBCheckpointer()
    
    return workflow.compile(checkpointer=checkpointer)


# Instantiate globally
app = build_workflow()

