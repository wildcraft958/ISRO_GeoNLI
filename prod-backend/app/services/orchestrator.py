import httpx
import logging
import re
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.core.database import SessionLocal
from app.db.models import ChatSession, Query
# Import the Factory from your registry
from app.services.modal_registry import get_model_adapter

logger = logging.getLogger(__name__)

# --- 1. State Definition ---
class ChatState(TypedDict):
    chat_id: str
    user_id: str
    query_text: str
    requested_mode: str 
    image_url: str
    image_type: str      # RGB, SAR, FCC, IR
    summary_context: str 
    final_mode: str      # VQA, CAPTIONING, GROUNDING, SAR_DIRECT, FCC_DIRECT, IR_DIRECT
    vqa_subtype: str    
    response_text: str
    response_metadata: Optional[Any]

# --- 2. Generic Helpers (Routing & Memory) ---
async def call_generic_classifier(system_prompt: str, user_query: str) -> str:
    """
    Calls Generic LLM for routing decisions (Strategy Pattern not applied here as it's a utility).
    """
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(settings.GENERIC_LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Standard OpenAI format parsing
            if "choices" in data:
                return data["choices"][0]["message"]["content"].upper()
            return data.get("response", "ERROR").upper()
        except Exception as e:
            logger.error(f"Generic LLM Classifier Failed: {e}")
            return "ERROR"

def parse_mode_from_text(text: str, options: list[str], default: str) -> str:
    for option in options:
        if re.search(rf"\b{option}\b", text):
            return option
    return default

async def summarize_conversation(current_summary: str, last_query: str, last_response: str) -> str:
    if not current_summary:
        prompt_content = f"Summarize:\nUser: {last_query}\nAI: {last_response}"
    else:
        prompt_content = (
            f"Update summary (max 150 words):\nCurrent: {current_summary}\n"
            f"New Q: {last_query}\nNew A: {last_response}\nNew Summary:"
        )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a context summarizer."},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.3, "max_tokens": 200
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(settings.GENERIC_LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return data.get("response", "")
        except Exception:
            # Fallback string concatenation if LLM fails
            return f"{current_summary}\nQ: {last_query} A: {last_response}"[-2000:]

# --- 3. Node: Mode Determination ---
async def determine_mode_node(state: ChatState):
    """
    Decides routing logic.
    """
    img_type = state["image_type"]
    
    # Bypass Logic for Non-RGB
    if img_type == "SAR": return {"final_mode": "SAR_DIRECT"}
    if img_type == "IR": return {"final_mode": "IR_DIRECT"}
    if img_type == "FCC": return {"final_mode": "FCC_DIRECT"}
    
    # Routing Logic for RGB
    req_mode = state["requested_mode"]
    if req_mode != "AUTO":
        return {"final_mode": req_mode}

    sys_prompt = "Classify query: GROUNDING (find/locate), CAPTIONING (describe image), VQA (specific question). One word only."
    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    detected_mode = parse_mode_from_text(llm_response, ["GROUNDING", "CAPTIONING", "VQA"], default="VQA")
    
    return {"final_mode": detected_mode}

# --- 4. Node: VQA Classification ---
async def classify_vqa_node(state: ChatState):
    """
    Determines VQA subtype using Generic LLM.
    """
    sys_prompt = "Classify VQA: BINARY (Yes/No), NUMERICAL (Count), GENERAL. One word only."
    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    subtype = parse_mode_from_text(llm_response, ["BINARY", "NUMERICAL", "GENERAL"], default="GENERAL")
    return {"vqa_subtype": subtype}

# --- 5. Node: Execute Model (DECOUPLED LOGIC) ---
async def execute_model_node(state: ChatState):
    """
    Delegates payload construction and parsing to the specific Model Adapter.
    This node no longer needs to know about specific JSON structures.
    """
    mode = state["final_mode"]
    subtype = state.get("vqa_subtype", "GENERAL")
    
    # 1. Get the correct Strategy (Adapter) from Registry
    adapter = get_model_adapter(mode, subtype)
    
    # 2. Get Configs & Payload from Adapter
    target_url = adapter.get_url()
    
    # The adapter internally handles all logic (including SAR/IR prompt selection via handlers)
    payload = adapter.construct_payload(state)
    
    # 3. Call API
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Calling Model: {target_url} Mode: {mode}")
            resp = await client.post(target_url, json=payload)
            resp.raise_for_status()
            
            # 4. Parse using Adapter
            # The adapter handles extraction logic (e.g. choices[0]... vs .response)
            text_out, metadata = adapter.parse_response(resp.json())
            
            return {"response_text": text_out, "response_metadata": metadata}

        except Exception as e:
            logger.error(f"Error calling Model ({target_url}): {e}")
            return {"response_text": "Error processing your request with the AI model."}

# --- 6. Node: Memory ---
async def memory_node(state: ChatState):
    db = SessionLocal()
    try:
        new_query = Query(
            chat_id=state["chat_id"],
            user_id=state["user_id"],
            query_text=state["query_text"],
            query_mode=f"{state['final_mode']}::{state.get('vqa_subtype', '')}",
            response_text=state["response_text"],
            response_metadata=state.get("response_metadata")
        )
        db.add(new_query)
        db.commit()
        
        updated_summary = await summarize_conversation(
            state.get("summary_context", ""), 
            state["query_text"], 
            state["response_text"]
        )
        
        session = db.query(ChatSession).filter(ChatSession.chat_id == state["chat_id"]).first()
        if session:
            session.summary_context = updated_summary
            db.commit()
            
        return {"summary_context": updated_summary}
    except Exception as e:
        logger.error(f"Memory Error: {e}")
        return {}
    finally:
        db.close()

# --- 7. Graph Construction ---
workflow = StateGraph(ChatState)

# Nodes
workflow.add_node("determine_mode", determine_mode_node)
workflow.add_node("classify_vqa", classify_vqa_node)
workflow.add_node("execute_model", execute_model_node)
workflow.add_node("memory", memory_node)

# Flow
workflow.set_entry_point("determine_mode")

def route_logic(state):
    # Only route to VQA Classifier if mode is VQA
    # SAR, IR, FCC, CAPTIONING, GROUNDING all go straight to execute
    if state["final_mode"] == "VQA":
        return "classify_vqa"
    return "execute_model"

workflow.add_conditional_edges("determine_mode", route_logic, {
    "classify_vqa": "classify_vqa",
    "execute_model": "execute_model"
})

workflow.add_edge("classify_vqa", "execute_model")
workflow.add_edge("execute_model", "memory")
workflow.add_edge("memory", END)

chat_app = workflow.compile()