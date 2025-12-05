import httpx
import logging
import re
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.db.models import ChatSession, Query

logger = logging.getLogger(__name__)

# --- 1. State Definition ---
class ChatState(TypedDict):
    chat_id: str
    user_id: str
    query_text: str
    requested_mode: str 
    image_url: str
    image_type: str      # RGB, SAR, FCC
    summary_context: str 
    final_mode: str      # VQA, CAPTIONING, GROUNDING, SAR_DIRECT, FCC_DIRECT
    vqa_subtype: str    
    response_text: str
    response_metadata: Optional[Any]

# --- 2. Helpers ---
async def call_generic_classifier(system_prompt: str, user_query: str) -> str:
    """Calls Generic LLM for RGB routing decisions."""
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
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or data.get("response", "")
            return raw_content.upper()
        except Exception as e:
            logger.error(f"Generic LLM Classifier Failed: {e}")
            return "ERROR"

def parse_mode_from_text(text: str, options: list[str], default: str) -> str:
    for option in options:
        if re.search(rf"\b{option}\b", text):
            return option
    return default

async def summarize_conversation(current_summary: str, last_query: str, last_response: str) -> str:
    # (Same summarization logic as before)
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
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or data.get("response", "")
        except Exception:
            return f"{current_summary}\nQ: {last_query} A: {last_response}"[-2000:]

# --- 3. Node: Mode Determination ---
async def determine_mode_node(state: ChatState):
    """
    Decides routing. 
    Crucial Update: Checks Image Type first.
    """
    img_type = state["image_type"]
    
    # --- BYPASS LOGIC ---
    if img_type == "SAR":
        return {"final_mode": "SAR_DIRECT"}
    
    if img_type == "FCC":
        return {"final_mode": "FCC_DIRECT"}
    
    # --- RGB LOGIC (Existing Pipeline) ---
    # If image_type is RGB (includes converted IR), we use the router
    req_mode = state["requested_mode"]
    if req_mode != "AUTO":
        return {"final_mode": req_mode}

    sys_prompt = (
        "Classify query: GROUNDING (find/locate), CAPTIONING (describe image), VQA (specific question). One word only."
    )
    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    detected_mode = parse_mode_from_text(llm_response, ["GROUNDING", "CAPTIONING", "VQA"], default="VQA")
    
    return {"final_mode": detected_mode}

# --- 4. Node: VQA Classification ---
async def classify_vqa_node(state: ChatState):
    # Only runs for RGB VQA path
    sys_prompt = "Classify VQA: BINARY (Yes/No), NUMERICAL (Count), GENERAL. One word only."
    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    subtype = parse_mode_from_text(llm_response, ["BINARY", "NUMERICAL", "GENERAL"], default="GENERAL")
    return {"vqa_subtype": subtype}

# --- 5. Node: Execute Model ---
async def execute_model_node(state: ChatState):
    mode = state["final_mode"]
    subtype = state.get("vqa_subtype", "GENERAL")
    
    target_url = settings.VQA_GENERAL_URL # Default
    
    # URL Selection
    if mode == "SAR_DIRECT":
        target_url = settings.SAR_URL
    elif mode == "FCC_DIRECT":
        target_url = settings.FCC_URL
    elif mode == "GROUNDING":
        target_url = settings.GROUNDING_URL
    elif mode == "CAPTIONING":
        target_url = settings.CAPTION_URL
    elif mode == "VQA":
        if subtype == "BINARY": target_url = settings.VQA_BINARY_URL
        elif subtype == "NUMERICAL": target_url = settings.VQA_NUMERICAL_URL
        else: target_url = settings.VQA_GENERAL_URL

    # Prepare Context
    context_header = f"Image Type: {state['image_type']}\n"
    if state["summary_context"]:
        context_header += f"History: {state['summary_context']}\n"
    
    full_prompt = f"{context_header}\nQuestion: {state['query_text']}"

    # Payload Construction
    # Assuming SAR/FCC models accept similar inputs. Adjust if they need specific formats.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": state["image_url"]}}
                ]
            }
        ],
        "max_tokens": 512
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(target_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Parse Response
            text_out = "No response"
            metadata = None
            
            if "choices" in data:
                text_out = data["choices"][0]["message"]["content"]
            elif "response" in data:
                text_out = data["response"]
            
            # Metadata handling for Grounding
            if mode == "GROUNDING":
                metadata = data 
                if not text_out or text_out == "No response":
                    text_out = "Here are the objects I located."

            return {"response_text": text_out, "response_metadata": metadata}

        except Exception as e:
            logger.error(f"Error calling Model ({target_url}): {e}")
            return {"response_text": "Error processing your request."}

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

# --- 7. Graph ---
workflow = StateGraph(ChatState)
workflow.add_node("determine_mode", determine_mode_node)
workflow.add_node("classify_vqa", classify_vqa_node)
workflow.add_node("execute_model", execute_model_node)
workflow.add_node("memory", memory_node)

workflow.set_entry_point("determine_mode")

def route_logic(state):
    # Only route to VQA Classifier if mode is VQA
    # SAR, FCC, CAPTIONING, GROUNDING all go straight to execute
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