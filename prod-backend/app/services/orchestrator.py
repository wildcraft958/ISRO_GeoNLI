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
    # Inputs
    chat_id: str
    user_id: str
    query_text: str
    requested_mode: str  # AUTO, VQA, GROUNDING, CAPTIONING
    
    # Context
    image_url: str
    image_type: str
    summary_context: str 
    
    # Routing Decisions
    final_mode: str      # VQA, CAPTIONING, GROUNDING
    vqa_subtype: str     # BINARY, NUMERICAL, GENERAL (Only if mode is VQA)
    
    # Outputs
    response_text: str
    response_metadata: Optional[Any]


# --- 2. Helper: Generic LLM Classifier ---
async def call_generic_classifier(system_prompt: str, user_query: str) -> str:
    """
    Calls the Generic LLM to classify intent.
    Expects the LLM to return a keyword (Regex extraction).
    """
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1 # Deterministic
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(settings.GENERIC_LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract content (Standard OpenAI format assumed)
            # Adjust if your generic LLM returns differently
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not raw_content:
                raw_content = data.get("response", "") # Fallback

            return raw_content.upper()
        except Exception as e:
            logger.error(f"Generic LLM Classifier Failed: {e}")
            return "ERROR"

def parse_mode_from_text(text: str, options: list[str], default: str) -> str:
    """
    Robustly finds one of the options in the LLM response using Regex.
    """
    for option in options:
        # Looks for whole word match, ignoring XML tags
        if re.search(rf"\b{option}\b", text):
            return option
    return default


# --- 3. Node: Mode Determination (Step 1) ---
async def determine_mode_node(state: ChatState):
    """
    Decides between VQA, CAPTIONING, GROUNDING.
    """
    req_mode = state["requested_mode"]

    if req_mode != "AUTO":
        return {"final_mode": req_mode}

    # System Prompt for the Router
    sys_prompt = (
        "You are an AI router. Classify the user query into one of these 3 modes:\n"
        "1. GROUNDING (if asking to find, locate, detect coordinates, or boxes)\n"
        "2. CAPTIONING (if asking to describe the whole image)\n"
        "3. VQA (if asking a specific question about content)\n"
        "Respond ONLY with one word: VQA, CAPTIONING, or GROUNDING."
    )

    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    
    # robust parsing
    detected_mode = parse_mode_from_text(
        llm_response, 
        ["GROUNDING", "CAPTIONING", "VQA"], 
        default="VQA"
    )
    
    logger.info(f"Auto-Router decided: {detected_mode}")
    return {"final_mode": detected_mode}


# --- 4. Node: VQA Sub-Classification (Step 2) ---
async def classify_vqa_node(state: ChatState):
    """
    If mode is VQA, decide: BINARY, NUMERICAL, or GENERAL.
    """
    sys_prompt = (
        "You are a VQA classifier. Analyze the question type:\n"
        "1. BINARY (Yes/No questions, e.g., 'Is there a ship?')\n"
        "2. NUMERICAL (Counting questions, e.g., 'How many cars?')\n"
        "3. GENERAL (Open-ended questions, e.g., 'What color is the car?')\n"
        "Respond ONLY with one word: BINARY, NUMERICAL, or GENERAL."
    )
    
    llm_response = await call_generic_classifier(sys_prompt, state["query_text"])
    
    subtype = parse_mode_from_text(
        llm_response, 
        ["BINARY", "NUMERICAL", "GENERAL"], 
        default="GENERAL"
    )
    
    logger.info(f"VQA Sub-classifier decided: {subtype}")
    return {"vqa_subtype": subtype}


# --- 5. Node: Model Execution (Step 3) ---
async def execute_model_node(state: ChatState):
    """
    Selects 1 of 5 URLs and calls it.
    """
    mode = state["final_mode"]
    subtype = state.get("vqa_subtype", "GENERAL")
    
    # 1. Select URL
    target_url = settings.VQA_GENERAL_URL
    
    if mode == "GROUNDING":
        target_url = settings.GROUNDING_URL
    elif mode == "CAPTIONING":
        target_url = settings.CAPTION_URL
    elif mode == "VQA":
        if subtype == "BINARY":
            target_url = settings.VQA_BINARY_URL
        elif subtype == "NUMERICAL":
            target_url = settings.VQA_NUMERICAL_URL
        else:
            target_url = settings.VQA_GENERAL_URL

    # 2. Prepare Context
    # "Before giving input to modal url add image_context and context_summary"
    full_prompt = state["query_text"]
    context_header = ""
    
    if state["summary_context"]:
        context_header += f"Chat History: {state['summary_context']}\n"
    
    # Add Image Type Context if relevant to the model
    context_header += f"Image Type: {state['image_type']}\n"
    
    full_prompt = f"{context_header}\nQuestion: {state['query_text']}"

    # 3. Construct Payload
    # Assuming all 5 specialist models accept a standard OpenAI Vision-like format
    # If they differ, use `if mode == ...` to adjust payload structure
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
        "max_tokens": 512,
        "temperature": 0.1
    }

    # 4. Call API
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(target_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # 5. Extract Text Message
            # Adjust path based on your specialist model response structure
            text_out = "No response"
            metadata = None
            
            if "choices" in data:
                text_out = data["choices"][0]["message"]["content"]
            elif "response" in data:
                text_out = data["response"]
            
            # If Grounding, we expect metadata
            if mode == "GROUNDING":
                # Assuming the model returns raw boxes/polygons in a specific key
                metadata = data # Store full response as metadata
                # Ensure we have a text message for the chat bubble
                if not text_out or text_out == "No response":
                    text_out = "Here are the objects I located."

            return {"response_text": text_out, "response_metadata": metadata}

        except Exception as e:
            logger.error(f"Error calling Specialist Model ({target_url}): {e}")
            return {"response_text": "Error processing your request with the AI model."}

# --- NEW HELPER: Context Summarizer ---
async def summarize_conversation(current_summary: str, last_query: str, last_response: str) -> str:
    """
    Calls the Generic LLM to condense the history + new turn into a concise summary.
    """
    # 1. Construct the prompt
    # If no previous summary, just summarize the current turn
    if not current_summary:
        prompt_content = (
            f"Summarize this interaction concisely for an AI context memory:\n"
            f"User asked: {last_query}\n"
            f"AI answered: {last_response}"
        )
    else:
        prompt_content = (
            f"Update the following summary with the new interaction. "
            f"Keep the summary concise (max 150 words), retaining key visual details found so far.\n\n"
            f"Current Summary: {current_summary}\n"
            f"New User Query: {last_query}\n"
            f"New AI Response: {last_response}\n\n"
            f"New Summary:"
        )

    # 2. Call the LLM
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a context summarizer. You condense conversation history into brief, factual notes."
            },
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.3, # Slightly higher than 0 to allow decent phrasing
        "max_tokens": 200   # Limit output size
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(settings.GENERIC_LLM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract content (Standard OpenAI format)
            new_summary = ""
            if "choices" in data:
                new_summary = data["choices"][0]["message"]["content"]
            elif "response" in data:
                new_summary = data["response"]
                
            return new_summary.strip()
            
        except Exception as e:
            logger.error(f"Summarization Failed: {e}")
            # Fallback: If LLM fails, append logically to avoid data loss
            return f"{current_summary}\nQ: {last_query} A: {last_response}"[-2000:]


# --- UPDATED NODE: Memory & Summarization (Step 4) ---
async def memory_node(state: ChatState):
    """
    Saves to DB and updates Summary buffer via LLM.
    """
    db = SessionLocal()
    try:
        # 1. Save the raw interaction to DB (Permanent Log)
        new_query = Query(
            chat_id=state["chat_id"],
            user_id=state["user_id"],
            query_text=state["query_text"],
            query_mode=f"{state['final_mode']}::{state.get('vqa_subtype', '')}",
            response_text=state["response_text"],
            response_metadata=state.get("response_metadata")
        )
        db.add(new_query)
        db.commit() # Commit early to ensure log is saved even if summary fails
        
        # 2. Generate New Summary using LLM (Async)
        old_summary = state.get("summary_context", "") or ""
        
        # This call is now intelligent. It reads the old summary and rewrites it.
        updated_summary = await summarize_conversation(
            current_summary=old_summary, 
            last_query=state["query_text"], 
            last_response=state["response_text"]
        )
        
        # 3. Update the Session Row with the new concise summary
        session = db.query(ChatSession).filter(ChatSession.chat_id == state["chat_id"]).first()
        if session:
            session.summary_context = updated_summary
            db.commit() # Update the session context
            
        return {"summary_context": updated_summary}
        
    except Exception as e:
        logger.error(f"Memory node critical failure: {e}")
        db.rollback()
        return {} # Prevent graph crash
    finally:
        db.close()


# --- 7. Graph Construction ---
workflow = StateGraph(ChatState)

# Add Nodes
workflow.add_node("determine_mode", determine_mode_node)
workflow.add_node("classify_vqa", classify_vqa_node)
workflow.add_node("execute_model", execute_model_node)
workflow.add_node("memory", memory_node)

# Entry
workflow.set_entry_point("determine_mode")

# Conditional Edge 1: After Mode Determination
def route_after_mode(state):
    if state["final_mode"] == "VQA":
        return "classify_vqa"
    return "execute_model" # Go straight to Caption/Grounding model

workflow.add_conditional_edges(
    "determine_mode",
    route_after_mode,
    {
        "classify_vqa": "classify_vqa",
        "execute_model": "execute_model"
    }
)

# Edge 2: After VQA Classification -> Execute
workflow.add_edge("classify_vqa", "execute_model")

# Edge 3: Execute -> Memory -> End
workflow.add_edge("execute_model", "memory")
workflow.add_edge("memory", END)

chat_app = workflow.compile()