import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.orchestrator_schema import ChatRequest, ChatResponse
from app.orchestrator import app
from app.utils.session_manager import SessionManager

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])
session_manager = SessionManager(app)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint.
    Accepts: image_url, query (optional), mode, session_id
    Returns: response content + execution log
    """
    try:
        # Validate session
        if not req.session_id:
            req.session_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": req.session_id}}
        
        # Get existing session state if available
        existing_state = session_manager.get_session_state(req.session_id)
        
        # Prepare inputs for LangGraph
        inputs = {
            "session_id": req.session_id,
            "image_url": req.image_url,
            "user_query": req.query,
            "mode": req.mode,
            "messages": existing_state.get("messages", []) if existing_state else [],
            "session_context": existing_state.get("session_context", {}) if existing_state else {},
            "caption_result": None,
            "vqa_result": None,
            "grounding_result": None,
            "execution_log": [],
        }
        
        # Invoke workflow
        result = app.invoke(inputs, config=config)
        
        # Determine response type
        response_type = "unknown"
        content = None
        
        if req.mode == "captioning":
            response_type = "caption"
            content = result.get("caption_result", "")
        elif req.mode == "vqa":
            response_type = "answer"
            content = result.get("vqa_result", "")
        elif req.mode == "grounding":
            response_type = "boxes"
            content = result.get("grounding_result", {})
        else:  # auto mode
            if result.get("caption_result"):
                response_type = "caption"
                content = result["caption_result"]
            elif result.get("vqa_result"):
                response_type = "answer"
                content = result["vqa_result"]
            elif result.get("grounding_result"):
                response_type = "boxes"
                content = result["grounding_result"]
        
        # Update message history in background
        background_tasks.add_task(
            update_message_history,
            req.session_id,
            req.query,
            response_type,
            content
        )
        
        return ChatResponse(
            session_id=req.session_id,
            response_type=response_type,
            content=content,
            execution_log=result.get("execution_log", [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def update_message_history(session_id: str, query: str, response_type: str, content):
    """Background task to append to message history"""
    # This would be called in post-processing; for now, logged via LangGraph
    # The state is already updated in the workflow
    pass


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve full session history"""
    history = session_manager.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}

