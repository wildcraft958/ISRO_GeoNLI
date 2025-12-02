# Cursor Prompt: Multimodal Chatbot Orchestrator (Simplified - No Deep Research)

## System Context

You are building a **production-grade multimodal chatbot** that orchestrates three specialized Vision-Language Model (VLM) services: **Grounding** (bounding box detection), **VQA** (visual question answering), and **Captioning** (image description). The system uses **LangGraph** for stateful workflow management, **FastAPI** for the orchestrator backend, **Modal** for VLM service deployment, and **React** for the frontend.

### Core Requirements

1. **Multi-Mode Routing**:
   - **Auto Mode**: Intent-driven routing (text-only → captioning; "find/locate" keywords → grounding; "what/how/why" → VQA)
   - **Single-Service Modes**: Direct invocation of one service (grounding, VQA, or captioning)

2. **Session Management**: Persistent conversation state across turns via LangGraph checkpoints (PostgreSQL in production)

3. **Modal Deployment**: All VLM services are HTTP endpoints deployed on Modal (you provide URLs)

4. **Performance**: Sub-5s single-service response

5. **Production Quality**: Error handling, retry logic, monitoring, logging, type safety

---

## Architecture Overview

```
Frontend (React)
    ↓ HTTP/WebSocket
FastAPI Backend (Orchestrator)
    ↓ invoke() with config
LangGraph Workflow (AgentState + Conditional Edges)
    ↓ HTTP requests
Modal Services (Grounding/VQA/Captioning endpoints)
```

**Key Design Principles**:
- Stateless service nodes (no side effects beyond HTTP calls)
- Conditional edges for routing (not if/else in nodes)
- Checkpoint-based session recovery
- Minimal branching: Auto router OR explicit mode
- Type-safe: Use TypedDict for AgentState, Pydantic for API schemas

---

## Phase 1: LangGraph Orchestrator (Core Logic)

### Task 1.1: Define AgentState

**File**: `backend/models/schema.py`

Create a TypedDict representing the full agent state:

```python
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
```

Also create Pydantic models for FastAPI:

```python
class ChatRequest(BaseModel):
    session_id: str
    image_url: str
    query: Optional[str] = None
    mode: Literal["auto", "grounding", "vqa", "captioning"] = "auto"

class ChatResponse(BaseModel):
    session_id: str
    response_type: str  # "caption", "answer", "boxes"
    content: Any
    execution_log: List[str]
```

**Requirements**:
- Use TypedDict for LangGraph state (not Pydantic; LangGraph prefers it)
- Separate Pydantic models for HTTP schemas
- All fields optional except session_id and mode (for flexibility)

---

### Task 1.2: Modal Service Wrapper

**File**: `backend/services/modal_client.py`

Create a client to call your three Modal-deployed services:

```python
import requests
import os
from typing import Optional

class ModalServiceClient:
    def __init__(self):
        self.base_url = os.getenv("MODAL_BASE_URL", "https://default.modal.run")
        self.timeout = 120
    
    def call_grounding(self, image_url: str, query: str) -> dict:
        """Call grounding service. Returns {bboxes: [...]}"""
        try:
            response = requests.post(
                f"{self.base_url}/grounding",
                json={"image_url": image_url, "query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Grounding service failed: {e}")
    
    def call_vqa(self, image_url: str, query: str) -> dict:
        """Call VQA service. Returns {answer: str, confidence: float}"""
        try:
            response = requests.post(
                f"{self.base_url}/vqa",
                json={"image_url": image_url, "query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"VQA service failed: {e}")
    
    def call_captioning(self, image_url: str) -> dict:
        """Call captioning service. Returns {caption: str}"""
        try:
            response = requests.post(
                f"{self.base_url}/captioning",
                json={"image_url": image_url},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Captioning service failed: {e}")
```

**Requirements**:
- Exponential backoff retry (1s, 2s, 4s, 8s) for transient failures
- Timeout: 120s per service
- Structured error messages
- Return JSON directly (state will be updated by node)

---

### Task 1.3: Auto Router Logic

**File**: `backend/orchestrator.py` (in a function `auto_router_func`)

Implement the classification logic that decides VQA vs Grounding vs Captioning:

```python
from langchain_openai import ChatOpenAI

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
```

**Requirements**:
- Use GPT-4o-mini for classification (cheap, fast)
- Return single node name matching workflow edges
- Log all routing decisions to `state["execution_log"]`
- Simple, direct routing: no parallel execution

---

### Task 1.4: Service Nodes (with Retries)

**File**: `backend/orchestrator.py` (node functions)

Create nodes that call Modal services with error handling:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

modal_client = ModalServiceClient()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_grounding_node(state: AgentState) -> dict:
    """Grounding service node with retry"""
    try:
        result = modal_client.call_grounding(state["image_url"], state["user_query"])
        state["execution_log"].append("Grounding service: Success")
        return {"grounding_result": result}
    except Exception as e:
        state["execution_log"].append(f"Grounding service: Failed - {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_vqa_node(state: AgentState) -> dict:
    """VQA service node with retry"""
    try:
        result = modal_client.call_vqa(state["image_url"], state["user_query"])
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
```

**Requirements**:
- Each node is a pure function: `(state: AgentState) -> dict`
- Return dict updates state (LangGraph merges automatically)
- Use @retry decorator for resilience
- Log success/failure to execution_log
- Graceful error propagation (don't swallow exceptions)

---

### Task 1.5: Graph Construction

**File**: `backend/orchestrator.py` (main function)

Wire everything into a LangGraph workflow:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

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
    # Use PostgreSQL in production, MemorySaver for development
    checkpointer = PostgresSaver(connection_string=os.getenv("DATABASE_URL"))
    # For dev: checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)

# Instantiate globally
app = build_workflow()
```

**Requirements**:
- Use conditional_edges for routing logic
- All service nodes directly to END (no compilation step)
- Simple, linear workflow
- Compile with checkpointer for session persistence
- Use PostgresSaver in production (thread-safe)

---

## Phase 2: FastAPI Backend

### Task 2.1: Session Management

**File**: `backend/utils/checkpoint.py`

Create utilities for session retrieval and history:

```python
from typing import Optional, List
import json

class SessionManager:
    def __init__(self, graph_app):
        self.app = graph_app
    
    def get_session_history(self, session_id: str) -> List[dict]:
        """Retrieve all messages in session"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except:
            return []
    
    def update_session_context(self, session_id: str, key: str, value):
        """Update session_context dict (e.g., user prefs)"""
        config = {"configurable": {"thread_id": session_id}}
        state = self.app.get_state(config)
        context = state.values.get("session_context", {})
        context[key] = value
        # Note: Update via next invocation, not direct state manipulation
```

**Requirements**:
- Retrieve state via `app.get_state(config)`
- All session ops are config-scoped (thread_id)
- Don't mutate state directly; use next `invoke()` to update

---

### Task 2.2: Chat Endpoint

**File**: `backend/routes/chat.py`

Main endpoint for chatbot interaction:

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.models.schema import ChatRequest, ChatResponse
from backend.orchestrator import app
from backend.utils.checkpoint import SessionManager
import uuid

router = APIRouter()
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
        
        # Prepare inputs for LangGraph
        inputs = {
            "session_id": req.session_id,
            "image_url": req.image_url,
            "user_query": req.query,
            "mode": req.mode,
            "messages": session_manager.get_session_history(req.session_id),
            "session_context": {},
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
        
        # Update session history
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
    pass

@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve full session history"""
    history = session_manager.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}
```

**Requirements**:
- Initialize empty state on first invoke
- Use thread_id for session isolation
- Return appropriate response type (caption, answer, boxes)
- Handle all modes (auto, single services)
- Minimal complexity

---

### Task 2.3: Main FastAPI App

**File**: `backend/main.py`

Assemble everything:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.chat import router as chat_router
import os

app = FastAPI(title="Multimodal Chatbot Orchestrator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Requirements**:
- CORS enabled for frontend
- Health check endpoint
- Runnable via `python -m backend.main`

---

## Phase 3: Frontend (React + Result Display)

### Task 3.1: Chat Hook

**File**: `frontend/src/hooks/useChat.ts`

HTTP hook for sending requests and receiving responses:

```typescript
import { useState, useCallback } from "react";

export const useChat = () => {
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => 
    localStorage.getItem("sessionId") || `session-${Date.now()}`
  );

  const sendMessage = useCallback(
    async (
      imageUrl: string,
      query: string,
      mode: "auto" | "grounding" | "vqa" | "captioning"
    ) => {
      setLoading(true);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            image_url: imageUrl,
            query,
            mode,
          }),
        });

        const data = await response.json();

        // Add to local message history
        setMessages((prev) => [
          ...prev,
          {
            role: "user",
            content: query || "(Image uploaded)",
            source: "user",
          },
          {
            role: "assistant",
            type: data.response_type,
            content: data.content,
            executionLog: data.execution_log,
          },
        ]);

        // Persist session
        localStorage.setItem("sessionId", sessionId);

        return data;
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  return { messages, loading, sessionId, sendMessage };
};
```

**Requirements**:
- Persist sessionId in localStorage
- Handle async fetch with error boundary
- Return both response data and message history

---

### Task 3.2: Chat Interface Component

**File**: `frontend/src/components/ChatInterface.tsx`

Main UI component:

```typescript
import React, { useState } from "react";
import { useChat } from "../hooks/useChat";
import { ImageUploader } from "./ImageUploader";
import { ModeSelector } from "./ModeSelector";
import { ResultDisplay } from "./ResultDisplay";
import "./ChatInterface.css";

export const ChatInterface: React.FC = () => {
  const { messages, loading, sessionId, sendMessage } = useChat();
  const [imageUrl, setImageUrl] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [mode, setMode] = useState<"auto" | "grounding" | "vqa" | "captioning">("auto");

  const handleSend = async () => {
    if (!imageUrl) {
      alert("Please upload an image first");
      return;
    }
    await sendMessage(imageUrl, query, mode);
    setQuery("");
  };

  return (
    <div className="chat-container">
      <div className="sidebar">
        <div className="session-info">
          <p>Session: {sessionId.slice(0, 8)}...</p>
        </div>
        <ImageUploader onImageSelect={setImageUrl} />
        <ModeSelector mode={mode} onModeChange={setMode} />
      </div>

      <div className="main-panel">
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <p>{msg.content}</p>
              {msg.role === "assistant" && (
                <ResultDisplay type={msg.type} content={msg.content} />
              )}
              {msg.executionLog && (
                <details>
                  <summary>Execution Log ({msg.executionLog.length} steps)</summary>
                  <pre>{msg.executionLog.join("\n")}</pre>
                </details>
              )}
            </div>
          ))}
        </div>

        <div className="input-area">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about the image..."
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !imageUrl}>
            {loading ? "Processing..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
};
```

**Requirements**:
- Session display
- Image preview
- Mode selector
- Real-time message rendering
- Result display based on type
- Execution log collapsible details

---

### Task 3.3: Result Display Component

**File**: `frontend/src/components/ResultDisplay.tsx`

Render different result types (caption, answer, boxes):

```typescript
import React from "react";

interface ResultDisplayProps {
  type: "caption" | "answer" | "boxes";
  content: any;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  type,
  content,
}) => {
  if (type === "caption") {
    return (
      <div className="result-box caption-box">
        <h4>📸 Image Caption</h4>
        <p>{content}</p>
      </div>
    );
  }

  if (type === "answer") {
    return (
      <div className="result-box answer-box">
        <h4>❓ Answer</h4>
        <p>{content}</p>
      </div>
    );
  }

  if (type === "boxes") {
    return (
      <div className="result-box boxes-box">
        <h4>📍 Detected Objects</h4>
        {content.bboxes && content.bboxes.length > 0 ? (
          <ul>
            {content.bboxes.map((bbox: any, i: number) => (
              <li key={i}>
                <strong>{bbox.label}</strong> (confidence: {(bbox.confidence * 100).toFixed(1)}%)
                <br />
                <small>x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}</small>
              </li>
            ))}
          </ul>
        ) : (
          <p>No objects detected</p>
        )}
      </div>
    );
  }

  return null;
};
```

**Requirements**:
- Different visual styles for each response type
- JSON rendering for bounding boxes
- Simple, readable formatting
- Icons for visual distinction

---

## Phase 4: Testing & Deployment

### Task 4.1: End-to-End Test

**File**: `tests/test_e2e.py`

```python
import pytest
from backend.models.schema import AgentState
from backend.orchestrator import app

def test_auto_mode_text_only():
    """Image only -> should route to captioning"""
    state = {
        "session_id": "test-1",
        "image_url": "https://example.com/image.jpg",
        "user_query": None,
        "mode": "auto",
        "messages": [],
        "session_context": {},
        "caption_result": None,
        "vqa_result": None,
        "grounding_result": None,
        "execution_log": [],
    }
    result = app.invoke(state, config={"configurable": {"thread_id": "test-1"}})
    assert result["caption_result"] is not None

def test_explicit_vqa_mode():
    """Explicit VQA mode should call VQA service"""
    state = {
        "session_id": "test-2",
        "image_url": "https://example.com/image.jpg",
        "user_query": "What is this?",
        "mode": "vqa",
        "messages": [],
        "session_context": {},
        "caption_result": None,
        "vqa_result": None,
        "grounding_result": None,
        "execution_log": [],
    }
    result = app.invoke(state, config={"configurable": {"thread_id": "test-2"}})
    assert result["vqa_result"] is not None

def test_session_recovery():
    """Session checkpoint should persist"""
    config = {"configurable": {"thread_id": "test-3"}}
    # Invoke 1
    app.invoke({...}, config=config)
    # Invoke 2 (should retain history)
    result = app.invoke({...}, config=config)
    assert len(result["messages"]) >= 1
```

**Requirements**:
- Test all modes
- Test auto-routing logic
- Test session checkpointing
- Mock Modal endpoints or use test fixtures

---

### Task 4.2: Docker Deployment

**File**: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/dist/ frontend/dist/

ENV DATABASE_URL=postgresql://user:pass@localhost/db
ENV MODAL_BASE_URL=https://org--app.modal.run
ENV OPENAI_API_KEY=sk-...

EXPOSE 8000

CMD ["python", "-m", "backend.main"]
```

**File**: `docker-compose.yml`

```yaml
version: "3.8"

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@db:5432/chatbot
      MODAL_BASE_URL: https://org--app.modal.run
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: chatbot
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**Requirements**:
- Separate DB for checkpoint persistence
- Environment variables for secrets
- Multi-stage build for optimization

---

## Detailed Implementation Checklist

### LangGraph Orchestrator
- [ ] AgentState TypedDict defined (schema.py)
- [ ] ModalServiceClient created with retry logic
- [ ] Auto router classifies intent (VQA vs Grounding vs Captioning)
- [ ] Conditional router handles modes
- [ ] Service nodes call Modal endpoints with @retry
- [ ] Graph constructed with conditional_edges
- [ ] Simple linear flow: Router → Service → End
- [ ] Checkpointer configured (PostgreSQL)

### FastAPI Backend
- [ ] ChatRequest/ChatResponse Pydantic models
- [ ] /api/chat endpoint (POST)
- [ ] /api/session/{id}/history endpoint (GET)
- [ ] Session manager for state retrieval
- [ ] CORS configured
- [ ] Error handling & logging
- [ ] Health check endpoint

### Frontend
- [ ] useChat hook with localStorage persistence
- [ ] ChatInterface component (main UI)
- [ ] ImageUploader component
- [ ] ModeSelector component
- [ ] ResultDisplay component (caption, answer, boxes)
- [ ] Message history display
- [ ] Execution log collapsible details
- [ ] CSS styling (Tailwind or custom)

### Testing & Deployment
- [ ] Unit tests for routing logic
- [ ] E2E tests for all modes
- [ ] Session checkpoint tests
- [ ] Docker build & docker-compose
- [ ] LangSmith integration (monitoring)
- [ ] Rate limiting (FastAPI middleware)

---

## Environment Variables

```bash
# Backend
DATABASE_URL=postgresql://user:pass@localhost/chatbot
MODAL_BASE_URL=https://your-org--app-name.modal.run
OPENAI_API_KEY=sk-your-api-key
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## Success Criteria

- [ ] Auto mode correctly routes to VQA, Grounding, or Captioning
- [ ] Single-service mode works correctly (<5s end-to-end)
- [ ] Session history persists across browser refreshes
- [ ] Result display renders correctly (captions, answers, boxes)
- [ ] Modal service failures retry gracefully
- [ ] Execution logs visible for debugging
- [ ] Frontend responsive on mobile
- [ ] LangSmith traces all workflow steps
- [ ] No N+1 queries or memory leaks
