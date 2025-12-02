# Multimodal Chatbot Orchestrator: Simplified Blueprint & Architecture

## Executive Summary

A production-grade **LangGraph-based multimodal chatbot** that routes queries across three specialized Vision-Language Model (VLM) services: **Grounding** (bounding box detection), **VQA** (visual question answering), and **Captioning** (image description). Features include:

- **Auto Mode**: Intent-driven routing (text-only → caption; "find/locate" → grounding; "what/how" → VQA)
- **Single Service Modes**: Direct invocation of one service (grounding, VQA, or captioning)
- **Session Memory**: Persistent conversation state across turns via LangGraph checkpoints
- **Modal Deployment**: All VLM services exposed as lightweight HTTP endpoints on Modal
- **FastAPI Backend**: RESTful API with async support and session-aware routing
- **Production UI**: Clean, responsive interface with streaming results and execution transparency

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React/Vue)                      │
│  (Image upload, text query input, mode selector, result display)│
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│              FastAPI Backend (Orchestrator)                      │
│  - Session Management (Redis/In-Memory)                         │
│  - Request validation & routing                                 │
│  - WebSocket streaming for long tasks                           │
└────────────────────┬─────────────────────────────────────────────┘
                     │ invoke(inputs, config)
┌────────────────────▼─────────────────────────────────────────────┐
│           LangGraph Workflow (Stateful Agent)                    │
│                                                                  │
│  1. Auto Router Node  (Intent classification)                   │
│  2. Conditional Edges (Mode → Service selection)                │
│  3. Service Nodes    (Call Modal endpoints)                     │
│  4. Memory Checkpoints (Session state persistence)              │
└────────────────────┬─────────────────────────────────────────────┘
                     │ HTTP requests
┌────────────────────▼─────────────────────────────────────────────┐
│          Modal Deployed Services (Open-Source VLMs)             │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │  Grounding      │  │  VQA            │  │  Captioning      │ │
│  │  (Detect BBoxes)│  │  (Answer Qs)    │  │  (Describe Img)  │ │
│  │  Fine-tuned VLM │  │  Fine-tuned VLM │  │  Fine-tuned VLM  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend (React/Vue + Result Display)

**Key Features:**
- Image upload with preview
- Multiline text input (query/description)
- Mode selector (Auto/Grounding/VQA/Captioning)
- Session ID persistence (localStorage)
- Real-time streaming of model outputs
- Execution logs with latency metrics
- Service response display (text, JSON, or formatted boxes)

**Example UI Flow (VQA Mode):**
```
User uploads image + enters question "What is in this?"
        ↓
[Mode: VQA selected]
[Analyzing...] (spinner with elapsed time)
        ↓
Result displayed:
  Service: VQA
  Answer: "A person sitting on a bench in a park"
  Latency: 4.2s
  
[Execution Log]
  - Router: Explicit mode VQA selected
  - VQA Service: Request sent
  - VQA Service: Response received (4.2s)
```

---

### 2. FastAPI Backend (Orchestrator)

**Responsibilities:**
- Validate requests & session IDs
- Invoke LangGraph workflows
- Stream responses via WebSocket for long tasks
- Cache session state
- Error handling & retry logic

**Core Endpoints:**
- `POST /api/chat` — Invoke graph
- `GET /api/session/{session_id}` — Retrieve session history
- `WS /ws/{session_id}` — Streaming updates

---

### 3. LangGraph State Management

**AgentState TypedDict:**
```
{
  session_id: str,                         # For checkpoint retrieval
  messages: List[Dict],                    # Chat history [{role, content, source, timestamp}]
  image_url: str,                          # Current image
  user_query: Optional[str],               # Current text input
  mode: "auto" | "grounding" | "vqa" | "captioning",
  
  # Service Output (only one populated per invoke)
  caption_result: Optional[str],           # "A photo of..."
  vqa_result: Optional[str],               # "Yes, the object is..."
  grounding_result: Optional[Dict],        # {bboxes: [{x, y, w, h, label}]}
  
  # Metadata
  execution_log: List[str],                # Debug: ["Router selected VQA", ...]
  session_context: Dict,                   # Persisted: user prefs, execution stats
}
```

---

### 4. Routing Logic (Auto Mode)

**Decision Tree:**
```
Query received
  ├─ Is user_query empty?
  │   └─ Yes → CAPTIONING (user only uploaded image)
  ├─ Does query contain keywords: "find", "where", "locate", "object", "bbox"?
  │   └─ Yes → GROUNDING (user wants spatial information)
  ├─ Does query contain keywords: "what", "how", "why", "explain", "describe"?
  │   └─ Yes → VQA (user wants semantic understanding)
  └─ Default → VQA (open-ended Q&A)
```

**Implementation:** LLM-based classifier (GPT-4o-mini) for robustness, with local rule-based fallback.

---

### 5. Session Memory & Checkpointing

**Checkpoint Strategy:**
- **Storage**: LangGraph MemorySaver (development) or PostgreSQL (production)
- **Granularity**: After each service call (enables resume on failure)
- **Context Retention**: `session_context` dict includes:
  - `conversation_history`: Last 5 turns (for multi-turn reasoning)
  - `user_preferences`: Auto vs. explicit mode history
  - `execution_stats`: Latency, service health

**Retrieval:**
```python
config = {"configurable": {"thread_id": session_id}}
result = app.invoke(inputs, config=config)
```

---

### 6. Modal Service Deployment

**Each service is an HTTP endpoint on Modal:**

```python
# grounding.py (Modal)
@stub.function(image=image, gpu="A100")
def grounding_service(image_url: str, query: str):
    # Load your fine-tuned grounding model
    # Return: {bboxes: [{x, y, w, h, label, confidence}]}
    pass

@stub.function()
@web_endpoint()
def grounding_endpoint(image_url: str, query: str):
    return grounding_service.remote(image_url, query)
```

**Exposed as:** `https://org--app.modal.run/grounding`

---

### 7. Error Handling & Resilience

**Retry Strategy:**
- Exponential backoff (1s, 2s, 4s, 8s max)
- Max 3 retries per service call
- Timeout: 120s per service

**Graceful Error Response:**
- If service fails after retries: return error message to user
- Log failure with full traceback for debugging
- Suggest retry or alternative mode

---

## Data Flow Examples

### Example 1: Auto Mode (Text-Heavy Query)

```
User Input:
  Image: landscape.jpg
  Query: "What trees are in this image?"
  Mode: auto

Flow:
1. Auto Router classifies: "what" keyword → VQA intent
2. Route to call_vqa node
3. VQA service called: → "Oak and Pine trees"
4. Return to user: {"type": "answer", "content": "Oak and Pine trees"}

Session Updated:
  messages: [..., {role: "user", content: "What trees...", source: "text", timestamp},
                  {role: "assistant", content: "Oak and Pine...", source: "vqa", timestamp}]
```

### Example 2: Explicit Grounding Mode

```
User Input:
  Image: car.jpg
  Query: "Find the headlight"
  Mode: grounding

Flow:
1. Router identifies explicit mode: grounding
2. Route to call_grounding node directly
3. Grounding service called: → {bboxes: [{x: 0.2, y: 0.15, w: 0.15, h: 0.08, label: "headlight"}]}
4. Return to user with bbox visualization

Session Updated:
  messages: [..., {role: "user", query: "Find the headlight", source: "user_input"},
             {role: "assistant", type: "grounding", 
              bboxes: [{...}]}]
```

---

## Production Checklist

- [ ] All service nodes have proper error handling & logging
- [ ] LangGraph graph is tested with all edge cases (empty query, bad image, service timeouts)
- [ ] Session checkpoints persisted to PostgreSQL (not in-memory)
- [ ] FastAPI endpoints secured with API key / session validation
- [ ] WebSocket streaming tested for long-running queries
- [ ] Modal services scaled with appropriate GPU allocation
- [ ] Monitoring & observability via LangSmith
- [ ] Rate limiting & quota enforcement per session
- [ ] Frontend responsive on mobile
- [ ] Tests cover: single-service modes, auto-routing, session recovery

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + TypeScript | UI with real-time streaming |
| **Backend** | FastAPI + Uvicorn | RESTful API & WebSocket |
| **Orchestration** | LangGraph | Stateful agent workflows |
| **LLM (Router)** | GPT-4o-mini (OpenAI API) | Intent classification |
| **Session Storage** | PostgreSQL + Redis | Checkpoint persistence & caching |
| **VLM Services** | Modal | Scalable inference for grounding/VQA/captioning |
| **Monitoring** | LangSmith | Agentic workflow observability |
| **Deployment** | Docker + Kubernetes (optional) | Production orchestration |

---

## Files & Project Structure

```
multimodal-chatbot/
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── orchestrator.py            # LangGraph workflow construction
│   ├── routes/
│   │   ├── chat.py                # Chat endpoint
│   │   └── session.py             # Session retrieval
│   ├── services/
│   │   └── modal_client.py        # Modal service wrappers
│   ├── models/
│   │   └── schema.py              # Pydantic models + AgentState
│   └── utils/
│       ├── checkpoint.py          # Checkpoint logic
│       └── logging.py             # Structured logging
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx   # Main UI
│   │   │   ├── ImageUploader.tsx   # Image input
│   │   │   ├── ModeSelector.tsx    # Mode toggle
│   │   │   └── ResultDisplay.tsx   # Result rendering
│   │   ├── hooks/
│   │   │   └── useChat.ts          # WebSocket/API hooks
│   │   └── App.tsx
│   └── package.json
├── modal_services/
│   ├── grounding.py               # Grounding model endpoint
│   ├── vqa.py                      # VQA model endpoint
│   └── captioning.py               # Captioning model endpoint
├── docker-compose.yml
├── README.md
└── tests/
    ├── test_orchestrator.py
    ├── test_routes.py
    └── test_e2e.py
```

---

## Next Steps

1. **Phase 1**: Build LangGraph workflow + Modal service endpoints
2. **Phase 2**: Implement FastAPI backend with session management
3. **Phase 3**: Build React frontend with result rendering
4. **Phase 4**: Integration testing (end-to-end)
5. **Phase 5**: Performance tuning & scaling
6. **Phase 6**: Production deployment (Kubernetes + monitoring)
