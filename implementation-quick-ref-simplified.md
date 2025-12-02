# Implementation Quick Reference: Simplified (No Deep Research)

## Critical LangGraph Patterns

### Pattern 1: Simple Conditional Router (No Parallel Execution)

**What You Want**: Route to one service based on intent or mode.

**How LangGraph Does It**:
```python
def router(state: AgentState) -> str:
    if state["mode"] == "auto":
        # Classify intent
        return auto_router_func(state)
    else:
        # Explicit mode
        return f"call_{state['mode']}"

workflow.add_conditional_edges(START, router, {
    "call_grounding": "call_grounding",
    "call_vqa": "call_vqa",
    "call_captioning": "call_captioning",
})
```

**Key Point**: Always return a **single string** (not a list). LangGraph routes to one node and ends. No fan-in/fan-out needed.

---

### Pattern 2: State Merging (Node Return Dict)

**What You Want**: Each node returns a dict that updates state.

**Correct**:
```python
def vqa_node(state: AgentState) -> dict:
    result = modal_client.call_vqa(...)
    return {"vqa_result": result}  # Dict → merged into state
```

**GOTCHA**: Never return the full state! Only return the keys you changed.

---

### Pattern 3: Direct End After Service

**What You Want**: After a service completes, end the workflow.

**Correct**:
```python
workflow.add_edge("call_vqa", END)
workflow.add_edge("call_grounding", END)
workflow.add_edge("call_captioning", END)
```

**GOTCHA**: This is simple! No complex post-service routing needed.

---

### Pattern 4: Checkpoint Config Scoping

**What You Want**: Each session has isolated state (multi-tenant safety).

**Correct**:
```python
config = {"configurable": {"thread_id": session_id}}
result = app.invoke(inputs, config=config)

# Next invoke with same thread_id retrieves previous state
next_result = app.invoke(new_inputs, config=config)
```

**GOTCHA**: `thread_id` is the isolation key. Different threads never interfere.

---

## FastAPI Integration Patterns

### Pattern 1: Async HTTP Calls from Sync Node

**Problem**: LangGraph nodes are sync functions, but you want to call HTTP.

**Solution**: Use `requests` library (sync HTTP) in nodes:
```python
def vqa_node(state: AgentState) -> dict:
    # Sync HTTP call
    response = requests.post(modal_url, json={...}, timeout=120)
    return {"vqa_result": response.json()}
```

FastAPI endpoint is async:
```python
@app.post("/chat")
async def chat(req: ChatRequest):
    result = await asyncio.to_thread(app.invoke, inputs, config)
    return result
```

**GOTCHA**: Don't use `async/await` inside LangGraph nodes. Use `requests` for HTTP.

---

### Pattern 2: Session ID Generation

**Frontend sends**:
```python
{
  "session_id": "user-123-abc",  # Optional; if missing, generate
  "image_url": "...",
  "query": "What is this?",
  "mode": "auto"
}
```

**Backend**:
```python
if not req.session_id:
    req.session_id = str(uuid.uuid4())

# Store in localStorage on frontend
localStorage.setItem("sessionId", req.session_id)
```

---

## Modal Service Deployment

### Pattern 1: Correct Endpoint Signature

**Your Modal service must return JSON**:

```python
# Modal: grounding.py
@stub.function(...)
@web_endpoint()
def grounding_service(image_url: str, query: str):
    # Your inference logic
    bboxes = [{"x": 10, "y": 20, "w": 100, "h": 150, "label": "car", "confidence": 0.95}]
    return {"bboxes": bboxes}  # JSON-serializable dict
```

**LangGraph node receives**:
```python
result = modal_client.call_grounding(...)
# result = {"bboxes": [...]}
state["grounding_result"] = result
```

**GOTCHA**: Modal web_endpoint must return JSON. Always convert to primitives.

---

### Pattern 2: Timeout Handling

**Modal services can take 30-120s**. Set appropriate timeouts:

```python
requests.post(url, json=payload, timeout=120)  # 2 min max
```

**GOTCHA**: HTTP requests timeout at ~60s by default. Set explicit timeout.

---

## Auto Router Intent Classification

### Best Prompts

**For VQA vs Grounding vs Captioning**:
```
"Classify: '{query}'\n"
"Answer ONLY ONE word:\n"
"- 'LOCATE' if user wants bounding boxes/spatial info\n"
"- 'QA' if user asks a question\n"
"- 'DESCRIBE' if user wants general description\n"
"Response:"
```

**Examples**:
- "Find the car" → LOCATE
- "Where is the cat?" → LOCATE
- "What color is the car?" → QA
- "Describe this image" → QA
- (no query) → DESCRIBE/CAPTION

**GOTCHA**: Users might be ambiguous. Default to QA in tie-breaker.

---

## Error Handling & Resilience

### Retry Strategy with Tenacity

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8)
)
def call_vqa_node(state):
    try:
        result = modal_client.call_vqa(...)
        return {"vqa_result": result}
    except Exception as e:
        state["execution_log"].append(f"Attempt failed: {e}")
        raise  # Re-raise for retry
```

**Backoff sequence**: 1s, 2s, 4s (max 8s between attempts)

**GOTCHA**: Retries happen **at the HTTP layer**. If Modal is down, all 3 retries fail fast.

---

### Simple Error Response

```python
def vqa_node(state):
    try:
        result = modal_client.call_vqa(...)
        state["execution_log"].append("VQA service: Success")
        return {"vqa_result": result}
    except Exception as e:
        state["execution_log"].append(f"VQA service: Failed - {str(e)}")
        raise  # Let FastAPI catch and return error
```

**GOTCHA**: Don't swallow exceptions. Let them propagate to FastAPI's error handler.

---

## Frontend Result Display

### Pattern 1: Render Different Result Types

```typescript
if (type === "caption") {
    return <div className="result-box">📸 {content}</div>;
}
if (type === "answer") {
    return <div className="result-box">❓ {content}</div>;
}
if (type === "boxes") {
    return (
        <div className="result-box">
            📍 Objects:
            {content.bboxes?.map((bbox) => (
                <div key={bbox.label}>{bbox.label}</div>
            ))}
        </div>
    );
}
```

**GOTCHA**: Check content type before rendering. Use optional chaining.

---

## Deployment Checklist

### Pre-Production

- [ ] All service endpoints tested independently (curl/Postman)
- [ ] Modal services logged and monitored
- [ ] LangGraph graph visualized and tested with mock data
- [ ] FastAPI endpoints tested with sample requests
- [ ] Frontend built and tested locally
- [ ] Session persistence tested (DB checkpoint works)
- [ ] Retry logic tested (simulate service failure)
- [ ] Error messages user-friendly (no stack traces)

### Production

- [ ] Docker image built and runs
- [ ] PostgreSQL initialized
- [ ] Environment variables set (secrets in vault)
- [ ] API key auth enabled
- [ ] Rate limiting configured
- [ ] LangSmith monitoring enabled
- [ ] Logs centralized
- [ ] Alerts set up (Modal service down, high latency)
- [ ] Frontend deployed (CDN, browser caching)
- [ ] HTTPS enabled
- [ ] CORS properly configured

---

## Performance Optimization Tips

### Single Service (Target: <5s)

1. Reuse Modal model instance (no reload per request)
2. Model quantization (int8, fp16) if available
3. Image preprocessing: resize to model's expected size before sending
4. Connection pooling: reuse HTTP session via modal_client

---

## Monitoring with LangSmith

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_PROJECT"] = "multimodal-chatbot"

# LangGraph automatically logs all steps
result = app.invoke(inputs, config=config)
# Traces visible in LangSmith dashboard
```

**What to monitor**:
- Route decisions (did auto-router pick the right service?)
- Service latencies (Modal response times)
- Error rates per service
- Session count (concurrent users)

---

## Common Mistakes to Avoid

1. **Forgetting thread_id in config** → Sessions collide
2. **Returning full state from node** → State merge broken
3. **Not sanitizing frontend content** → XSS vulnerability
4. **Hardcoding Modal URLs** → Breaks when migrating services
5. **No retry logic** → Single transient failure = total failure
6. **Sync HTTP in async function** → Blocking event loop (use `asyncio.to_thread`)
7. **No session persistence** → User refreshes page, loses history
8. **Trying to parallelize** → Just route to one service per invoke
9. **LLM model too expensive** → Use gpt-4o-mini for router
10. **Returning list from router** → Use only strings for routing

---

## Simplified Integration Test Template

```python
import pytest
from backend.orchestrator import app

def test_auto_mode_text_only():
    """Image only -> should route to captioning"""
    inputs = {
        "session_id": "test-1",
        "image_url": "mock://image.jpg",
        "user_query": None,
        "mode": "auto",
        "messages": [],
        "session_context": {},
        "caption_result": None,
        "vqa_result": None,
        "grounding_result": None,
        "execution_log": [],
    }
    result = app.invoke(inputs, config={"configurable": {"thread_id": "test-1"}})
    assert result["caption_result"] is not None

def test_explicit_vqa_mode():
    """Explicit VQA mode should call VQA service"""
    inputs = {
        "session_id": "test-2",
        "image_url": "mock://image.jpg",
        "user_query": "What is this?",
        "mode": "vqa",
        "messages": [],
        "session_context": {},
        "caption_result": None,
        "vqa_result": None,
        "grounding_result": None,
        "execution_log": [],
    }
    result = app.invoke(inputs, config={"configurable": {"thread_id": "test-2"}})
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

---

## Final Reminders (Simplified)

1. **One service per request** — No parallel execution, no compiler node
2. **TypedDict for LangGraph**, Pydantic for FastAPI APIs
3. **Conditional edges** for routing, not if/else in nodes
4. **Return dict from nodes**, not full state
5. **Checkpoint config** for session isolation
6. **LLM-based router** for robustness
7. **Session context** for multi-turn reasoning
8. **Retry logic** for resilience
9. **Monitor everything** via LangSmith + structured logs
10. **Keep it simple** — Route → Service → End

---

## Key Differences from Original Design

| Aspect | Original | Simplified |
|--------|----------|-----------|
| **Modes** | Auto, Single, Deep Research | Auto, Single |
| **Graph Flow** | Conditional routing + optional compiler | Simple linear: Route → Service → End |
| **Compilation** | Parallel 3 services → Markdown synthesis | Single service per request |
| **Response Time** | Deep research: ~8-20s | All modes: <5s |
| **UI Complexity** | Markdown renderer, citations | Simple result display (caption/answer/boxes) |
| **Graph Complexity** | 4 nodes (3 services + compiler) | 3 nodes (3 services only) |
| **Lines of Code** | ~500 backend | ~250 backend |
| **Testing** | Parallel execution edge cases | Simple sequential tests |
