# Backend — DRISHTI Inference Server

This directory contains the FastAPI-based backend orchestrator for the DRISHTI system.

## Architecture

The backend implements a **LangGraph-based workflow** that orchestrates multimodal queries across specialized inference engines:

```
User Query → Modality Detection → Task Router → Specialized Model → Response
                    ↓                  ↓
              IR2RGB Preprocess   SAM3 (Numeric)
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Orchestrator** | `app/orchestrator.py` | LangGraph workflow with state management |
| **Modality Router** | `app/services/modality_router.py` | Classifies RGB/SAR/IR/FCC imagery |
| **IR2RGB Service** | `app/services/ir2rgb_service.py` | FCC to RGB reconstruction |
| **Modal Client** | `app/services/modal_client.py` | Interface to GPU services |

### Modal Services (GPU)

These run on [Modal](https://modal.com) for GPU-accelerated inference:

| Service | File | Purpose |
|---------|------|---------|
| `sam3_agent.py` | Pyramidal SAM3 | Object counting & area estimation |
| `vllm_modal_deploy.py` | vLLM Server | Qwen3-VL inference |
| `captioning.py` | Captioning Model | Specialized RS captions |
| `vqa.py` | VQA Model | Visual question answering |
| `grounding.py` | Grounding Model | Object localization |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestrator/chat` | POST | Main multimodal chat endpoint |
| `/orchestrator/ir2rgb` | POST | Standalone IR→RGB conversion |
| `/orchestrator/modality/status` | GET | Service health check |
| `/health` | GET | Backend health check |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
MONGO_URL=mongodb://localhost:27017
OPENAI_API_KEY=...          # For task router
MODAL_BASE_URL=https://...  # Modal deployment URL
```

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --port 8000
```

## Deploying Modal Services

```bash
cd modal_services
modal deploy sam3_agent.py
modal deploy vllm_modal_deploy.py
```
