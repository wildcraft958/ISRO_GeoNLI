# Backend — DRISHTI Inference Server

This directory contains the FastAPI-based backend orchestrator for the DRISHTI system, implementing the microservices-based architecture for multimodal satellite imagery analysis.

## Architecture Overview

The backend implements a **LangGraph-based workflow** that orchestrates multimodal queries across specialized inference engines:

```
User Query → Modality Detection → Task Router → Specialized Model → Response
                    ↓                  ↓
              IR2RGB Preprocess   SAM3 (Numeric VQA)
```

### Dual-Routing Strategy

| Router | Implementation | Function |
|--------|----------------|----------|
| **Visual Routing** | ResNet-18 classifier | Categorizes imagery (RGB, SAR, IR, FCC) |
| **Query Routing** | Qwen3-VL-30B-A3B | Dispatches to Captioning/Grounding/VQA pipelines |

---

## Core Components

### `app/orchestrator.py` — LangGraph Workflow

The orchestrator defines the complete inference pipeline with state management:

```python
workflow = StateGraph(AgentState)
workflow.add_node("detect_modality", detect_modality_node)
workflow.add_node("preprocess_ir2rgb", preprocess_ir2rgb_node)
workflow.add_node("route_to_service", conditional_router)
workflow.add_node("vqa_subclassify", vqa_subclassify_node)
# ... task-specific nodes
```

**Key Functions:**
- `detect_modality_node()` — ResNet-based modality detection
- `preprocess_ir2rgb_node()` — FCC → RGB reconstruction
- `conditional_router()` — Task classification (Captioning/VQA/Grounding)
- `vqa_subclassify_node()` — VQA sub-type routing (Semantic/Binary/Numeric)

### `app/services/ir2rgb_service.py` — FCC-to-RGB Reconstruction

Implements the Root Polynomial Color Correction (RPCC) algorithm from the paper:

1. **Channel Classification** — Dual-path classifier identifies each FCC channel
2. **Missing Band Detection** — `B_miss = {R, G, B} \ {ŷ₁, ŷ₂, ŷ₃}`
3. **Reconstruction**:
   - R/G missing → RPCC: `C_out = ψ(x)ᵀv` where `ψ(x) = [x₁, x₂, x₃, √(x₁x₂), √(x₁x₃), √(x₂x₃), 1]ᵀ`
   - B missing → 3D-LUT with trilinear interpolation

### `app/services/modality_router.py` — Modality Detection

ResNet-18 classifier for image modality detection:
- **RGB** — Standard optical imagery
- **SAR** — Synthetic Aperture Radar
- **IR** — Thermal infrared
- **FCC** — False Color Composite (triggers IR2RGB preprocessing)

### `app/services/modal_client.py` — Modal GPU Interface

Client for calling Modal-deployed GPU services:

```python
class ModalServiceClient:
    async def call_captioning(image_b64: str, query: str) -> str
    async def call_vqa(image_b64: str, question: str) -> str
    async def call_grounding(image_b64: str, query: str) -> List[BBox]
    async def call_sam3_count(image_b64: str, prompt: str) -> int
    async def call_sam3_area(image_b64: str, prompt: str, gsd: float) -> float
```

---

## Modal Services (GPU)

Located in `modal_services/`, these run on [Modal](https://modal.com) for GPU-accelerated inference.

### `sam3_agent.py` — SAM3 Pyramidal Counting/Area

Implements the **Adaptive Hierarchical Grounding Network (AHG-Net)** from Section IV-D of the paper:

**Pyramidal Tiling Strategy:**
```python
# Extract overlapping crops at multiple tile sizes
tiles = extract_pyramidal_tiles(image, tile_sizes=[256, 512, 1024])
# Run SAM3 on each crop
masks = [sam3.segment(tile, prompt) for tile in tiles]
# Map back to global coordinates
global_masks = merge_with_nms(masks, iou_threshold=0.5)
```

**Six-Stage Pipeline:**
1. Subject-Reference Query Decomposition
2. Pyramidal SAM3 Segmentation
3. Centroid-based Clustering
4. Subject Confidence Estimation (HIGH/MEDIUM/LOW)
5. Chunked VLM Refinement
6. Final Spatial Reasoning

### `vllm_modal_deploy.py` — vLLM Server

Qwen3-VL-8B-Instruct with LoRA adapters, served via vLLM for efficient inference.

### `captioning.py` / `vqa.py` / `grounding.py`

Task-specialized models fine-tuned on DRISHTI-GCV splits:

| Service | Base Model | Fine-tuning | Data Split |
|---------|------------|-------------|------------|
| Captioning | Qwen3-VL-8B | LoRA + DPO | DRISHTI-GCV Specialized Caption |
| VQA | Qwen3-VL-8B | LoRA + DPO | DRISHTI-GCV VQA |
| Grounding | SAM3 + VLM | AHG-Net | DRISHTI-GCV Grounding |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestrator/chat` | POST | Main multimodal chat endpoint |
| `/orchestrator/ir2rgb` | POST | Standalone IR→RGB conversion |
| `/orchestrator/modality/status` | GET | Service health check |
| `/health` | GET | Backend health check |

### Request/Response Examples

**Chat Endpoint:**
```json
POST /orchestrator/chat
{
  "session_id": "abc123",
  "image_b64": "...",
  "query": "How many ships are in the harbor?",
  "mode": "auto"  // auto | captioning | vqa | grounding
}
```

**Response:**
```json
{
  "response": "There are 12 ships visible in the harbor.",
  "task_type": "vqa_numeric",
  "confidence": 0.92,
  "metadata": {
    "sam3_count": 12,
    "detected_modality": "rgb"
  }
}
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=isro_vision

# Modal Services
MODAL_VQA_URL=https://...
MODAL_CAPTIONING_URL=https://...
MODAL_SAM3_URL=https://...
MODAL_ROUTER_URL=https://...

# OpenAI (for task router fallback)
OPENAI_API_KEY=sk-...
```

---

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
modal deploy captioning.py
modal deploy vqa.py
```

---

## Training Hyperparameters (Reference)

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-VL-8B-Instruct |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| LoRA Dropout | 0.05 |
| Quantization | 4-bit QLoRA |
| Learning Rate | 5 × 10⁻⁵ |
| Optimizer | AdamW-8bit |
| DPO β | 0.1 |
