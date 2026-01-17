# DRISHTI

> **Deep Remote-sensing Intelligence for Semantic Hybrid Text-Image Understanding**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-akshadrishti.space-blue)](https://www.akshadrishti.space/)

---

## Overview

DRISHTI is a unified Vision-Language Model (VLM) framework for natural language interaction with satellite and remote sensing imagery. It enables non-expert users to analyze geospatial data through intuitive text queries.

### Key Features

- **Multi-Modal Support**: Handles RGB, Infrared (IR), SAR, and False Color Composite (FCC) imagery.
- **Task Router**: Automatically classifies queries into Captioning, VQA, or Grounding.
- **SAM3 Integration**: Leverages Segment Anything Model 3 for accurate object counting and area estimation.
- **IR2RGB Preprocessing**: Converts non-RGB imagery for VLM compatibility.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DRISHTI System                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Frontend (React/Vite)                                                  │
│    └── User Interface for image upload and natural language queries    │
├─────────────────────────────────────────────────────────────────────────┤
│  Backend (FastAPI)                                                      │
│    ├── Orchestrator (LangGraph workflow)                                │
│    ├── Task Router (VQA / Captioning / Grounding)                       │
│    ├── Modality Detector (RGB, IR, SAR, FCC)                            │
│    └── IR2RGB Preprocessor                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Modal Services (GPU)                                                   │
│    ├── vLLM (Qwen-VL based VLM)                                         │
│    ├── SAM3 Agent (counting, area estimation)                           │
│    └── ResNet Classifier (modality fallback)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── backend/                 # FastAPI Application
│   ├── app/                 # Core application logic
│   │   ├── api/             # API routes (orchestrator, chat, etc.)
│   │   ├── services/        # Business logic (ir2rgb, modality_router, etc.)
│   │   └── orchestrator.py  # LangGraph workflow definition
│   └── modal_services/      # GPU-accelerated services for Modal deployment
│       ├── sam3_agent.py    # SAM3 counting/area module
│       ├── vllm_modal_deploy.py  # VLM deployment
│       └── ...
├── frontend/                # React/Vite UI
│   └── src/
├── docs/                    # Documentation & Reports
└── internal_data/           # Sample images and SAM3 agent source
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB (for session persistence)
- Modal account (for GPU services)

### Backend Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Or: uv pip install -r requirements.txt
cp .env.example .env              # Configure environment variables
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Documentation

- [Final Report (PDF)](docs/DRISHTI_REPORT_FINAL.pdf) — Detailed methodology, architecture, and results.
- [Problem Statement (PDF)](docs/ISRO_M3_TechMeet14.pdf) — ISRO TechMeet 14 challenge description.

---

## License

This project was developed for ISRO Inter-IIT TechMeet 14.
