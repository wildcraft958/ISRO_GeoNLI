# Modal Services for Multimodal Chatbot

This directory contains scaffold implementations for three VLM services deployed on Modal.

**Note**: This project uses `pyproject.toml` and `uv` for dependency management. See `pyproject.toml` for dependencies.

1. **grounding.py** - Bounding box detection service
2. **vqa.py** - Visual question answering service
3. **captioning.py** - Image captioning service

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

### Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Authenticate with Modal:
```bash
uv run modal token new
```

3. Deploy each service:
```bash
uv run modal deploy grounding.py
uv run modal deploy vqa.py
uv run modal deploy captioning.py
```

## Configuration

After deployment, Modal will provide HTTP endpoints for each service. Update the `MODAL_BASE_URL` environment variable in your backend configuration to point to your Modal deployment.

Example:
```bash
MODAL_BASE_URL=https://your-org--app-name.modal.run
```

## Implementation Notes

These are scaffold files with placeholder implementations. To use them in production:

1. Replace placeholder logic with actual VLM model inference
2. Load your fine-tuned models in the `@stub.function` decorators
3. Adjust GPU requirements based on model size
4. Add proper error handling and logging
5. Implement image preprocessing (resize, normalization, etc.)

## Service Endpoints

Each service exposes an HTTP endpoint that accepts:
- **grounding**: `POST /grounding` with `{"image_url": str, "query": str}`
- **vqa**: `POST /vqa` with `{"image_url": str, "query": str}`
- **captioning**: `POST /captioning` with `{"image_url": str}`

## Response Formats

- **grounding**: `{"bboxes": [{"x": float, "y": float, "w": float, "h": float, "label": str, "confidence": float}]}`
- **vqa**: `{"answer": str, "confidence": float}`
- **captioning**: `{"caption": str}`

