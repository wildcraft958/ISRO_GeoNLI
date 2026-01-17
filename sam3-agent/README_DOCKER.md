# SAM3 Agent Docker Deployment

This directory contains the Docker configuration for running the SAM3 Agent as a standalone FastAPI service.

## Prerequisites

- NVIDIA GPU with drivers installed
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- HuggingFace Token (SAM3 is a gated model)

## Structure

The application is structured as follows:

```
sam3/
  sam3_app/          # Core Application Logic
    api/             # FastAPI Endpoints & Schemas
    core/            # Model, VLM, and Inference Logic
    main.py          # Entrypoint
  Dockerfile         # Container definition
  docker-compose.yml # Orchestration config
  pyproject.toml     # Python dependencies and project configuration
```

## Quick Start (Docker Compose)

1. **Set your HuggingFace Token**:

   ```bash
   export HF_TOKEN=hf_...
   ```

2. **Build and Run**:

   ```bash
   docker-compose up --build
   ```

   The service will be available at `http://localhost:8000`.

## API Documentation

Once running, access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

- `POST /sam3/count`: Count objects in an image.
- `POST /sam3/area`: Calculate area of objects.
- `POST /sam3/segment`: Full segmentation.

## Configuration

Environment variables:

- `HF_TOKEN`: Required. HuggingFace token for downloading SAM3.
- `PORT`: Optional. Port to run on (default 8000).

## Note on Model Loading

The SAM3 model is large and requires a GPU. The model is loaded on application startup. Ensure your Docker container has GPU access (`--gpus all` or via docker-compose `deploy.resources`).
