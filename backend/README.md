# ISRO Vision API - Multimodal Chatbot Orchestrator

A production-grade multimodal chatbot orchestrator using LangGraph, FastAPI, and Modal services.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

## Installation

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

### Setup Project

1. Clone the repository and navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies using uv:
```bash
uv sync
```

This will:
- Create a virtual environment automatically
- Install all project dependencies from `pyproject.toml`
- Install dev dependencies (if needed, use `uv sync --dev`)

## Configuration

Create a `.env` file in the `backend/` directory:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000

# Database settings
MONGO_URL=mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000
MONGO_DB_NAME=isro_vision

# App settings
SECRET_KEY=your_secret_key_here

# Modal service settings
MODAL_BASE_URL=https://your-org--app-name.modal.run

# OpenAI settings (for auto-router)
OPENAI_API_KEY=sk-your-api-key

# LangSmith settings (optional, for monitoring)
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=multimodal-chatbot
```

## Running the Application

### Initialize Database

```bash
uv run python init_db.py
```

### Start the Server

```bash
uv run python run.py
```

Or use the start script:
```bash
./start.sh
```

The API will be available at `http://localhost:8000`

## Development

### Install Dev Dependencies

```bash
uv sync --dev
```

### Run with Auto-reload

The server runs with auto-reload enabled by default when using `run.py`.

### Format Code

```bash
uv run black .
```

### Lint Code

```bash
uv run ruff check .
```

### Type Checking

```bash
uv run mypy .
```

## API Endpoints

### Orchestrator Endpoints

- `POST /api/orchestrator/chat` - Main chat endpoint
- `GET /api/orchestrator/session/{session_id}/history` - Get session history

### Health Check

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── orchestrator_routes.py
│   ├── core/
│   │   ├── checkpoint.py
│   │   └── database.py
│   ├── models/
│   ├── schemas/
│   │   └── orchestrator_schema.py
│   ├── services/
│   │   └── modal_client.py
│   ├── utils/
│   │   └── session_manager.py
│   ├── orchestrator.py
│   ├── config.py
│   └── main.py
├── pyproject.toml
├── run.py
└── start.sh
```

## Dependencies

All dependencies are managed in `pyproject.toml` using `uv`. Key dependencies:

- **FastAPI** - Web framework
- **LangGraph** - Workflow orchestration
- **LangChain OpenAI** - LLM integration for auto-routing
- **Motor** - Async MongoDB driver
- **Tenacity** - Retry logic
- **Pydantic** - Data validation

## Modal Services

The orchestrator integrates with Modal-deployed VLM services:
- Grounding (bounding box detection)
- VQA (visual question answering)
- Captioning (image description)

See `../modal_services/` for service implementations.

## License

ISC

