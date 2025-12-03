# ISRO Vision API - Multimodal Chatbot Orchestrator

A production-grade multimodal chatbot orchestrator using LangGraph, FastAPI, and Modal services.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- `make` - Build automation tool (usually pre-installed on Linux/Mac)

## Quick Start

The easiest way to get started is using the Makefile:

```bash
cd backend
make setup    # Complete setup (installs uv, deps, creates .env, initializes DB)
make run      # Start the server
```

Or see all available commands:
```bash
make help
```

## Installation

### Install uv

The Makefile will automatically install uv if missing, or you can install it manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

### Setup Project

**Using Makefile (Recommended):**

```bash
cd backend
make setup
```

This will:
- Check/install uv automatically
- Install all dependencies from `pyproject.toml`
- Create `.env` template if missing
- Initialize the database

**Manual Setup:**

```bash
cd backend
uv sync              # Install dependencies
uv run python init_db.py  # Initialize database
```

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

# LLM settings (for task routing and VQA classification)
LLM_MODEL_NAME=                    # Optional: specific model name
LLM_TIMEOUT=30                     # Timeout for LLM calls in seconds

# Optional: Custom system prompts (defaults are provided)
# LLM_ROUTER_SYSTEM_PROMPT=...     # Custom task router prompt
# LLM_VQA_CLASSIFIER_SYSTEM_PROMPT=...  # Custom VQA classifier prompt

# LangSmith settings (optional, for monitoring)
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=multimodal-chatbot
```

## Running the Application

**Using Makefile (Recommended):**

```bash
make run    # Start production server
make dev    # Start development server (same as run, with auto-reload)
```

**Manual:**

```bash
uv run python run.py
```

The API will be available at `http://localhost:8000`

## Development

### Common Makefile Commands

```bash
# Testing
make test              # Run all tests
make test-unit         # Run only unit tests
make test-integration  # Run only integration tests
make test-cov          # Run tests with coverage report
make test-fast         # Run tests excluding slow ones

# Code Quality
make lint              # Run linter (ruff)
make lint-fix          # Run linter and auto-fix issues
make format            # Format code with black
make format-check      # Check formatting without changes
make type-check        # Run type checker (mypy)
make check             # Run all checks (lint, format, type-check)

# Maintenance
make clean             # Clean cache files and artifacts
make clean-all         # Clean everything including venv
make status            # Check service status
```

### Install Dev Dependencies

```bash
make install           # Installs all dependencies including dev
# or manually:
uv sync --dev
```

### Run with Auto-reload

The server runs with auto-reload enabled by default:
```bash
make dev
```

### Manual Development Commands

If you prefer not using Makefile:

```bash
uv run black app/ tests/           # Format code
uv run ruff check app/ tests/      # Lint code
uv run mypy app/                   # Type check
uv run pytest tests/               # Run tests
```

## API Endpoints

### Orchestrator Endpoints

- `POST /orchestrator/chat` - Main chat endpoint with task routing and VQA sub-classification
- `POST /orchestrator/ir2rgb` - Standalone IR2RGB conversion endpoint
- `GET /orchestrator/ir2rgb/status` - Check IR2RGB service status
- `GET /orchestrator/modality/status` - Check modality detection service status
- `GET /orchestrator/llm/status` - Check LLM service status
- `POST /orchestrator/router/test` - Test task router classification
- `POST /orchestrator/vqa-classifier/test` - Test VQA sub-classification
- `GET /orchestrator/session/{session_id}/history` - Get session history
- `GET /orchestrator/sessions` - Get user sessions
- `GET /orchestrator/sessions/{session_id}/messages` - Get session messages
- `POST /orchestrator/sessions/{session_id}/summarize` - Summarize conversation
- `DELETE /orchestrator/session/{session_id}` - Clear session

### Health Check

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint

## Task Routing and VQA Sub-Classification

The orchestrator uses a Modal-deployed LLM for intelligent task routing:

### Task Router
Classifies user queries into one of three task pipelines:
- **VQA**: Visual Question Answering (what, why, how, count, area)
- **Grounding**: Object detection/localization (where, locate, find)
- **Captioning**: General image description (no specific question)

### VQA Sub-Classification
For VQA tasks, queries are further classified into sub-types:
- **yesno**: Yes/no questions ("Is there a building?")
- **general**: Open-ended questions ("What type of land cover is shown?")
- **counting**: Count objects ("How many vehicles are there?")
- **area**: Area calculations ("What is the area of the forest region?")

### System Prompts
Both the task router and VQA classifier use configurable system prompts.
You can customize them via environment variables:
- `LLM_ROUTER_SYSTEM_PROMPT`
- `LLM_VQA_CLASSIFIER_SYSTEM_PROMPT`

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
├── Makefile          # Build automation
├── run.py
└── init_db.py
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

The orchestrator integrates with Modal-deployed services:

### VLM Services
- **Grounding** - Bounding box detection for object localization
- **VQA** - Visual question answering with sub-type support
- **Captioning** - Image description generation

### LLM Services
- **Task Router** - Classifies queries into VQA/Grounding/Captioning
- **VQA Classifier** - Classifies VQA queries into yesno/general/counting/area

The LLM services use a general-purpose LLM endpoint (`/v1/chat/completions`) 
with configurable system prompts for different classification tasks.

See `modal_services/` for service implementations.

## License

ISC

