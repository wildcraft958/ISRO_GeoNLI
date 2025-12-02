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

# OpenAI settings (for auto-router)
OPENAI_API_KEY=sk-your-api-key

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

The orchestrator integrates with Modal-deployed VLM services:
- Grounding (bounding box detection)
- VQA (visual question answering)
- Captioning (image description)

See `modal_services/` for service implementations.

## License

ISC

