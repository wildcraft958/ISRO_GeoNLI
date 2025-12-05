# AI Chat Backend for Satellite Imagery Analysis

## Overview

This project implements a robust, asynchronous FastAPI backend designed to process and analyze multi-modal satellite imagery (RGB, SAR, IR, FCC). It leverages a sophisticated AI orchestration engine powered by LangGraph to manage conversation state and route queries to specialized AI models.

The system is built to handle complex workflows, including automatic intent classification, dynamic prompt engineering for specialized modalities, and structured data extraction (bounding boxes) for grounding tasks.

## Key Capabilities

### Multi-Modality Handling

- **RGB**: Standard vision pipeline supporting Visual Question Answering (VQA), Captioning, and Grounding
- **SAR (Synthetic Aperture Radar)**: Dedicated handling with expert-system prompts for radar signature analysis
- **IR (Infrared)**: Dedicated handling for thermal signature analysis
- **FCC (False Color Composite)**: Automated conversion to RGB for compatibility with standard vision models

### Intelligent Routing & Orchestration

- **LangGraph State Machine**: Manages the decision flow, context retention, and model execution steps
- **Auto-Intent Detection**: Uses a Generic LLM to classify user queries (e.g., "Find all ships" \(\rightarrow\) Grounding Mode)
- **Model Registry (Strategy Pattern)**: Decouples core logic from model API details, allowing easy integration of new models

### Contextual Memory

Maintains a rolling summary of the conversation history, rewritten by an LLM after each turn to ensure context is preserved without token bloat.

### Scalable Architecture

- **Async/Await**: Built on httpx for non-blocking I/O operations
- **PostgreSQL + JSONB**: Efficient storage for unstructured metadata (like bounding box coordinates)
- **Dockerized**: Ready for deployment with a lightweight container footprint

## Architecture

The system follows a layered architecture separating concerns between API handling, business logic, and external model integration.

### Pipeline Workflow

#### Ingestion (Step 0)

**Endpoint**: `POST /api/v1/image/upload`

**Process**:

- Uploads image to AWS S3
- Classifies image type (RGB, SAR, IR) using a ResNet model
- **Preprocessing**:
  - If FCC detected \(\rightarrow\) Convert to RGB via fcc_converter
  - If SAR/IR \(\rightarrow\) Tag as specialized modality
- Creates a Session in PostgreSQL

#### Orchestration (Step 1)

**Endpoint**: `POST /api/v1/orchestration/chat`

**Router Logic**:

- **SAR/IR**: Direct routing to SARHandler / IRHandler. These handlers dynamically select a system prompt (Captioning vs. VQA) based on the user's text
- **RGB**: Enters the LangGraph orchestrator
  - **Intent Node**: Classifies query into VQA, CAPTIONING, or GROUNDING
  - **VQA Classifier**: If VQA, sub-classifies into BINARY, NUMERICAL, or GENERAL
  - **Execution Node**: Uses the Model Registry to select the correct adapter and call the external Model API
  - **Memory Node**: Saves the interaction to the DB and updates the conversation summary

## Project Structure

```
app/
├── api/
│   └── v1/
│       └── endpoints/
│           ├── upload.py           # Image ingestion & classification
│           ├── chat.py             # Main chat orchestration endpoint
│           ├── user.py             # User management
│           └── test_pipeline.py    # End-to-end testing suite
├── core/
│   ├── config.py                   # Environment variables & settings
│   └── database.py                 # Database connection & session
├── db/
│   └── models.py                   # SQLAlchemy ORM models
├── schemas/
│   └── test_structure.py           # Pydantic models for testing
├── services/
│   ├── orchestrator.py             # LangGraph state machine & workflow
│   ├── model_registry.py           # Strategy Pattern for Model APIs
│   ├── classifier.py               # Image modality classification
│   ├── s3_service.py               # S3 upload utility
│   ├── fcc_converter.py            # FCC to RGB conversion service
│   └── modalities/
│       ├── ir_handler.py           # Specialized IR prompt logic
│       └── sar_handler.py          # Specialized SAR prompt logic
└── main.py                         # Application entry point
```

## Setup & Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- AWS S3 Bucket access
- External AI Model Endpoints (e.g., Modal.com)
- Docker & Docker Compose (Optional but recommended)

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd prod-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory with your credentials:

```env
# Core
PROJECT_NAME="AI Chat Backend"
API_V1_STR="/api/v1"

# Database
DATABASE_URL="postgresql://user:password@localhost:5432/your_db_name"

# AWS S3
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
AWS_REGION="us-east-1"
S3_BUCKET_NAME="your_bucket_name"

# Auth
CLERK_SECRET_KEY="your_clerk_secret_key"

# AI Services - Model Endpoints
RESNET_URL="https://.../classify"
IR_CONVERT_URL="https://.../convert-ir"
FCC_CONVERT_URL="https://.../convert-fcc"
GENERIC_LLM_URL="https://.../llm-router"

# Specialist Models
VQA_BINARY_URL="https://.../vqa-binary"
VQA_NUMERICAL_URL="https://.../vqa-numerical"
VQA_GENERAL_URL="https://.../vqa-general"
CAPTION_URL="https://.../captioning"
GROUNDING_URL="https://.../grounding"
SAR_URL="https://.../sar-special"
IR_URL="https://.../ir-special"
```

### Running the Server

#### Option A: Using Docker (Recommended)

**Build the Image**:

```bash
docker build -t ai-chat-backend .
```

**Run the Container**:

```bash
docker run -d --name ai-backend -p 8000:8000 --env-file .env ai-chat-backend
```

This command runs the container in detached mode (`-d`), mapping port 8000 on the host to port 8000 in the container (`-p 8000:8000`), and injects your environment variables from the `.env` file (`--env-file .env`).

**View Logs**:

```bash
docker logs -f ai-backend
```

**Stop Container**:

```bash
docker stop ai-backend
```

#### Option B: Using Uvicorn (Local Dev)

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Access the Swagger UI documentation at `http://localhost:8000/docs`.

## API Usage

### User Management

- **Create User**: `POST /api/v1/users/`
- **Get History**: `GET /api/v1/users/{user_id}/history`

### Image Upload

**Endpoint**: `POST /api/v1/image/upload`

**Payload**: `multipart/form-data` with `file` and `user_id`

**Response**:

```json
{
  "chat_id": "uuid-string",
  "image_url": "https://s3.../image.jpg",
  "image_type": "RGB"
}
```

### Chat Interaction

**Endpoint**: `POST /api/v1/orchestration/chat`

**Payload**:

```json
{
  "chat_id": "uuid-string",
  "user_id": "user_123",
  "query_text": "Count the number of aircraft on the runway",
  "mode": "AUTO"
}
```

**Response**:

```json
{
  "response_text": "I counted 3 aircraft on the main runway.",
  "mode_used": "VQA",
  "metadata": null
}
```

### Testing

**Endpoint**: `POST /api/v1/test`

**Description**: Runs an automated end-to-end test sequence (Captioning → Grounding → VQA) on a provided image URL to validate the entire pipeline.

**Payload**:

```json
{
  "image_url": "https://example.com/test-image.jpg",
  "user_id": "test_user"
}
```

