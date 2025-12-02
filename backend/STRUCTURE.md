# Directory Structure

This document describes the directory structure of the ISRO Vision API backend.

## Root Structure

```
backend/
├── app/                    # Main application code
│   ├── api/               # API routes and dependencies
│   ├── core/              # Core functionality (database, checkpointing)
│   ├── models/            # Database models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic services
│   ├── utils/             # Utility functions
│   ├── orchestrator.py    # LangGraph workflow
│   ├── config.py          # Configuration
│   └── main.py            # FastAPI application
│
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── fixtures/         # Test data generators
│   └── conftest.py       # Shared test fixtures
│
├── ir2rgb_models/         # IR2RGB conversion models
│   ├── ir2rgb_model.py
│   └── model_weights/
│       └── models_ir_rgb.npz
│
├── resnet_models/         # ResNet modality classifier models
│   ├── resnet_model.py
│   └── model_weights/
│       └── best_resnet.pth
│
├── modal_services/        # Modal-deployed services
│   ├── resnet_classifier.py
│   ├── upload_resnet_weights.py
│   ├── captioning.py
│   ├── vqa.py
│   └── grounding.py
│
├── Makefile              # Build automation
├── pyproject.toml        # Project dependencies
├── run.py                # Application entry point
└── init_db.py            # Database initialization
```

## Path References

### IR2RGB Model Path
- **Service Location**: `app/services/ir2rgb_service.py`
- **Model Location**: `ir2rgb_models/model_weights/models_ir_rgb.npz`
- **Path Calculation**: `Path(__file__).parent.parent.parent / "ir2rgb_models" / "model_weights" / "models_ir_rgb.npz"`
  - From `app/services/ir2rgb_service.py`:
    - `.parent` = `app/services/`
    - `.parent.parent` = `app/`
    - `.parent.parent.parent` = `backend/`
    - Result: `backend/ir2rgb_models/model_weights/models_ir_rgb.npz`

### ResNet Model Path
- **Upload Script**: `modal_services/upload_resnet_weights.py`
- **Model Location**: `resnet_models/model_weights/best_resnet.pth`
- **Default Path**: `../resnet_models/model_weights/best_resnet.pth`
  - When running from `backend/modal_services/`:
    - `../resnet_models/` = `backend/resnet_models/`
    - Result: `backend/resnet_models/model_weights/best_resnet.pth`

## Key Files

### Configuration
- `app/config.py` - Application settings (loads from `.env`)
- `pyproject.toml` - Python dependencies and project metadata

### Entry Points
- `run.py` - Start the FastAPI server
- `init_db.py` - Initialize MongoDB database

### Services
- `app/services/modality_router.py` - Image modality detection
- `app/services/ir2rgb_service.py` - IR2RGB conversion
- `app/services/resnet_classifier_client.py` - ResNet classifier client
- `app/services/modal_client.py` - Modal service client

### Workflow
- `app/orchestrator.py` - LangGraph workflow definition
- `app/core/checkpoint.py` - Workflow state persistence

## Testing

All tests are in the `tests/` directory:
- `tests/unit/` - Fast, isolated unit tests
- `tests/integration/` - Integration tests with mocked services
- `tests/fixtures/` - Test data generators

Run tests with:
```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-cov          # With coverage report
```

## Deployment

### Modal Services
Modal services are in `modal_services/`:
- Deploy from `backend/modal_services/` directory
- Example: `cd backend/modal_services && modal deploy resnet_classifier.py`

### Model Weights
- IR2RGB: `backend/ir2rgb_models/model_weights/models_ir_rgb.npz`
- ResNet: `backend/resnet_models/model_weights/best_resnet.pth`

