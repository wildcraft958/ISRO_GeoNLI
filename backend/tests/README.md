# ISRO Vision API Test Suite

Comprehensive pytest test suite for the ISRO Vision API backend.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and mocks
├── pytest.ini               # Pytest configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_modality_router.py
│   ├── test_ir2rgb_service.py
│   ├── test_resnet_client.py
│   ├── test_schemas.py
│   └── test_memory_service.py
├── integration/             # Integration tests
│   ├── test_orchestrator_routes.py
│   └── test_orchestrator.py
└── fixtures/               # Test data generators
    └── test_data.py
```

## Setup

1. **Install test dependencies:**
   ```bash
   cd backend
   uv sync --dev
   # or
   pip install -e ".[dev]"
   ```

2. **Set environment variables** (optional, defaults in pytest.ini):
   ```bash
   export MONGODB_URL="mongodb://localhost:27017"
   export OPENAI_API_KEY="test-key"
   ```

## Running Tests

```bash
# Run all tests
cd backend
pytest

# Run only unit tests
pytest tests/unit -m unit

# Run only integration tests
pytest tests/integration -m integration

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_modality_router.py

# Run specific test
pytest tests/unit/test_modality_router.py::TestImageLoader::test_load_image_arr_from_pil

# Run in parallel (faster)
pytest -n auto
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_db` - Tests requiring database
- `pytest.mark.requires_modal` - Tests requiring Modal services

Run tests excluding slow ones:
```bash
pytest -m "not slow"
```

## Coverage

Generate coverage report:
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html  # View in browser
```

Target coverage: 70%+ (configured in pyproject.toml)

## Test Count

- **Unit Tests**: 130+ tests
- **Integration Tests**: 35+ tests
- **Total**: 165+ tests

## Troubleshooting

### ModuleNotFoundError: No module named 'app'

Make sure you're running pytest from the `backend/` directory:
```bash
cd backend
pytest
```

### Missing dependencies

Install all dependencies:
```bash
cd backend
uv sync --dev
```

### MongoDB connection errors

Tests use mocked MongoDB by default. If you see connection errors, check that mocks are properly configured in `conftest.py`.
