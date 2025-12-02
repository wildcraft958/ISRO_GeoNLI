.PHONY: help install setup test test-unit test-integration test-cov run dev clean lint format check-env init-db

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := uv run python
UV := uv
PYTEST := uv run pytest
PYTEST_ARGS := -v
COV_ARGS := --cov=app --cov-report=term-missing --cov-report=html

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)ISRO Vision API - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

check-uv: ## Check if uv is installed, install if missing
	@if ! command -v $(UV) &> /dev/null; then \
		echo "$(YELLOW)📦 uv is not installed. Installing uv...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.cargo/bin:$$PATH"; \
		echo "$(GREEN)✅ uv installed successfully$(NC)"; \
	else \
		echo "$(GREEN)✅ uv is already installed$(NC)"; \
	fi

install: check-uv ## Install dependencies using uv
	@echo "$(BLUE)📥 Installing dependencies with uv...$(NC)"
	$(UV) sync
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

check-env: ## Check if .env file exists, create template if missing
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  .env file not found. Creating template...$(NC)"; \
		cat > .env << 'EOF'; \
# Server settings\
HOST=0.0.0.0\
PORT=8000\
\
# Database settings\
MONGO_URL=mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000\
MONGO_DB_NAME=isro_vision\
\
# App settings\
SECRET_KEY=your_secret_key_here\
\
# Modal service settings\
MODAL_BASE_URL=https://your-org--app-name.modal.run\
\
# OpenAI settings (for auto-router)\
OPENAI_API_KEY=sk-your-api-key\
\
# LangSmith settings (optional, for monitoring)\
LANGSMITH_API_KEY=\
LANGSMITH_PROJECT=\
EOF
		echo "$(GREEN)✅ Created .env template. Please update it with your actual values.$(NC)"; \
	else \
		echo "$(GREEN)✅ .env file exists$(NC)"; \
	fi

init-db: ## Initialize database
	@echo "$(BLUE)🗄️  Initializing database...$(NC)"
	$(PYTHON) init_db.py
	@echo "$(GREEN)✅ Database initialized$(NC)"

setup: check-uv install check-env init-db ## Complete setup (install deps, create .env, init DB)
	@echo ""
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo ""
	@echo "To start the server, run:"
	@echo "  $(GREEN)make run$(NC)"
	@echo "or"
	@echo "  $(GREEN)make dev$(NC)"

test: ## Run all tests
	@echo "$(BLUE)🧪 Running all tests...$(NC)"
	$(PYTEST) tests/ $(PYTEST_ARGS)

test-unit: ## Run only unit tests
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	$(PYTEST) tests/unit -m unit $(PYTEST_ARGS)

test-integration: ## Run only integration tests
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	$(PYTEST) tests/integration -m integration $(PYTEST_ARGS)

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)🧪 Running tests with coverage...$(NC)"
	$(PYTEST) tests/ $(PYTEST_ARGS) $(COV_ARGS)
	@echo ""
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/index.html$(NC)"

test-fast: ## Run tests excluding slow ones
	@echo "$(BLUE)🧪 Running fast tests...$(NC)"
	$(PYTEST) tests/ -m "not slow" $(PYTEST_ARGS)

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)🧪 Running tests in watch mode...$(NC)"
	$(PYTEST) tests/ --watch $(PYTEST_ARGS)

run: ## Start the application server
	@echo "$(BLUE)🚀 Starting application server...$(NC)"
	$(PYTHON) run.py

dev: ## Start development server with auto-reload
	@echo "$(BLUE)🚀 Starting development server...$(NC)"
	$(PYTHON) run.py

lint: ## Run linter (ruff)
	@echo "$(BLUE)🔍 Running linter...$(NC)"
	$(UV) run ruff check app/ tests/
	@echo "$(GREEN)✅ Linting complete$(NC)"

lint-fix: ## Run linter and auto-fix issues
	@echo "$(BLUE)🔍 Running linter with auto-fix...$(NC)"
	$(UV) run ruff check --fix app/ tests/
	@echo "$(GREEN)✅ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	$(UV) run black app/ tests/
	@echo "$(GREEN)✅ Formatting complete$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)✨ Checking code formatting...$(NC)"
	$(UV) run black --check app/ tests/

type-check: ## Run type checker (mypy)
	@echo "$(BLUE)🔍 Running type checker...$(NC)"
	$(UV) run mypy app/ --ignore-missing-imports
	@echo "$(GREEN)✅ Type checking complete$(NC)"

check: lint format-check type-check ## Run all checks (lint, format, type-check)
	@echo "$(GREEN)✅ All checks passed$(NC)"

clean: ## Clean cache files and generated artifacts
	@echo "$(BLUE)🧹 Cleaning cache files...$(NC)"
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .mypy_cache 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf dist 2>/dev/null || true
	@rm -rf build 2>/dev/null || true
	@rm -rf *.egg-info 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)🧹 Cleaning virtual environment...$(NC)"
	@rm -rf .venv 2>/dev/null || true
	@echo "$(GREEN)✅ Complete cleanup done$(NC)"

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "$(BLUE)📝 Generating requirements.txt...$(NC)"
	$(UV) pip compile pyproject.toml -o requirements.txt
	@echo "$(GREEN)✅ requirements.txt generated$(NC)"

requirements-dev: ## Generate requirements-dev.txt
	@echo "$(BLUE)📝 Generating requirements-dev.txt...$(NC)"
	$(UV) pip compile pyproject.toml --extra dev -o requirements-dev.txt
	@echo "$(GREEN)✅ requirements-dev.txt generated$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	docker build -t isro-vision-api:latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	docker run -p 8000:8000 --env-file .env isro-vision-api:latest

logs: ## Show application logs (if using systemd or similar)
	@echo "$(BLUE)📋 Showing logs...$(NC)"
	@journalctl -u isro-vision-api -f 2>/dev/null || echo "$(YELLOW)No systemd service found$(NC)"

status: ## Check service status
	@echo "$(BLUE)📊 Service Status:$(NC)"
	@echo "  Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "  uv: $$(uv --version 2>/dev/null || echo 'Not found')"
	@echo "  Dependencies: $$([ -d .venv ] && echo 'Installed' || echo 'Not installed')"
	@echo "  .env file: $$([ -f .env ] && echo 'Exists' || echo 'Missing')"

