#!/bin/bash

# Setup script for ISRO Vision API using uv

set -e

echo "🚀 Setting up ISRO Vision API..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Install dependencies
echo "📥 Installing dependencies with uv..."
uv sync

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
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
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
EOF
    echo "✅ Created .env template. Please update it with your actual values."
else
    echo "✅ .env file exists"
fi

# Initialize database
echo "🗄️  Initializing database..."
uv run python init_db.py

echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  uv run python run.py"
echo "or"
echo "  ./start.sh"

