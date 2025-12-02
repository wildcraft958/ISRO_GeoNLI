#!/bin/bash

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
uv sync

# Initialize database
echo "Initializing database..."
uv run python init_db.py

# Start application server
echo "Starting application server..."
uv run python run.py