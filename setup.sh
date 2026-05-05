#!/bin/bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — add your API keys before running"
else
    echo ".env already exists, skipping"
fi

echo "Setup complete. Run: python -m uvicorn api:app --port 8000"