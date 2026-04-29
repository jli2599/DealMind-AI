# DealMind AI

An agentic M&A analyst built with LangGraph and Llama 3.3 70B (via HuggingFace).

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/dealmind-ai.git
cd dealmind-ai
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your HuggingFace token
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create one with **"Make calls to serverless Inference API"** permission.

```bash
cp .env.example .env
# open .env and paste your token
```

## Running

### Terminal
```bash
python graph.py
```

### LangGraph Studio
```bash
langgraph dev
```
Then open: **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024**

Paste this into the input panel and hit Submit:
```json
{
  "messages": [
    {"role": "user", "content": "Who are the likely buyers for a $250M SaaS company?"}
  ]
}
```

## Project Structure

```
dealmind-ai/
├── graph.py          # LangGraph graph definition
├── requirements.txt  # Python dependencies
├── .env.example      # Token template — copy to .env
├── .gitignore        # Keeps .env out of git
└── README.md
```

## Model

Uses **Llama 3.3 70B Instruct** via the HuggingFace Inference API — runs on HuggingFace's servers, no GPU required.