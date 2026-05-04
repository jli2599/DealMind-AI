# DealMind AI

An agentic M&A analyst built with LangGraph, FastAPI, and Llama 3.3 70B (via Groq).

## Architecture

DealMind uses a multi-agent LangGraph system where an orchestrator routes tasks to six specialist sub-agents, a critic evaluates their outputs in a closed feedback loop, and a synthesizer produces the final report.

```
input → orchestrator → [sourcing, valuation, strategy, risk, financing, positioning]
                     ↑                                                              ↓
                     └──────────────────── critic ←──────────────────────────────┘
                                              ↓ (approved)
                                         synthesizer → END
```

The **sourcing agent** is fully expanded with web search tooling to ground buyer identification in real, recent M&A data. The remaining five sub-agents are intentional LLM calls kept simple for prototype scope — the architecture supports upgrading them independently.

## Project Structure

```
dealmind-ai/
├── graph.py          # LangGraph graph — all agents and routing logic
├── api.py            # FastAPI server — exposes /analyse endpoint
├── index.html        # Frontend — company input form and tabbed report UI
├── requirements.txt  # Python dependencies
├── langgraph.json    # LangGraph Studio config
├── .env.example      # Token template — copy to .env
├── .gitignore        # Keeps secrets and checkpoints out of git
└── README.md
```

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

### 4. Add your API keys
```bash
cp .env.example .env
# open .env and fill in your keys
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit card needed.

## Running

### Option A — Web UI (how it is supposed to be run)

**Terminal 1** — start the API:
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2** — open the frontend:
```bash
open index.html
```

Enter a company description, select seller objectives, and hit **Run Analysis**. The UI shows a live progress tracker and renders the final report in tabbed sections.

### Option B — LangGraph Studio (to see the behind the scenes)

```bash
langgraph dev --port 8123
```

Then open: **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123**

Type your deal description into the **User Message** field and hit Submit. Studio shows each node executing in real time with full state inspection at every step.

### Option C — Terminal
```bash
python graph.py
```

## Example Input

```
PulseFlow is a B2B SaaS company providing AI-powered workforce scheduling and
labor analytics to mid-sized healthcare networks across the US. Founded 2021,
78 employees, $16M ARR, 122% net revenue retention, 14% EBITDA margin,
headquartered in Denver CO. Seeking a $130M exit within 9 months.
Management willing to roll 15% equity. Run a full M&A analysis.
```

## Model

Uses **Llama 3.3 70B Instruct** via [Groq](https://console.groq.com) — fast, free tier, no GPU required. Groq's free tier provides 14,400 requests/day which is sufficient for development and demo use.

> **Note on rate limits:** Each full analysis makes 10-15 LLM calls. On Groq's free tier (12,000 tokens per minute), running multiple analyses back-to-back may trigger a 429 rate limit error (can happen on 2-3 consecutive runs). If this happens, wait some time before retrying — the token window resets every minute, or ideally replace the free API token with paid tokens. The code includes automatic retry logic with exponential backoff, but very rapid consecutive runs may still hit the limit. For demos, allow at least 1-2 minutes between runs.

## Agent Design

| Agent | Type | Notes |
|---|---|---|
| Orchestrator | LLM | Routes tasks, tracks completion, reads critic feedback |
| Sourcing | **Expanded — web search** | Grounds buyer identification in real M&A data |
| Valuation | LLM | EV multiples, implied range, value drivers |
| Strategy | LLM | Synergy analysis, integration complexity |
| Risk | LLM | Regulatory, execution, market, technology risks |
| Financing | LLM | Deal structure, consideration mix, leverage |
| Positioning | LLM | Growth narrative, process type, materials |
| Critic | LLM | Evaluates quality, retasks failed agents (max 2 iterations) |
| Synthesizer | LLM | Compiles final ranked buyer list and report |
