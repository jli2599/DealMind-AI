# graph.py

from __future__ import annotations

from typing import Annotated, TypedDict
import json

from huggingface_hub import InferenceClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import os

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

def call_llm(system: str, user: str) -> str:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# Add "next" to State so the orchestrator can signal which agent to call
class State(TypedDict, total=False):
    user_message: str
    messages: Annotated[list, add_messages]
    next: str
    completed_agents: list        # tracks which agents have run
    orchestrator_brief: str
    sourcing_output: str
    valuation_output: str
    strategy_output: str
    risk_output: str
    financing_output: str
    positioning_output: str

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Orchestrator now decides which agent to call
ORCHESTRATOR_SYSTEM = """You are the Orchestrator for DealMind AI, an M&A intelligence platform.
Your job is to run a full M&A analysis by calling each specialist agent one at a time.

The agents you must call are:
- sourcing
- valuation
- strategy
- risk
- financing
- positioning

You will be told which agents have already been called. Pick the next one that hasn't run yet.
Once ALL six agents have been called, set agent to "finish".

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "agent": "<agent_name or finish>",
  "brief": {
    "company": "<company name if mentioned, else null>",
    "sector": "<sector>",
    "deal_size": "<deal size if mentioned, else null>",
    "key_facts": ["<fact 1>", "<fact 2>", "<fact 3>"],
    "task": "<specific instruction for the agent in 1-2 sentences>"
  }
}"""

SOURCING_SYSTEM = """You are the Sourcing Agent for DealMind AI.
Identify 8 credible buyer candidates (mix of strategic buyers and financial sponsors).
For each return: Name | Type (Strategic/Sponsor) | HQ | Rationale (1 sentence).
Format as a markdown table."""

VALUATION_SYSTEM = """You are the Valuation Agent for DealMind AI.
Provide a concise valuation analysis:
- EV/Revenue and EV/EBITDA multiple ranges (based on comparable SaaS transactions)
- Implied enterprise value range
- 3 key value drivers
Format as clean markdown."""

STRATEGY_SYSTEM = """You are the Strategy Agent for DealMind AI.
Assess the strategic rationale for an acquisition:
- Top 3 synergy opportunities (revenue, cost, or technology)
- Integration complexity: Low / Medium / High (with one line explanation)
- Competitive positioning post-acquisition
Format as clean markdown."""

RISK_SYSTEM = """You are the Risk Agent for DealMind AI.
Identify the top risks for this transaction.
For each risk provide: Risk | Rating (Low/Medium/High) | Mitigation
Cover: regulatory, execution, market timing, and technology risks.
Format as a markdown table."""

FINANCING_SYSTEM = """You are the Financing Agent for DealMind AI.
Outline the likely deal financing structure:
- Cash vs stock vs earnout split
- Leverage capacity for sponsor buyers
- Recommended deal structure (merger, asset sale, etc.)
Format as clean markdown."""

POSITIONING_SYSTEM = """You are the Positioning Agent for DealMind AI.
Recommend how to position and run this deal process:
- 2-3 sentence growth narrative for the CIM
- Process type: broad auction vs targeted (justify in one sentence)
- Top 3 materials needed to launch
Format as clean markdown."""

SYNTHESIZER_SYSTEM = """You are the Synthesizer for DealMind AI.
You have received analysis from 6 specialist M&A agents. Combine their outputs into a final report.

Your report must include:

## Executive Summary
3-4 sentences summarising the deal and key findings.

## Ranked Buyer List
Rank the top 5 buyers from the sourcing analysis. For each include:
- Rank, Name, Type (Strategic/Sponsor)
- Strategic rationale (2-3 sentences)
- Fit score out of 10
- Recommended outreach approach

## Valuation Summary
Key multiples and implied EV range.

## Risk Overview
Top 3 risks and mitigations.

## Process Recommendation
Auction vs targeted, timeline, first steps.

Be specific. No generic statements."""

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_node(state: State) -> dict:
    return {"messages": [HumanMessage(content=state["user_message"])]}

def orchestrator_node(state: State) -> dict:
    completed = state.get("completed_agents", [])
    
    prompt = f"""User request: {state["user_message"]}

    Agents already called: {completed if completed else "none yet"}
    Pick the next agent that has NOT been called yet, or finish if all six are done."""

    raw = call_llm(ORCHESTRATOR_SYSTEM, prompt)

    try:
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)
        next_agent = parsed.get("agent", "finish")
        brief = parsed.get("brief", {})
    except json.JSONDecodeError:
        raise ValueError(f"Orchestrator did not return valid JSON: {raw}")

    return {
        "next": next_agent,
        "orchestrator_brief": json.dumps(brief, indent=2),
        "messages": [AIMessage(content=f"[Orchestrator] routing to: {next_agent}\n{json.dumps(brief, indent=2)}")],
    }

def route(state: State) -> str:
    next_agent = state.get("next")
    if not next_agent:
        raise ValueError("Orchestrator did not set a next agent")
    return next_agent

def sourcing_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(SOURCING_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["sourcing"],
        "messages": [AIMessage(content=f"[Sourcing]\n{result}")],
    }

def valuation_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(VALUATION_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["valuation"],
        "messages": [AIMessage(content=f"[Valuation]\n{result}")],
    }

def strategy_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(STRATEGY_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["strategy"],
        "messages": [AIMessage(content=f"[Strategy]\n{result}")],
    }

def risk_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(RISK_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["risk"],
        "messages": [AIMessage(content=f"[Risk]\n{result}")],
    }

def financing_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(FINANCING_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["financing"],
        "messages": [AIMessage(content=f"[Financing]\n{result}")],
    }

def positioning_node(state: State) -> dict:
    brief = json.loads(state["orchestrator_brief"])
    user_prompt = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    result = call_llm(POSITIONING_SYSTEM, user_prompt)
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": result,
        "completed_agents": completed + ["positioning"],
        "messages": [AIMessage(content=f"[Positioning]\n{result}")],
    }

def synthesizer_node(state: State) -> dict:
    prompt = f"""
        Company: {state.get('user_message', '')}

        SOURCING OUTPUT:
        {state.get('sourcing_output', '')}

        VALUATION OUTPUT:
        {state.get('valuation_output', '')}

        STRATEGY OUTPUT:
        {state.get('strategy_output', '')}

        RISK OUTPUT:
        {state.get('risk_output', '')}

        FINANCING OUTPUT:
        {state.get('financing_output', '')}

        POSITIONING OUTPUT:
        {state.get('positioning_output', '')}
        """
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,        # increased from 1024
        temperature=0.2,
    )
    result = response.choices[0].message.content
    return {
        "messages": [AIMessage(content=f"[Final Report]\n{result}")],
    }

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(State)

    g.add_node("input", input_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("sourcing", sourcing_node)
    g.add_node("valuation", valuation_node)
    g.add_node("strategy", strategy_node)
    g.add_node("risk", risk_node)
    g.add_node("financing", financing_node)
    g.add_node("positioning", positioning_node)
    g.add_node("synthesizer", synthesizer_node)   # new

    g.add_edge(START, "input")
    g.add_edge("input", "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        route,
        {
            "sourcing": "sourcing",
            "valuation": "valuation",
            "strategy": "strategy",
            "risk": "risk",
            "financing": "financing",
            "positioning": "positioning",
            "finish": "synthesizer",              # finish now goes to synthesizer
        }
    )

    g.add_edge("sourcing", "orchestrator")
    g.add_edge("valuation", "orchestrator")
    g.add_edge("strategy", "orchestrator")
    g.add_edge("risk", "orchestrator")
    g.add_edge("financing", "orchestrator")
    g.add_edge("positioning", "orchestrator")
    g.add_edge("synthesizer", END)               # synthesizer goes to END

    return g.compile()

graph = build_graph()