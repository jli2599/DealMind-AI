# graph.py

from __future__ import annotations

from typing import Annotated, TypedDict
import json
import time

from huggingface_hub import InferenceClient
from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import os

from sourcing_agent import sourcing_subgraph, extract_json

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
''' HuggingFace Version
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
'''
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    time.sleep(2)
    max_retries = 5
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait = retry_delay * (attempt + 1)
                    print(f"[Rate limit hit] Waiting {wait}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")
            else:
                raise

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# Add "next" to State so the orchestrator can signal which agent to call
class State(TypedDict, total=False):
    user_message: str
    messages: Annotated[list, add_messages]
    next: str
    completed_agents: list
    critic_feedback: dict        # critic: orchestrator instructions
    iteration: int               # tracks how many loops we've done
    critic_approved: bool        # critic: route decision
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

#Replaced with subagent
''' 
SOURCING_SYSTEM = """You are the Sourcing Agent for DealMind AI.
Identify 8 credible buyer candidates (mix of strategic buyers and financial sponsors).

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "buyers": [
    {
      "name": "<buyer name>",
      "type": "<Strategic or Sponsor>",
      "hq": "<city, country>",
      "rationale": "<1 sentence>",
      "fit_score": <1-10>
    }
  ]
}"""
'''

VALUATION_SYSTEM = """You are the Valuation Agent for DealMind AI.

If specific financial data is provided, use it directly.
If financial data is missing, use industry benchmarks for the sector and explicitly state your assumptions in the value_drivers field. Never return a range without justification.

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "ev_revenue_multiple": {"low": "<x>x", "high": "<x>x"},
  "ev_ebitda_multiple": {"low": "<x>x", "high": "<x>x"},
  "implied_ev": {"low": "<$XM>", "high": "<$XM>"},
  "value_drivers": ["<specific driver or assumption with reasoning>", "<specific driver or assumption with reasoning>", "<specific driver or assumption with reasoning>"]
}"""

STRATEGY_SYSTEM = """You are the Strategy Agent for DealMind AI.

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "synergies": [
    {"type": "<Revenue/Cost/Technology>", "description": "<1 sentence>"}
  ],
  "integration_complexity": "<Low/Medium/High>",
  "integration_rationale": "<1 sentence>",
  "competitive_positioning": "<1-2 sentences>"
}"""

RISK_SYSTEM = """You are the Risk Agent for DealMind AI.

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "risks": [
    {
      "risk": "<risk name>",
      "category": "<Regulatory/Execution/Market/Technology>",
      "rating": "<Low/Medium/High>",
      "mitigation": "<1 sentence>"
    }
  ]
}"""

FINANCING_SYSTEM = """You are the Financing Agent for DealMind AI.

If deal size is unknown, estimate it based on available company data and state your assumption in recommended_structure. Never return an error — always produce a complete JSON object even if figures are estimated.

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "deal_structure": "<merger/asset sale/carve-out>",
  "consideration_mix": {
    "cash": "<x%>",
    "stock": "<x%>",
    "earnout": "<x%>"
  },
  "leverage_capacity": "<1 sentence including any assumptions made>",
  "recommended_structure": "<1-2 sentences including any estimates or assumptions>"
}"""

POSITIONING_SYSTEM = """You are the Positioning Agent for DealMind AI.

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "growth_narrative": "<2-3 sentences>",
  "process_type": "<broad auction/targeted>",
  "process_justification": "<1 sentence>",
  "materials_needed": ["<material 1>", "<material 2>", "<material 3>"]
}"""

SYNTHESIZER_SYSTEM = """You are the Synthesizer for DealMind AI.
You will receive structured JSON outputs from 6 specialist M&A agents.
Combine them into a polished final report in clean markdown.

Your report must include:

## Executive Summary
3-4 sentences summarising the deal and key findings.

## Ranked Buyer List
Rank the top 5 buyers by fit score. For each include:
- Rank, Name, Type, Fit Score
- Strategic rationale (2-3 sentences)
- Recommended outreach approach

## Valuation Summary
EV/Revenue and EV/EBITDA multiples, implied EV range, top value drivers.

## Risk Overview
Top 3 risks, ratings, and mitigations.

## Financing & Deal Structure
Recommended structure, consideration mix, leverage capacity.

## Process Recommendation
Process type, growth narrative, timeline, materials needed.

Be specific. Pull directly from the data provided."""

CRITIC_SYSTEM = """You are the Critic Agent for DealMind AI.
Review the outputs from 6 specialist M&A agents.

ONLY fail an agent if ANY of the following hard failures are present:
- The output contains an "error" key
- A required field is null, missing, or an empty string
- A list field (buyers, risks, synergies) has fewer than 3 items
- fit_score is missing or not a number

Do NOT fail an agent for:
- Subjective quality issues (e.g. "could be more detailed")
- Missing HQ when the data was not available
- Ranges that are justified with reasoning
- Generic language if the core fields are populated

Reply ONLY with a JSON object in this exact format, nothing else:
{
  "approved": <true or false>,
  "feedback": {
    "sourcing": "<'pass' or specific hard failure only>",
    "valuation": "<'pass' or specific hard failure only>",
    "strategy": "<'pass' or specific hard failure only>",
    "risk": "<'pass' or specific hard failure only>",
    "financing": "<'pass' or specific hard failure only>",
    "positioning": "<'pass' or specific hard failure only>"
  }
}

Approve if all hard failures above are absent. Be lenient on quality."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_node(state: State) -> dict:
    time.sleep(3)
    return {"messages": [HumanMessage(content=state["user_message"])]}

def orchestrator_node(state: State) -> dict:
    completed = state.get("completed_agents", [])
    feedback = state.get("critic_feedback", {})

    prompt = f"""User request: {state["user_message"]}

        Agents already completed satisfactorily: {completed if completed else "none yet"}
        Critic feedback for improvement: {json.dumps(feedback, indent=2) if feedback else "none"}

        Pick the next agent that has NOT been completed yet, or finish if all six are done."""

    raw = call_llm(ORCHESTRATOR_SYSTEM, prompt)
    parsed = extract_json(raw)
    next_agent = parsed.get("agent", "finish")
    brief = parsed.get("brief", {})

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
    result = sourcing_subgraph.invoke({"brief": brief})
    buyers = result.get("buyers", {"error": "no output"})
    completed = state.get("completed_agents", [])
    return {
        "sourcing_output": json.dumps(buyers, indent=2),
        "completed_agents": completed + ["sourcing"],
        "messages": [AIMessage(content=f"[Sourcing]\n{json.dumps(buyers, indent=2)}")],
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
    raw = call_llm(VALUATION_SYSTEM, user_prompt)
    result = extract_json(raw)
    completed = state.get("completed_agents", [])
    return {
        "valuation_output": json.dumps(result, indent=2),
        "completed_agents": completed + ["valuation"],
        "messages": [AIMessage(content=f"[Valuation]\n{json.dumps(result, indent=2)}")],
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
    raw = call_llm(STRATEGY_SYSTEM, user_prompt)
    result = extract_json(raw)
    completed = state.get("completed_agents", [])
    return {
        "strategy_output": json.dumps(result, indent=2),
        "completed_agents": completed + ["strategy"],
        "messages": [AIMessage(content=f"[Strategy]\n{json.dumps(result, indent=2)}")],
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
    raw = call_llm(RISK_SYSTEM, user_prompt)
    result = extract_json(raw)
    completed = state.get("completed_agents", [])
    return {
        "risk_output": json.dumps(result, indent=2),
        "completed_agents": completed + ["risk"],
        "messages": [AIMessage(content=f"[Risk]\n{json.dumps(result, indent=2)}")],
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
    raw = call_llm(FINANCING_SYSTEM, user_prompt)
    result = extract_json(raw)
    completed = state.get("completed_agents", [])
    return {
        "financing_output": json.dumps(result, indent=2),
        "completed_agents": completed + ["financing"],
        "messages": [AIMessage(content=f"[Financing]\n{json.dumps(result, indent=2)}")],
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
    raw = call_llm(POSITIONING_SYSTEM, user_prompt)
    result = extract_json(raw)
    completed = state.get("completed_agents", [])
    return {
        "positioning_output": json.dumps(result, indent=2),
        "completed_agents": completed + ["positioning"],
        "messages": [AIMessage(content=f"[Positioning]\n{json.dumps(result, indent=2)}")],
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
    result = call_llm(SYNTHESIZER_SYSTEM, prompt, max_tokens=4096)
    return {
        "messages": [AIMessage(content=f"[Final Report]\n{result}")],
    }

def critic_node(state: State) -> dict:
    iteration = state.get("iteration", 0)

    if iteration >= 2:
        return {
            "critic_approved": True,
            "critic_feedback": {},
            "iteration": iteration + 1,
            "messages": [AIMessage(content="[Critic] Max iterations reached. Approving.")],
        }

    previous_feedback = state.get("critic_feedback", {})
    already_passed = [agent for agent, note in previous_feedback.items() if note == "pass"]

    prompt = f"""
        Company being analysed: {state.get('user_message', '')}

        Agents that already passed review and must be marked "pass" again: {already_passed}

        Only re-evaluate agents NOT in the above list:
        SOURCING OUTPUT: {state.get('sourcing_output', 'missing')}
        VALUATION OUTPUT: {state.get('valuation_output', 'missing')}
        STRATEGY OUTPUT: {state.get('strategy_output', 'missing')}
        RISK OUTPUT: {state.get('risk_output', 'missing')}
        FINANCING OUTPUT: {state.get('financing_output', 'missing')}
        POSITIONING OUTPUT: {state.get('positioning_output', 'missing')}

        For agents in the already passed list, set their feedback to "pass" without re-evaluating.
        Only strictly evaluate the agents NOT in the already passed list.
        """
    raw = call_llm(CRITIC_SYSTEM, prompt)
    parsed = extract_json(raw)
    approved = parsed.get("approved", False)
    feedback = parsed.get("feedback", {})

    # Force passing agents to stay as pass regardless of LLM output
    for agent in already_passed:
        feedback[agent] = "pass"

    failed_agents = [agent for agent, note in feedback.items() if note != "pass"]
    passing_agents = [agent for agent, note in feedback.items() if note == "pass"]

    current_completed = state.get("completed_agents", [])
    updated_completed = [a for a in current_completed if a not in failed_agents]
    for agent in passing_agents:
        if agent not in updated_completed:
            updated_completed.append(agent)

    return {
        "critic_approved": approved,
        "critic_feedback": feedback,
        "completed_agents": updated_completed,
        "iteration": iteration + 1,
        "messages": [AIMessage(content=f"[Critic] Approved: {approved}\nFailed: {failed_agents}\nPassing: {passing_agents}\n{json.dumps(feedback, indent=2)}")],
    }

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def route_critic(state: State) -> str:
    return "synthesizer" if state.get("critic_approved") else "orchestrator"

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
    g.add_node("critic", critic_node)
    g.add_node("synthesizer", synthesizer_node)

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
            "finish": "critic",         # finish now goes to critic first
        }
    )

    g.add_edge("sourcing", "orchestrator")
    g.add_edge("valuation", "orchestrator")
    g.add_edge("strategy", "orchestrator")
    g.add_edge("risk", "orchestrator")
    g.add_edge("financing", "orchestrator")
    g.add_edge("positioning", "orchestrator")

    # Critic either approves → synthesizer or rejects → orchestrator
    g.add_conditional_edges(
        "critic",
        route_critic,
        {
            "synthesizer": "synthesizer",
            "orchestrator": "orchestrator",
        }
    )

    g.add_edge("synthesizer", END)

    return g.compile()

graph = build_graph()