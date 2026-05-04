# sourcing_agent.py

from __future__ import annotations

import json
import os
import time

from groq import Groq
from tavily import TavilyClient

from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

llm = Groq(api_key=os.environ["GROQ_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    time.sleep(2)
    max_retries = 5
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            response = llm.chat.completions.create(
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
                    print(f"[Sourcing: Rate limit hit] Waiting {wait}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")
            else:
                raise

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = 5) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = tavily.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            output = []
            for r in results.get("results", []):
                output.append(f"- {r.get('title', '')}: {r.get('content', '')[:400]}")
            return "\n".join(output) if output else "No results found."
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"[Sourcing: Tavily rate limit] Waiting {wait}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                else:
                    return "Search failed: rate limit exceeded"
            else:
                return f"Search failed: {str(e)}"

# ---------------------------------------------------------------------------
# Sourcing agent prompts
# ---------------------------------------------------------------------------

QUERY_BUILDER_SYSTEM = """You are a research assistant for an M&A team.
    Given a company profile, generate 3 targeted web search queries to find:
    1. Specific companies that have recently acquired businesses in this exact sector
    2. Named strategic buyers or acquirers active in this space with deal examples
    3. Named private equity firms that have completed investments in this sector

    Keep queries short and specific — 5 to 8 words max. Do not use boolean operators like AND/OR.

    Reply ONLY with a JSON object in this exact format, nothing else:
    {
    "queries": [
        "<query 1>",
        "<query 2>",
        "<query 3>"
    ]
    }"""

BUYER_SYNTHESIS_SYSTEM = """You are the Sourcing Agent for DealMind AI.
    You have been given a company profile and real web search results about recent M&A activity in this sector.
    Use this data to identify 8 credible buyer candidates grounded in the search results.
    Mix strategic buyers and financial sponsors. Prioritise buyers mentioned in the search results.

    Reply ONLY with a JSON object in this exact format, nothing else:
    {
    "buyers": [
        {
        "name": "<buyer name>",
        "type": "<Strategic or Sponsor>",
        "hq": "<city, country>",
        "rationale": "<1 sentence — reference specific strategic fit or recent activity>",
        "fit_score": <1-10>,
        "source": "<'web search' if found in results, 'industry knowledge' if inferred>"
        }
    ]
    }"""

SELF_CHECK_SYSTEM = """You are a quality checker for a sourcing agent.
    You will receive a JSON output from a buyer synthesis step.
    Check if the output is valid and complete.

    A valid output:
    - Has a "buyers" key with a list of at least 5 buyers
    - Each buyer has: name, type, hq, rationale, fit_score, source
    - No "error" key present
    - fit_score is a number between 1-10

    If valid, return the output unchanged.
    If invalid or has an error key, fix it and return a corrected version.

    Reply ONLY with a valid JSON object with a "buyers" key, nothing else."""

# ---------------------------------------------------------------------------
# Subgraph state
# ---------------------------------------------------------------------------

class SourcingState(TypedDict, total=False):
    brief: dict
    queries: list
    search_results: str
    buyers: dict
    self_check_feedback: str      # feedback for buyer_synthesis on retry
    retry_count: int              # prevent infinite loops
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Subgraph nodes
# ---------------------------------------------------------------------------

def query_builder_node(state: SourcingState) -> dict:
    brief = state["brief"]
    profile = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        Task: {brief.get('task', '')}
        """
    raw = call_llm(QUERY_BUILDER_SYSTEM, profile)
    parsed = extract_json(raw)
    queries = parsed.get("queries", [
        f"recent acquisitions {brief.get('sector', '')} 2024 2025",
        f"strategic buyers {brief.get('sector', '')} M&A",
        f"private equity {brief.get('sector', '')} investments",
    ])
    return {
        "queries": queries,
        "messages": [AIMessage(content=f"[Sourcing: Query Builder]\n{queries}")],
    }

def web_search_node(state: SourcingState) -> dict:
    queries = state.get("queries", [])
    search_results = []
    for query in queries:
        result = search_web(query)
        search_results.append(f"Query: {query}\nResults:\n{result}")
        time.sleep(1)
    combined = "\n\n".join(search_results)
    return {
        "search_results": combined,
        "messages": [AIMessage(content=f"[Sourcing: Web Search]\n{combined[:500]}...")],
    }

def buyer_synthesis_node(state: SourcingState) -> dict:
    brief = state["brief"]
    profile = f"""
        Company: {brief.get('company', 'Unknown')}
        Sector: {brief.get('sector', 'Unknown')}
        Deal Size: {brief.get('deal_size', 'Unknown')}
        Key Facts: {', '.join(brief.get('key_facts', []))}
        """
    search_results = state.get('search_results', '')[:3000]
    feedback = state.get('self_check_feedback', '')

    synthesis_prompt = f"""
        Company Profile:
        {profile}

        Web Search Results:
        {search_results}

        {"Previous attempt failed. Feedback: " + feedback if feedback else ""}

        Using the above search results, identify 8 credible buyers for this company.
        """
    raw = call_llm(BUYER_SYNTHESIS_SYSTEM, synthesis_prompt, max_tokens=2048)
    result = extract_json(raw)

    return {
        "buyers": result,
        "messages": [AIMessage(content=f"[Sourcing: Synthesis]\n{json.dumps(result, indent=2)}")],
    }

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output that may contain markdown fences."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences anywhere in the string
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { and last } and try to parse between them
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    return {"error": raw}

def self_check_node(state: SourcingState) -> dict:
    buyers = state.get("buyers", {})
    retry_count = state.get("retry_count", 0)

    has_error = "error" in buyers
    buyer_list = buyers.get("buyers", [])
    has_enough = len(buyer_list) >= 5
    all_valid = all(
        b.get("name") and b.get("type") and b.get("rationale") and b.get("fit_score")
        for b in buyer_list
    )

    passed = not has_error and has_enough and all_valid

    if passed:
        return {
            "self_check_feedback": "",
            "messages": [AIMessage(content="[Sourcing: Self Check] Passed.")],
        }

    # Build specific feedback
    issues = []
    if has_error:
        issues.append("output contained an error key — JSON parsing failed")
    if not has_enough:
        issues.append(f"only {len(buyer_list)} buyers returned, need at least 5")
    if not all_valid:
        issues.append("some buyers are missing required fields (name, type, rationale, fit_score)")

    feedback = "; ".join(issues)

    return {
        "self_check_feedback": feedback,
        "retry_count": retry_count + 1,
        "messages": [AIMessage(content=f"[Sourcing: Self Check] Failed — {feedback}. Retrying...")],
    }

# ---------------------------------------------------------------------------
# Compiled subgraph — imported by graph.py 
# ---------------------------------------------------------------------------

def route_self_check(state: SourcingState) -> str:
    buyers = state.get("buyers", {})
    retry_count = state.get("retry_count", 0)

    has_error = "error" in buyers
    buyer_list = buyers.get("buyers", [])
    has_enough = len(buyer_list) >= 5
    all_valid = all(
        b.get("name") and b.get("type") and b.get("rationale") and b.get("fit_score")
        for b in buyer_list
    )

    passed = not has_error and has_enough and all_valid

    if passed or retry_count >= 2:
        return "end"
    return "buyer_synthesis"


def build_sourcing_graph():
    g = StateGraph(SourcingState)

    g.add_node("query_builder", query_builder_node)
    g.add_node("web_search", web_search_node)
    g.add_node("buyer_synthesis", buyer_synthesis_node)
    g.add_node("self_check", self_check_node)

    g.add_edge(START, "query_builder")
    g.add_edge("query_builder", "web_search")
    g.add_edge("web_search", "buyer_synthesis")
    g.add_edge("buyer_synthesis", "self_check")

    g.add_conditional_edges(
        "self_check",
        route_self_check,
        {
            "buyer_synthesis": "buyer_synthesis",
            "end": END,
        }
    )

    return g.compile()

sourcing_subgraph = build_sourcing_graph()