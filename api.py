from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph
import json

app = FastAPI()

# Allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DealRequest(BaseModel):
    company_description: str
    seller_objectives: list[str]
    deal_parameters: dict

@app.post("/analyse")
def analyse(request: DealRequest):
    # Build the user message from the form inputs
    objectives_str = ", ".join(request.seller_objectives)
    params = request.deal_parameters

    user_message = f"""
        {request.company_description}

        Seller Objectives: {objectives_str}

        Deal Parameters:
        - Target exit: {params.get('target_exit', 'Not specified')}
        - Timeline: {params.get('timeline', 'Not specified')}
        - Equity rollover: {params.get('equity_rollover', 'Not specified')}
        - Deal type: {params.get('deal_type', 'Not specified')}

        Run a full M&A analysis.
        """

    result = graph.invoke({"user_message": user_message.strip()})

    # Extract the final report from messages
    final_report = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and "[Final Report]" in msg.content:
            final_report = msg.content.replace("[Final Report]\n", "")
            break

    return {
        "final_report": final_report,
        "sourcing_output": result.get("sourcing_output", ""),
        "valuation_output": result.get("valuation_output", ""),
        "strategy_output": result.get("strategy_output", ""),
        "risk_output": result.get("risk_output", ""),
        "financing_output": result.get("financing_output", ""),
        "positioning_output": result.get("positioning_output", ""),
    }

@app.get("/health")
def health():
    return {"status": "ok"}