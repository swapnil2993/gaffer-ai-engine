import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from backend.app.config import setup_cors
from backend.app.database import (
    EMBEDDING_MODEL_NAME,
    STATS_COLLECTION,
    THEORY_COLLECTION,
    VECTOR_DIM,
    count_entities,
    init_collections,
)
from backend.app.engine import ScoutIntelRAG, _lm_name, init_dspy, is_dspy_ready
from backend.app.schemas import EvalRequest
from backend.app.services.theory import build_theory_html
from backend.app.services.evaluation import evaluate_scouting_brief

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup ordering matters:

    init_collections() forks Milvus Lite's embedded server. That fork MUST happen
    before init_dspy() — dspy/litellm (and torch) spawn threads, and forking after
    threads exist crashes natively (SIGSEGV) on macOS. So: DB first, LLM second.
    """
    init_collections()
    init_dspy()
    yield


app = FastAPI(title="Gaffer AI Engine API", version="1.0.0", lifespan=lifespan)
setup_cors(app)

scout_engine = ScoutIntelRAG()


@app.get("/")
async def root():
    return {"message": "Welcome to the Gaffer AI Engine"}


@app.get("/system")
async def system_status():
    """Describe the system so the UI can show the indexing/model layers."""
    return {
        "vector_store": "Milvus Lite (embedded)",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_dim": VECTOR_DIM,
        "collections": {
            "player_stats": {
                "name": STATS_COLLECTION,
                "rows": count_entities(STATS_COLLECTION),
                "description": "Per-player season stats (Index B)",
            },
            "tactical_theory": {
                "name": THEORY_COLLECTION,
                "rows": count_entities(THEORY_COLLECTION),
                "description": "Chunks of 'The Inverted Pyramid' (Index A)",
            },
        },
        "seasons": ["2023/2024", "2024/2025", "2025/2026"],
        "llm": {"ready": is_dspy_ready(), "model": _lm_name()},
        "pipeline": [
            "Embed query",
            "Retrieve theory (Index A)",
            "Retrieve players (Index B)",
            "Enrich (manager + cost)",
            "Project similarity (PCA)",
            "DSPy ChainOfThought brief",
        ],
    }


@app.get("/theory/document", response_class=HTMLResponse)
async def theory_document():
    """Render the tactical theory with full enriched metadata as HTML."""
    return HTMLResponse(build_theory_html())


@app.post("/query")
async def search_scouting_reports(
    query_str: str,
    season: str = None,
    position: str = None,
    player_name: str = None,
    scouting_for: str = None,
):
    """
    Endpoint for semantic football queries.

    All filters are optional. Provide `player_name` to evaluate a specific player;
    `season`/`position` narrow the candidate pool. Filters are relaxed
    automatically if they match nothing (see `retrieval.filter_used`).
    """
    if not is_dspy_ready():
        raise HTTPException(
            status_code=503,
            detail=("Language model not configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY and restart."),
        )
    try:
        result = scout_engine(query_str, season, position, player_name, scouting_for=scouting_for)
        return {
            "tactical_query": query_str,
            "filters": {
                "season": season,
                "position": position,
                "player_name": player_name,
                "scouting_for": scouting_for,
            },
            "retrieval": result["retrieval"],
            "phases": result["phases"],
            "similarity_plot": result.get("similarity_plot"),
            "scouting_brief": result["scouting_brief"],
            "reasoning": result["reasoning"],
            "candidates": result["candidates"],
            "context_used": result["context_used"],
            "notes": {
                "manager": "Manager and playing style are season-aware values from managers.csv.",
                "cost": "Cost reflects current wages; transfer fees are not in the dataset.",
            },
        }
    except Exception as e:
        # Log the full traceback server-side; the HTTP detail is just the message.
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/evaluate")
def evaluate_brief(req: EvalRequest):
    """Score a scouting brief with DeepEval metrics.

    Measures Faithfulness (grounded in context, no invented facts) and Answer
    Relevancy (addresses the tactical query). Slow — call on demand.
    Uses OpenRouter judge when OPENROUTER_API_KEY is set, else DeepEval default.
    """
    return evaluate_scouting_brief(req.query, req.scouting_brief, req.context)


