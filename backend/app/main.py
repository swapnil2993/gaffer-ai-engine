import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from backend.app.config import setup_cors
from backend.app.database import (
    BLENDED_COLLECTION,
    EMBEDDING_MODEL_NAME,
    STATS_COLLECTION,
    THEORY_COLLECTION,
    CAREER_COLLECTION,
    VECTOR_DIM,
    count_entities,
    init_collections,
)
from backend.app.engine import ScoutIntelRAG, _lm_name, init_dspy, is_dspy_ready
from backend.app.schemas import EvalRequest
from backend.app.services.theory import build_theory_html
from backend.app.services.evaluation import evaluate_scouting_brief
from backend.app.tactical_reference import get_system_info as get_tactical_reference_info

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
            "player_profiles_blended": {
                "name": BLENDED_COLLECTION,
                "rows": count_entities(BLENDED_COLLECTION),
                "status": "ACTIVE ✅",
                "description": "Unified collection: season-specific + career metrics in one record",
                "coverage": "1,466 blended records (1,219 players × 1-2 seasons)",
                "architecture": "One query returns both current form (season-specific) AND 3-year trends (career)",
                "benefits": [
                    "DeepEval gets complete context (season + career)",
                    "Hybrid queries work naturally (improving AND sharp now)",
                    "Simplified retrieval pipeline",
                    "Consistent position across seasons (from 2023/24 reference)"
                ]
            },
            "legacy_collections": {
                "note": "Preserved for backward compatibility, not actively used",
                "player_stats": {
                    "name": STATS_COLLECTION,
                    "rows": count_entities(STATS_COLLECTION),
                    "description": "[LEGACY] Per-player season stats",
                    "status": "Preserved, not used"
                },
                "player_career": {
                    "name": CAREER_COLLECTION,
                    "rows": count_entities(CAREER_COLLECTION),
                    "description": "[LEGACY] Multi-season career profiles",
                    "status": "Preserved, not used"
                },
            },
        },
        "tactical_reference": get_tactical_reference_info(),
        "seasons": ["2023/2024", "2024/2025", "2025/2026"],
        "data_sources": {
            "player_stats_23-24.csv": "1,019 players with defensive stats (Tkl, Int) — position authoritative source",
            "fbref_PL_2024-25.csv": "574 players with full season metrics",
            "merged_25_26.csv": "573 players from merged stats + defensive data",
            "player_wages.csv": "562 players with annual/weekly wages",
            "managers.csv": "Manager profiles and playing styles",
        },
        "position_granularity": {
            "defenders": "CB (centre-back), LB (left-back), RB (right-back), LWB, RWB",
            "midfielders": "CM (central), DM (defensive), AM (attacking)",
            "forwards": "ST (striker), IF (inside forward), W (winger)",
            "inference": "From authoritative 2023/24 reference, consistent across all seasons",
        },
        "llm": {"ready": is_dspy_ready(), "model": _lm_name()},
        "pipeline": [
            "1. Embed query (384-d vector space)",
            "2. Understand query (position, player, club, stats from wording)",
            "3. Retrieve tactical theory (Index A: semantic + tactical concepts)",
            "4. Retrieve candidates (Index B: blended collection, hybrid scalar+vector)",
            "   └─ Optional season filter for current form, or no filter for 3-year analysis",
            "5. Rank by stats (blend 50/50 cosine similarity + per-90 metrics)",
            "6. Enrich (manager, wages, career metrics from single record)",
            "7. Project to 2D (PCA similarity visualization)",
            "8. Generate brief (DSPy ChainOfThought with complete context)",
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
):
    """
    Endpoint for semantic football queries.

    Position, player name, and club (scouting_for) are inferred from the query text
    automatically via Phase 2 (Query Understanding). No explicit parameters needed.

    Optional:
    - `season`: Override the detected season (default: latest available).
      Format: "2024/2025", "2025/2026", etc.

    All filters are relaxed automatically if they match nothing (see retrieval.filter_used).
    """
    if not is_dspy_ready():
        raise HTTPException(
            status_code=503,
            detail=("Language model not configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY and restart."),
        )
    try:
        result = scout_engine(query_str, season)
        return {
            # QUERY CONTEXT
            "query": {
                "tactical_query": query_str,
                "season_override": season,
                "inferred_from_query": {
                    "note": "Position, player name, and club are extracted from query text in Phase 2"
                },
            },

            # SCOUTING BRIEF - Pure narrative (no candidate cards)
            "brief": {
                "narrative": result["scouting_brief"],
                "reasoning": result["reasoning"],
            },

            # RETRIEVED & ENRICHED CANDIDATES - Full structured data
            "candidates": {
                "retrieved_and_enriched": result["candidates"],
                "count": len(result["candidates"]),
            },

            # RETRIEVAL & RANKING ANALYSIS
            "analysis": {
                "retrieval_strategy": result["retrieval"],
                "similarity_plot": result.get("similarity_plot"),
            },

            # EXPLAINABILITY - All 15 improvements
            "explainability": result.get("explainability", {}),

            # CONTEXT USED
            "context_used": result["context_used"],

            # 8-PHASE PIPELINE TRACE
            "phases": result["phases"],

            # DATA NOTES
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


