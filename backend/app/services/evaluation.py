"""DeepEval integration for scouting brief quality assessment."""

import os
from typing import List
from fastapi import HTTPException
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from backend.app.engine import _lm_name


def evaluate_scouting_brief(
    query: str, scouting_brief: str, context: List[str]
) -> dict:
    """Score a scouting brief with DeepEval metrics.

    Measures Faithfulness (grounded in retrieved context) and Answer Relevancy
    (addresses the tactical query). Slow (several LLM calls) — call on demand.
    """
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise HTTPException(status_code=503, detail="No LLM key for the eval judge.")
    if not context:
        raise HTTPException(status_code=422, detail="No retrieval context to evaluate against.")

    judge = None
    if os.environ.get("OPENROUTER_API_KEY"):
        from backend.app.llm import OpenRouterJudge

        judge = OpenRouterJudge()

    threshold = 0.7
    test_case = LLMTestCase(
        input=query,
        actual_output=scouting_brief,
        retrieval_context=context,
    )

    metric_specs = [
        (
            "faithfulness",
            "Brief quality",
            FaithfulnessMetric(threshold=threshold, model=judge, async_mode=False),
        ),
        (
            "answer_relevancy",
            "Brief quality",
            AnswerRelevancyMetric(threshold=threshold, model=judge, async_mode=False),
        ),
        (
            "contextual_relevancy",
            "Retrieval quality",
            ContextualRelevancyMetric(threshold=threshold, model=judge, async_mode=False),
        ),
    ]

    results = {}
    for key, kind, metric in metric_specs:
        try:
            metric.measure(test_case)
            results[key] = {
                "score": round(float(metric.score or 0.0), 3),
                "reason": metric.reason,
                "passed": bool(metric.is_successful()),
                "kind": kind,
            }
        except Exception as e:
            results[key] = {
                "score": None,
                "reason": f"evaluation failed: {e}",
                "passed": None,
                "kind": kind,
            }

    scores = [r["score"] for r in results.values() if r["score"] is not None]
    overall = round(sum(scores) / len(scores), 3) if scores else None

    return {
        "metrics": results,
        "overall": overall,
        "threshold": threshold,
        "judge": _lm_name(),
    }
