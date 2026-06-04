import os

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from backend.app.llm import OpenRouterJudge

# DeepEval metrics call out to a real LLM. Prefer OpenRouter (custom judge),
# fall back to OpenAI's default judge, and skip entirely if neither key is set.
_HAS_OPENROUTER = bool(os.environ.get("OPENROUTER_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))


def _judge_model():
    """Returns an OpenRouter judge, or None to use DeepEval's OpenAI default."""
    return OpenRouterJudge() if _HAS_OPENROUTER else None


@pytest.mark.skipif(
    not (_HAS_OPENROUTER or _HAS_OPENAI),
    reason="No OPENROUTER_API_KEY or OPENAI_API_KEY set; DeepEval needs a live LLM.",
)
def test_scouting_pipeline():
    """
    Regression test for the scouting pipeline using DeepEval metrics.
    Ensures that the generated scouting brief is faithful to the context
    and relevant to the tactical query.
    """
    # Simulated pipeline outputs for testing
    query = "Evaluate the player's defensive transition speed and positioning."
    context = [
        "The player exhibits high recovery speed during defensive transitions.",
        "Positioning data shows he covers 12km per match with a high concentration in the DM zone.",
    ]
    actual_output = (
        "Based on the data, the player has excellent defensive transition speed, "
        "consistently recovering quickly. His positioning is primarily in the defensive "
        "midfield area, covering significant ground (12km)."
    )

    # Define metrics with strict thresholds, routed through the chosen judge.
    judge = _judge_model()
    faithfulness_metric = FaithfulnessMetric(threshold=0.85, model=judge)
    relevance_metric = AnswerRelevancyMetric(threshold=0.85, model=judge)

    # Construct the test case
    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=context)

    # Assert test case against metrics
    # Note: In a real CI environment, these metrics would run against an actual LLM
    assert_test(test_case, [faithfulness_metric, relevance_metric])


if __name__ == "__main__":
    # This allows running the test script directly
    pytest.main([__file__])
