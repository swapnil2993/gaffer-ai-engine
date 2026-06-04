import dspy

from backend.app.engine import ScoutingReportSignature, ScoutIntelRAG


def test_scout_intel_rag_structure():
    """Verify that the ScoutIntelRAG module is correctly initialized."""
    rag = ScoutIntelRAG()
    assert isinstance(rag.generate_report, dspy.predict.chain_of_thought.ChainOfThought)
    # In DSPy 3.x, signature is nested within the internal predictor
    assert rag.generate_report.predict.signature is not None


def test_scouting_report_signature():
    """Verify the fields of the ScoutingReportSignature."""
    sig = ScoutingReportSignature
    assert "tactical_context" in sig.input_fields
    assert "player_stats" in sig.input_fields
    assert "tactical_query" in sig.input_fields
    assert "scouting_brief" in sig.output_fields
