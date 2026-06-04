"""Response enrichment modules - explain and contextualize query results."""

from backend.app.enrichment.impact_analysis import (
    extract_impact_metrics,
    build_ui_response_envelope,
)
from backend.app.enrichment.visualization import build_similarity_plot
from backend.app.enrichment.progression import (
    calculate_progression,
    format_progression,
)

__all__ = [
    "extract_impact_metrics",
    "build_ui_response_envelope",
    "build_similarity_plot",
    "calculate_progression",
    "format_progression",
]
