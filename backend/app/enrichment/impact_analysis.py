"""Analyze and surface the impact of career data on scouting decisions."""

from typing import Dict, List, Any, Optional


def extract_impact_metrics(candidate: Dict, is_career_query: bool = False) -> Dict[str, Any]:
    """Extract the key metrics that drove the recommendation.

    This surfaces:
    - Why this player was chosen (matching metrics)
    - Career context (if available)
    - Risk/confidence indicators
    - Comparative strengths

    Args:
        candidate: Enriched player profile with stats + career data
        is_career_query: Whether the original query asked about career/progression

    Returns:
        Dict with impact_drivers, confidence, risk_factors, and career_context
    """
    impact = {
        "player_name": candidate.get("player_name", "Unknown"),
        "impact_drivers": [],
        "career_context": None,
        "risk_factors": [],
        "confidence_level": "medium",
    }

    # Extract top-performing stats (normalized, in top 3 buckets)
    stats = candidate.get("stats", {})
    strong_stats = []
    for key, value in stats.items():
        if value and isinstance(value, (int, float)) and value > 0:
            # Highlight stats that are notably non-zero
            if key in ["goals", "assists", "tackles", "interceptions", "recoveries"]:
                if value > 20:  # Arbitrary threshold for "strong"
                    strong_stats.append((key, value))

    if strong_stats:
        sorted_stats = sorted(strong_stats, key=lambda x: x[1], reverse=True)[:3]
        for stat, value in sorted_stats:
            stat_label = stat.replace("_", " ").title()
            impact["impact_drivers"].append({
                "metric": stat_label,
                "value": int(value) if isinstance(value, float) else value,
                "source": "this season",
            })

    # Extract career context if available
    progression = candidate.get("progression", {})
    if progression and progression.get("trend"):
        trend = progression["trend"]
        momentum = progression.get("momentum", 0)
        consistency = progression.get("yoy_changes", {})

        impact["career_context"] = {
            "trend": trend,
            "momentum": momentum,
            "best_season": progression.get("trajectory", [])[-1] if progression.get("trajectory") else None,
            "consistency": describe_consistency(consistency),
        }

        # Update confidence based on trajectory
        if trend == "improving" and momentum and momentum > 0.05:
            impact["confidence_level"] = "high"  # Improving players are lower risk
        elif trend == "declining" and momentum and momentum < -0.05:
            impact["risk_factors"].append("Downward trajectory (declining form)")
            impact["confidence_level"] = "low"

    # Add season comparison if career data exists
    if impact["career_context"]:
        current_stats = stats
        career_avg = candidate.get("aggregates", {})
        if career_avg:
            for key in ["goals", "assists", "tackles"]:
                current_val = current_stats.get(key, 0)
                career_val = career_avg.get(key, {}).get("avg", 0)
                if career_val > 0:
                    diff_pct = ((current_val - career_val) / career_val) * 100
                    if diff_pct > 10:
                        impact["impact_drivers"].append({
                            "metric": f"{key.title()} (vs career avg)",
                            "value": f"{diff_pct:+.0f}%",
                            "source": "this season vs 3-year avg",
                        })

    # Manager/team fit
    manager = candidate.get("current_manager")
    if manager and manager != "manager unknown":
        impact["impact_drivers"].append({
            "metric": "Manager Fit",
            "value": f"Playing under {manager}",
            "source": "current assignment",
        })

    return impact


def describe_consistency(yoy_changes: Dict[str, float]) -> str:
    """Describe consistency from year-over-year changes.

    Args:
        yoy_changes: Dict of {stat: pct_change}

    Returns:
        Consistency description (very consistent, variable, etc.)
    """
    if not yoy_changes:
        return "limited data"

    changes = [abs(v) for v in yoy_changes.values() if v is not None]
    if not changes:
        return "unknown"

    avg_change = sum(changes) / len(changes)
    if avg_change < 0.1:
        return "very consistent"
    elif avg_change < 0.25:
        return "consistent"
    elif avg_change < 0.4:
        return "variable"
    else:
        return "highly variable"


def build_ui_response_envelope(
    candidates: List[Dict],
    reasoning: str,
    brief: str,
    is_career_query: bool = False,
) -> Dict[str, Any]:
    """Build a response envelope with impact metrics for UI consumption.

    Args:
        candidates: Top candidate players with all enrichment
        reasoning: DSPy reasoning bullets
        brief: Final scouting brief
        is_career_query: Whether this was a career/progression query

    Returns:
        Dict with recommendation + impact analysis for UI
    """
    return {
        "recommendation": {
            "primary": candidates[0]["player_name"] if candidates else None,
            "runner_up": candidates[1]["player_name"] if len(candidates) > 1 else None,
        },
        "brief": brief,
        "reasoning": reasoning,
        "impact": {
            "primary": extract_impact_metrics(candidates[0], is_career_query) if candidates else None,
            "runner_up": extract_impact_metrics(candidates[1], is_career_query) if len(candidates) > 1 else None,
        },
        "query_type": "career_progression" if is_career_query else "season_form",
        "candidates_considered": len(candidates),
    }
