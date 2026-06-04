"""Score candidates against tactical requirements from the query."""

from typing import Dict, List


# Tactical requirements from the reference guide
TACTICAL_REQUIREMENTS = {
    "high_press": {
        "pressing_success": 0.75,
        "tackles_att_3rd": 0.70,
        "interceptions": 0.60,
    },
    "defensive_transition": {
        "tackles": 0.70,
        "interceptions": 0.65,
        "recoveries": 0.75,
    },
    "ball_progression": {
        "prgp": 0.65,
        "prgc": 0.60,
    },
    "creative_playmaking": {
        "assists": 0.50,
        "xag": 0.65,
    },
    "goal_threat": {
        "goals": 0.60,
        "xg": 0.65,
    },
}

# Map stat names to player profile keys
STAT_KEYS = {
    "pressing_success": "succ_press",
    "tackles_att_3rd": "tkl_att_3rd",
    "interceptions": "interceptions",
    "tackles": "tackles",
    "recoveries": "recoveries",
    "prgp": "prgp",
    "prgc": "prgc",
    "assists": "assists",
    "xag": "xag",
    "goals": "goals",
    "xg": "xg",
}


def build_metrics_checklist(tactics: List[str], candidate: Dict) -> str:
    """Build a checklist comparing candidate stats to tactical requirements.

    Args:
        tactics: List of tactical concepts (e.g., ["high_press", "creative_playmaking"])
        candidate: Player profile dict with stats + optional career data

    Returns:
        Formatted checklist for the LLM to use as ground truth
    """
    if not tactics or not candidate:
        return "No tactical requirements to evaluate."

    stats = candidate.get("stats", {})
    lines = []

    for tactic in tactics:
        if tactic not in TACTICAL_REQUIREMENTS:
            continue

        tactic_name = tactic.replace("_", " ").title()
        lines.append(f"\n{tactic_name.upper()}:")

        requirements = TACTICAL_REQUIREMENTS[tactic]
        for metric, threshold in requirements.items():
            # Find the actual stat value
            stat_key = STAT_KEYS.get(metric, metric)
            actual_value = stats.get(stat_key)

            if actual_value is None:
                lines.append(f"  - {metric}: [data not available]")
            else:
                status = "✓" if actual_value >= threshold else "✗"
                pct = f"{actual_value*100:.0f}%" if isinstance(actual_value, float) and actual_value < 2 else f"{actual_value:.2f}"
                lines.append(
                    f"  - {metric} > {threshold}: Player {pct} {status}"
                )

    # Add career context if available
    progression = candidate.get("progression", {})
    if progression and progression.get("trend"):
        lines.append("\nCARIER TRAJECTORY:")
        trend = progression.get("trend", "unknown")
        momentum = progression.get("momentum", 0)
        momentum_str = f"{momentum*100:+.0f}%" if momentum is not None else "N/A"

        if trend == "improving":
            lines.append(f"  - Direction: IMPROVING {momentum_str} YoY ✓")
        elif trend == "declining":
            lines.append(f"  - Direction: DECLINING {momentum_str} YoY ✗")
        else:
            lines.append(f"  - Direction: STABLE {momentum_str} YoY →")

        consistency = progression.get("yoy_changes", {})
        if consistency:
            avg_yoy = sum(consistency.values()) / len(consistency) if consistency else 0
            consistency_pct = f"{abs(avg_yoy)*100:.0f}%"
            lines.append(f"  - Consistency: {progression.get('consistency', 'variable')}")

    return "\n".join(lines)
