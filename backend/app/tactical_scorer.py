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
    "creativity": {
        "key_passes": 0.70,
        "assists": 0.50,
        "xag": 0.60,
    },
    "chance_creation": {
        "key_passes": 0.75,
        "assists": 0.60,
        "xag": 0.70,
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
    "key_passes": "key_passes",
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
    has_matching_metrics = False

    for tactic in tactics:
        if tactic not in TACTICAL_REQUIREMENTS:
            continue

        tactic_name = tactic.replace("_", " ").title()
        tactic_lines = [f"\n{tactic_name.upper()}:"]
        tactic_has_data = False

        requirements = TACTICAL_REQUIREMENTS[tactic]
        for metric, threshold in requirements.items():
            # Find the actual stat value
            stat_key = STAT_KEYS.get(metric, metric)
            actual_value = stats.get(stat_key)

            if actual_value is None:
                # Metric not available in dataset — skip silently rather than showing [data not available]
                # This lets the LLM reason with available data without highlighting absences
                continue
            else:
                tactic_has_data = True
                status = "✓" if actual_value >= threshold else "✗"
                # Format the value: percentages as %, absolute numbers as is
                if isinstance(actual_value, float) and actual_value < 2:
                    pct = f"{actual_value*100:.0f}%"
                else:
                    pct = f"{actual_value:.1f}" if isinstance(actual_value, float) else f"{actual_value}"
                tactic_lines.append(
                    f"  - {metric.replace('_', ' ')} (target: {threshold}): {pct} {status}"
                )

        # Only add tactic section if it has matching metrics
        if tactic_has_data:
            lines.extend(tactic_lines)
            has_matching_metrics = True

    # Add career context if available
    progression = candidate.get("progression", {})
    if progression and progression.get("trend"):
        lines.append("\nCAREER TRAJECTORY:")
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

    # If no matching metrics were found, return a helpful message
    if not has_matching_metrics:
        return (
            f"No matching metrics available for {', '.join(tactics)}. "
            "Recommend based on available stats: "
            f"{candidate.get('player_name', 'Player')} | "
            f"{', '.join(f'{k}={v}' for k,v in list(stats.items())[:3])}"
        )

    return "\n".join(lines)
