"""Calculate player progression metrics across seasons."""

from typing import Dict, List


def calculate_progression(season_stats: List[Dict]) -> Dict:
    """Calculate progression metrics from multi-season stats.

    Args:
        season_stats: List of stat dicts sorted by season (earliest first)

    Returns:
        {
            'trajectory': [season, season, ...],  # seasons in order
            'yoy_changes': {'goals': 0.25, ...},  # (current - prev) / prev
            'trend': 'improving' | 'declining' | 'stable',
            'momentum': float (current vs 2-year avg)
        }
    """
    if not season_stats or len(season_stats) < 2:
        return {
            "trajectory": [s.get("season") for s in season_stats],
            "yoy_changes": {},
            "trend": None,
            "momentum": None,
        }

    # Extract numeric stats we care about
    stat_keys = ["goals", "assists", "xg", "xag", "prgc", "prgp", "tackles", "interceptions"]

    # Year-over-year changes: current vs previous
    yoy_changes = {}
    current = season_stats[-1]
    prev = season_stats[-2] if len(season_stats) >= 2 else None

    if prev:
        for key in stat_keys:
            curr_val = current.get(key, 0) or 0
            prev_val = prev.get(key, 0) or 0
            if prev_val > 0:
                yoy_changes[key] = (curr_val - prev_val) / prev_val
            elif curr_val > 0:
                yoy_changes[key] = 1.0  # went from 0 to positive = infinite improvement
            else:
                yoy_changes[key] = 0.0

    # Trend: compare current to 2-year average
    momentum = None
    if len(season_stats) >= 2:
        recent_avg = sum((season_stats[-1].get(key, 0) or 0) for key in stat_keys) / len(stat_keys)
        older_avg = sum((season_stats[-2].get(key, 0) or 0) for key in stat_keys) / len(stat_keys)
        if older_avg > 0:
            momentum = (recent_avg - older_avg) / older_avg

    # Overall trend direction
    trend = None
    if yoy_changes:
        avg_yoy = sum(yoy_changes.values()) / len(yoy_changes)
        if avg_yoy > 0.05:
            trend = "improving"
        elif avg_yoy < -0.05:
            trend = "declining"
        else:
            trend = "stable"

    return {
        "trajectory": [s.get("season") for s in season_stats],
        "yoy_changes": yoy_changes,
        "trend": trend,
        "momentum": round(momentum, 3) if momentum is not None else None,
    }


def format_progression(progression: Dict) -> str:
    """Human-readable progression summary."""
    if not progression or not progression.get("trend"):
        return "Limited history"

    trend = progression["trend"]
    momentum = progression.get("momentum") or 0

    if trend == "improving":
        if momentum > 0.2:
            return f"Rapidly improving (+{momentum*100:.0f}%)"
        return f"Improving (+{momentum*100:.0f}%)"
    elif trend == "declining":
        if momentum < -0.2:
            return f"Rapidly declining ({momentum*100:.0f}%)"
        return f"Declining ({momentum*100:.0f}%)"
    else:
        return "Stable form"
