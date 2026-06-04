"""Calculate player progression metrics across seasons.

Note: Handles both full seasons (per-appearance) and partial seasons (per-90).
For in-season data (25-26), uses per-90 stats since MP is low.
"""

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
            'momentum': float (current vs 2-year avg),
            'stability_score': float 0-1 (consistency across seasons),
            'improvement_score': float 0-1 (upward trajectory strength),
            'consistency_pct': float 0-100 (variance reduction = stability)
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

    # STABILITY SCORE: How consistent is performance across seasons?
    # Low variance = high stability. Measured by coefficient of variation.
    stability_score = 1.0
    if len(season_stats) >= 2:
        all_stats = []
        for s in season_stats:
            for key in stat_keys:
                val = s.get(key, 0) or 0
                if val > 0:
                    all_stats.append(val)

        if all_stats:
            mean_stat = sum(all_stats) / len(all_stats)
            if mean_stat > 0:
                variance = sum((x - mean_stat) ** 2 for x in all_stats) / len(all_stats)
                std_dev = variance ** 0.5
                coeff_variation = std_dev / mean_stat
                # Lower CV = higher stability. Clip to 0-1 range
                stability_score = max(0, 1 - coeff_variation / 2)

    # IMPROVEMENT SCORE: How strong is the upward trajectory?
    # Positive momentum + improving trend = higher score
    improvement_score = 0.0
    if trend == "improving" and momentum is not None:
        improvement_score = min(1.0, max(0, momentum))  # Clip momentum to 0-1
    elif trend == "improving":
        improvement_score = 0.7  # Some improvement even if momentum unclear

    return {
        "trajectory": [s.get("season") for s in season_stats],
        "yoy_changes": yoy_changes,
        "trend": trend,
        "momentum": round(momentum, 3) if momentum is not None else None,
        "stability_score": round(stability_score, 2),  # 0-1: higher = more consistent
        "improvement_score": round(improvement_score, 2),  # 0-1: higher = more improving
        "consistency_pct": round(stability_score * 100, 0),  # For human readability
    }


def format_progression(progression: Dict) -> str:
    """Human-readable progression summary with explicit stability/improvement."""
    if not progression or not progression.get("trend"):
        return "Limited history"

    trend = progression["trend"]
    momentum = progression.get("momentum") or 0
    stability = progression.get("stability_score", 0)
    improvement = progression.get("improvement_score", 0)

    if trend == "improving":
        if momentum > 0.2:
            return f"Rapidly improving (+{momentum*100:.0f}%) | Improvement score: {improvement:.0%}"
        return f"Improving (+{momentum*100:.0f}%) | Improvement score: {improvement:.0%}"
    elif trend == "declining":
        if momentum < -0.2:
            return f"Rapidly declining ({momentum*100:.0f}%)"
        return f"Declining ({momentum*100:.0f}%)"
    else:
        # For stable players, emphasize stability score
        return f"Stable form ({stability:.0%} consistency) | Stability score: {stability:.0%}"
