"""Generate career aggregate profiles from multi-season player data."""

from typing import Dict, List, Optional


def _per90(value: float, minutes: float) -> float:
    """Calculate per-90 rate, with minimum games guard."""
    if not minutes or minutes < 1:
        return 0.0
    return (value / minutes) * 90.0


def _stat_consistency(values: List[float]) -> str:
    """Describe consistency across seasons (stable, variable, declining, improving)."""
    if len(values) < 2:
        return "limited data"
    non_zero = [v for v in values if v is not None and v > 0]
    if not non_zero:
        return "no data"
    if len(non_zero) == 1:
        return "single season"
    avg = sum(non_zero) / len(non_zero)
    variance = sum((v - avg) ** 2 for v in non_zero) / len(non_zero)
    std_dev = variance ** 0.5
    coeff_var = std_dev / avg if avg > 0 else 0
    if coeff_var < 0.2:
        return "very consistent"
    elif coeff_var < 0.4:
        return "consistent"
    elif coeff_var < 0.6:
        return "variable"
    else:
        return "highly variable"


def aggregate_player_seasons(
    player_name: str, seasons_data: List[Dict]
) -> Optional[Dict]:
    """Aggregate multi-season stats into a career profile.

    Args:
        player_name: Player name
        seasons_data: List of season stat dicts, sorted by season ascending

    Returns:
        Career aggregate dict with averaged stats, trends, best season, etc.
        Returns None if insufficient data (< 2 seasons).
    """
    if not seasons_data or len(seasons_data) < 2:
        return None

    # Determine position and best squad
    position = seasons_data[-1].get("position", "Unknown")
    positions = [s.get("position") for s in seasons_data if s.get("position")]
    position = positions[-1] if positions else "Unknown"

    squads = [s.get("squad") for s in seasons_data if s.get("squad")]
    squad_counts = {}
    for sq in squads:
        squad_counts[sq] = squad_counts.get(sq, 0) + 1
    best_squad = max(squad_counts, key=squad_counts.get) if squad_counts else "Unknown"

    # Extract numeric stats
    stat_keys = [
        "goals",
        "assists",
        "xg",
        "xag",
        "prgc",
        "prgp",
        "minutes",
        "appearances",
        "tackles",
        "tackles_won",
        "interceptions",
        "recoveries",
        "big_chances_created",
        "clean_sheets",
        "saves",
    ]

    aggregates = {}
    for key in stat_keys:
        values = [float(s.get(key) or 0) for s in seasons_data]
        if any(v > 0 for v in values):
            aggregates[key] = {
                "total": sum(values),
                "avg": sum(values) / len(values),
                "best": max(values),
                "worst": min(values),
                "consistency": _stat_consistency(values),
            }

    # Per-90 rates (using minutes)
    rate_stats = {}
    for key in ["goals", "assists", "tackles", "interceptions"]:
        per90_vals = []
        for s in seasons_data:
            val = float(s.get(key) or 0)
            mins = float(s.get("minutes") or 0)
            if mins > 0:
                per90_vals.append(_per90(val, mins))
        if per90_vals:
            rate_stats[f"{key}_per90"] = {
                "avg": sum(per90_vals) / len(per90_vals),
                "best": max(per90_vals),
            }

    # Determine trend and momentum (using progression logic)
    if len(seasons_data) >= 2:
        current = seasons_data[-1]
        prev = seasons_data[-2]
        yoy_change = 0.0
        for key in ["goals", "assists", "tackles", "interceptions"]:
            curr_val = float(current.get(key) or 0)
            prev_val = float(prev.get(key) or 0)
            if prev_val > 0:
                yoy_change += (curr_val - prev_val) / prev_val
        yoy_change /= 4
        if yoy_change > 0.05:
            trend = "improving"
        elif yoy_change < -0.05:
            trend = "declining"
        else:
            trend = "stable"
        momentum = round(yoy_change, 3)
    else:
        trend = "limited data"
        momentum = None

    # Best season
    best_season_idx = 0
    best_season_score = 0
    for idx, season in enumerate(seasons_data):
        score = (
            float(season.get("goals") or 0)
            + float(season.get("assists") or 0)
            + float(season.get("tackles") or 0)
            + float(season.get("interceptions") or 0)
        )
        if score > best_season_score:
            best_season_score = score
            best_season_idx = idx
    best_season = seasons_data[best_season_idx].get("season", "unknown")

    return {
        "player_name": player_name,
        "position": position,
        "best_squad": best_squad,
        "seasons": [s.get("season") for s in seasons_data],
        "aggregates": aggregates,
        "rate_stats": rate_stats,
        "trend": trend,
        "momentum": momentum,
        "best_season": best_season,
        "consistency_overall": _stat_consistency(
            [a.get("avg", 0) for a in aggregates.values()]
        ),
    }


def describe_career_aggregate(agg: Dict, seasons_data: List[Dict]) -> str:
    """Generate prose describing a player's 3-year career profile.

    Args:
        agg: Career aggregate dict from aggregate_player_seasons
        seasons_data: Original season data for context

    Returns:
        Natural-language career summary
    """
    if not agg:
        return ""

    name = agg["player_name"]
    pos = agg["position"]
    position_word = {
        "GK": "goalkeeper",
        "DF": "defender",
        "MF": "midfielder",
        "FW": "forward",
    }.get(pos.split(",")[0].strip(), "player")

    squad = agg["best_squad"]
    seasons = agg["seasons"]
    trend = agg["trend"]
    momentum = agg.get("momentum") or 0

    parts = [
        f"{name} is a {position_word} (primarily {squad}) across {' – '.join(seasons)}."
    ]

    # Goals and assists (career totals + consistency)
    if "goals" in agg["aggregates"]:
        g_info = agg["aggregates"]["goals"]
        g_total = int(g_info["total"])
        g_avg = g_info["avg"]
        g_consistency = g_info["consistency"]
        parts.append(
            f"Scored {g_total} goals in {len(seasons)} seasons ({g_avg:.1f} per season, {g_consistency})."
        )

    if "assists" in agg["aggregates"]:
        a_info = agg["aggregates"]["assists"]
        a_total = int(a_info["total"])
        a_avg = a_info["avg"]
        parts.append(f"Provided {a_total} assists ({a_avg:.1f} per season).")

    # Defensive stats (tackles + interceptions)
    tkl_info = agg["aggregates"].get("tackles", {})
    int_info = agg["aggregates"].get("interceptions", {})
    if tkl_info or int_info:
        defensive = []
        if tkl_info:
            defensive.append(f"{int(tkl_info['total'])} tackles")
        if int_info:
            defensive.append(f"{int(int_info['total'])} interceptions")
        parts.append(f"Defensive: {', '.join(defensive)} across {len(seasons)} seasons.")

    # Progression and momentum
    if trend and trend != "limited data":
        momentum_pct = abs(momentum * 100)
        if trend == "improving":
            parts.append(
                f"Trajectory: improving (+{momentum_pct:.0f}% YoY). "
                f"Best season: {agg['best_season']}."
            )
        elif trend == "declining":
            parts.append(
                f"Trajectory: declining ({momentum_pct:.0f}% YoY). "
                f"Peak: {agg['best_season']}."
            )
        else:
            parts.append(f"Trajectory: stable across seasons. Peak: {agg['best_season']}.")

    # Consistency summary
    consistency = agg["consistency_overall"]
    if consistency and consistency != "limited data":
        parts.append(f"Profile: {consistency} performer across years.")

    return " ".join(parts)
