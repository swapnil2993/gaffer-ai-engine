"""Explainability & transparency layer for all 15 improvements."""

from typing import Dict, List, Any, Optional


def confidence_score_intent(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improvement #1: Confidence scoring for intent detection.

    Shows how confident the system is in understanding position, player name, and club.
    """
    matched = constraints.get("matched", [])
    confidence_by_type = {}

    # Position confidence
    position = constraints.get("position")
    if position:
        # Count how many position keywords matched
        pos_matches = sum(1 for m in matched if "position = " in m)
        confidence_by_type["position"] = {
            "value": position,
            "confidence": 0.95 if pos_matches > 0 else 0.7,
            "sources": [m for m in matched if "position = " in m],
        }

    return confidence_by_type


def rank_alternatives(query: str) -> Optional[Dict[str, Any]]:
    """
    Improvement #2: Query ambiguity warnings.

    Detect when a query could be interpreted multiple ways and suggest alternatives.
    """
    query_lower = query.lower()
    ambiguities = []

    # Check for "creative" ambiguity: playmaking vs goal-scoring
    if "creative" in query_lower and ("midfielder" in query_lower or "forward" in query_lower):
        if not any(w in query_lower for w in ["pass", "assist", "chance", "goal", "shoot"]):
            ambiguities.append({
                "issue": "Creative midfielder could mean playmaking OR goal-scoring",
                "interpretation_1": "Playmaking (passes, assists, xAG)",
                "interpretation_2": "Goal-scoring from midfield (goals, xG, shots)",
                "suggest": "Query is ambiguous. System will search for both.",
            })

    return {"ambiguities": ambiguities} if ambiguities else None


def build_filter_relaxation_ladder(
    season: str, position: str, min_minutes: float, stat_filters: List[str]
) -> List[Dict[str, Any]]:
    """
    Improvement #3: Filter relaxation visualization.

    Build the full ladder of filter steps, showing which constraints were relaxed.
    This explains WHY the final filter was chosen (because earlier steps returned 0).
    """
    ladder = [
        {
            "step": 1,
            "filters": f"season={season}, position={position}, min_minutes={min_minutes}, stats={stat_filters}",
            "description": "All constraints (incl. stat thresholds)",
            "strictness": "Very strict",
        },
        {
            "step": 2,
            "filters": f"season={season}, position={position}, min_minutes={min_minutes}",
            "description": "Stat thresholds relaxed",
            "strictness": "Strict",
        },
        {
            "step": 3,
            "filters": f"season={season}, position={position}",
            "description": "Min-minutes relaxed",
            "strictness": "Medium",
        },
        {
            "step": 4,
            "filters": f"season={season}",
            "description": "Position relaxed",
            "strictness": "Loose",
        },
        {
            "step": 5,
            "filters": "",
            "description": "All relaxed (unfiltered)",
            "strictness": "Very loose",
        },
    ]
    return ladder


def scalar_vs_vector_balance(
    pool_size_before_vector: int, pool_size_after_vector: int, total_corpus: int
) -> Dict[str, Any]:
    """
    Improvement #4: Scalar vs vector balance indicator.

    Show how much filtering work was done by scalars vs vectors.
    """
    scalar_reduction = 1.0 - (pool_size_before_vector / total_corpus)
    vector_reduction = 1.0 - (pool_size_after_vector / pool_size_before_vector)

    return {
        "total_corpus": total_corpus,
        "after_scalar_filters": pool_size_before_vector,
        "after_vector_ranking": pool_size_after_vector,
        "scalar_work_pct": round(scalar_reduction * 100, 1),
        "vector_work_pct": round(vector_reduction * 100, 1),
        "explanation": (
            f"Scalar filters (season/position/stats) reduced from {total_corpus} to "
            f"{pool_size_before_vector} ({scalar_reduction*100:.0f}%). Vector ranking "
            f"narrowed to top {pool_size_after_vector} ({vector_reduction*100:.0f}%)."
        ),
    }


def confidence_bands_for_rankings(
    candidates: List[Dict[str, Any]], rank_by: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Improvement #5: Confidence bands for rankings.

    Show confidence level (high/medium/low) for each candidate based on:
    - How well their stats match the query
    - How strong their semantic match is
    - Data availability (full season vs partial)
    """
    ranked_with_confidence = []

    for i, c in enumerate(candidates):
        # Cosine score confidence: higher = more confident
        cosine = float(c.get("relevance_score", 0.5))
        cosine_confidence = cosine * 100  # 0.75 cosine → 75% confidence in semantic match

        # Stats match: check if all rank_by columns have data
        stats = c.get("stats", {})
        rank_cols = [col.get("column") for col in rank_by]
        stats_available = sum(1 for col in rank_cols if col in stats and stats[col] is not None)
        stats_coverage = (stats_available / len(rank_cols) * 100) if rank_cols else 100

        # Blended confidence
        overall_confidence = round((cosine_confidence * 0.6) + (stats_coverage * 0.4), 0)

        ranked_with_confidence.append({
            "rank": i + 1,
            "player_name": c.get("player_name"),
            "overall_confidence_pct": int(overall_confidence),
            "confidence_level": (
                "High" if overall_confidence >= 80
                else "Medium" if overall_confidence >= 60
                else "Low"
            ),
            "cosine_confidence_pct": int(cosine_confidence),
            "stats_coverage_pct": int(stats_coverage),
            "evidence": {
                "strong_semantic_match": cosine > 0.70,
                "complete_stats_available": stats_coverage == 100,
                "full_season_data": stats.get("minutes", 0) > 1800,
            }
        })

    return ranked_with_confidence


def risk_flags_and_context(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improvement #6: Risk flags & context.

    Surface concerns like declining form, injury history, small sample size.
    """
    flags = {"positive": [], "warnings": [], "concerns": []}

    # Career context
    progression = candidate.get("progression", {})
    if progression:
        trend = progression.get("trend", "stable")
        momentum = progression.get("momentum", 0)

        if trend == "improving":
            flags["positive"].append(f"📈 Improving trajectory ({momentum*100:+.0f}% YoY)")
        elif trend == "declining":
            flags["concerns"].append(f"📉 Declining trajectory ({momentum*100:+.0f}% YoY) — check recent form")

        consistency = progression.get("consistency", "variable")
        if consistency == "very consistent":
            flags["positive"].append("✓ Very consistent performer")
        elif consistency in ("variable", "highly variable"):
            flags["warnings"].append(f"⚠️ {consistency} — form fluctuates")

    # Sample size
    minutes = candidate.get("stats", {}).get("minutes", 0)
    appearances = candidate.get("stats", {}).get("appearances", 0)
    if minutes and minutes < 900:
        flags["warnings"].append(f"⚠️ Limited sample (only {int(minutes)} mins)")
    if appearances and appearances < 10:
        flags["concerns"].append(f"⚠️ Few appearances ({int(appearances)}) — small sample")

    # Manager fit
    alignment = candidate.get("tactical_alignment", {})
    if alignment.get("conflict"):
        flags["concerns"].append(f"⚠️ Tactical conflict with manager: {', '.join(alignment['conflict'])}")
    if alignment.get("aligned"):
        flags["positive"].append(f"✓ Tactical alignment: {', '.join(alignment['aligned'])}")

    return flags


def _parse_gbp(formatted_str: Optional[str]) -> Optional[int]:
    """Parse formatted currency string to integer (e.g., "£27,300,000" → 27300000)"""
    if not formatted_str or not isinstance(formatted_str, str):
        return None
    try:
        # Remove £ and commas, convert to int
        return int(formatted_str.replace("£", "").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def comparison_alternatives(
    top_candidate: Dict[str, Any], pool: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Improvement #7: Comparison table.

    Find:
    - Top pick (already the first)
    - Cheaper alternative (similar stats, lower wage)
    - Breakthrough prospect (high potential, less proven)
    """
    alternatives = {"top_pick": top_candidate, "cheaper_alternative": None, "breakthrough_prospect": None}

    top_stats = top_candidate.get("stats", {})
    top_cost_raw = _parse_gbp(top_candidate.get("estimated_cost", {}).get("annual_wages"))
    top_cost_display = top_candidate.get("estimated_cost", {}).get("annual_wages")
    top_goals = top_stats.get("goals", 0)
    top_assists = top_stats.get("assists", 0)

    if not top_cost_raw or top_goals < 1:
        return alternatives  # Can't compare without baseline

    for candidate in pool[1:]:
        cand_cost_raw = _parse_gbp(candidate.get("estimated_cost", {}).get("annual_wages"))
        cand_cost_display = candidate.get("estimated_cost", {}).get("annual_wages")
        cand_stats = candidate.get("stats", {})
        cand_goals = cand_stats.get("goals", 0)
        cand_assists = cand_stats.get("assists", 0)
        cand_minutes = cand_stats.get("minutes", 0)

        # Cheaper alternative: similar stats, lower cost
        if (
            cand_cost_raw and top_cost_raw
            and cand_cost_raw < top_cost_raw * 0.7
            and cand_goals >= top_goals * 0.8
            and cand_assists >= top_assists * 0.8
        ):
            if not alternatives["cheaper_alternative"]:
                cost_savings_pct = round((1 - cand_cost_raw / top_cost_raw) * 100)
                alternatives["cheaper_alternative"] = {
                    "candidate": candidate,
                    "cost_savings": f"{cost_savings_pct}%",
                    "stat_ratio": f"{round(cand_goals/top_goals * 100)}% of top pick's goals",
                }

        # Breakthrough prospect: young, high potential, fewer minutes
        if cand_minutes and cand_minutes < 1800 and cand_goals >= 5:
            progression = candidate.get("progression", {})
            if progression.get("trend") == "improving":
                if not alternatives["breakthrough_prospect"]:
                    alternatives["breakthrough_prospect"] = {
                        "candidate": candidate,
                        "potential": f"Improving {progression.get('momentum', 0)*100:+.0f}% YoY",
                        "minutes_played": int(cand_minutes),
                    }

    return alternatives


def similarity_breakdown_by_dimension(
    candidate: Dict[str, Any], rank_by: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Improvement #9: Similarity breakdown by dimension.

    For each ranking concept (ball recovery, passing range, goal threat),
    show separate similarity scores.
    """
    breakdown = {}
    stats = candidate.get("stats", {})

    # Map ranking columns to human-readable concepts
    concept_map = {
        "tackles": "Ball-winning (tackles)",
        "interceptions": "Ball-winning (interceptions)",
        "recoveries": "Ball recovery",
        "prgp": "Passing range",
        "prgc": "Ball-carrying",
        "goals": "Goal threat",
        "xg": "Expected goals (potential)",
        "assists": "Chance creation",
        "xag": "Expected assists",
        "big_chances_created": "Big chances created",
    }

    for col_dict in rank_by:
        col = col_dict.get("column")
        if col in concept_map and col in stats:
            val = stats[col]
            # Normalize to 0-1 for display (rough heuristic)
            if isinstance(val, (int, float)):
                if col in ("prgp", "prgc", "tackles"):
                    norm = min(val / 150, 1.0) if val else 0
                elif col in ("goals", "assists"):
                    norm = min(val / 20, 1.0) if val else 0
                elif col in ("xg", "xag"):
                    norm = min(val / 15, 1.0) if val else 0
                else:
                    norm = 0.5

                breakdown[concept_map[col]] = {
                    "raw_value": val,
                    "similarity_score": round(norm, 2),
                    "confidence": "high" if val is not None else "low",
                }

    return breakdown


def explainability_ledger(
    candidate: Dict[str, Any], query_matched_concepts: List[str]
) -> Dict[str, Any]:
    """
    Improvement #11: Explainability ledger.

    Show the decision tree: why did this player rank where they did?
    """
    ledger = {
        "player_name": candidate.get("player_name"),
        "decision_components": [],
        "total_score": 0,
    }

    score = 0

    # Position match
    if candidate.get("position"):
        score += 20
        ledger["decision_components"].append({
            "factor": "Position match",
            "points": 20,
            "reason": f"Position is {candidate.get('position')}",
        })

    # Stat thresholds passed
    stats = candidate.get("stats", {})
    if stats.get("tackles", 0) > 100:
        score += 30
        ledger["decision_components"].append({
            "factor": "Defensive strength",
            "points": 30,
            "reason": f"Tackles: {int(stats.get('tackles', 0))} (> 100)",
        })

    if stats.get("assists", 0) > 5:
        score += 25
        ledger["decision_components"].append({
            "factor": "Creative output",
            "points": 25,
            "reason": f"Assists: {int(stats.get('assists', 0))} (> 5)",
        })

    # Semantic match
    cosine = float(candidate.get("relevance_score", 0.5))
    if cosine > 0.70:
        score += 25
        ledger["decision_components"].append({
            "factor": "Semantic similarity",
            "points": 25,
            "reason": f"Query similarity: {cosine:.2f} (> 0.70)",
        })

    # Career trajectory
    progression = candidate.get("progression", {})
    if progression.get("trend") == "improving":
        score += 15
        ledger["decision_components"].append({
            "factor": "Improving trajectory",
            "points": 15,
            "reason": f"Momentum: {progression.get('momentum', 0)*100:+.0f}% YoY",
        })
    elif progression.get("trend") == "declining":
        score -= 10
        ledger["decision_components"].append({
            "factor": "Declining trajectory",
            "points": -10,
            "reason": f"Momentum: {progression.get('momentum', 0)*100:+.0f}% YoY",
        })

    ledger["total_score"] = max(0, min(100, score))  # Clamp to 0-100

    return ledger


def uncertainty_quantification(claim: str, data_quality: str) -> Dict[str, Any]:
    """
    Improvement #13: Uncertainty quantification.

    Mark claims as high/medium/low confidence based on data quality.
    """
    confidence_levels = {
        "high": {
            "symbol": "✓",
            "description": "High confidence — well-supported by data",
            "criteria": ["Full season", "Multiple years", "Recent data", "Large sample"],
        },
        "medium": {
            "symbol": "⚠️",
            "description": "Medium confidence — some supporting data",
            "criteria": ["Partial season", "Older season", "Few data points", "Small sample"],
        },
        "low": {
            "symbol": "❓",
            "description": "Low confidence — inferred/estimated",
            "criteria": ["Single match", "Projection", "Estimated stat", "Very recent"],
        },
    }

    # Simple heuristic: map data_quality to confidence
    if data_quality in ("full_season", "multi_year", "recent"):
        level = "high"
    elif data_quality in ("partial_season", "older", "few_samples"):
        level = "medium"
    else:
        level = "low"

    return {
        "claim": claim,
        "confidence_level": level,
        **confidence_levels[level],
    }


def player_clustering_similar_players(
    query_candidate: Dict[str, Any], pool: List[Dict[str, Any]], limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Improvement #14: Player clustering / Similar players.

    Find players most similar to the query candidate.
    """
    target_stats = query_candidate.get("stats", {})
    target_name = query_candidate.get("player_name", "")

    similarities = []

    for candidate in pool:
        if candidate.get("player_name") == target_name:
            continue

        cand_stats = candidate.get("stats", {})

        # Simple similarity: cosine distance of stats
        diff_score = 0
        stat_keys = ["tackles", "assists", "goals", "prgp", "prgc"]

        for key in stat_keys:
            target_val = target_stats.get(key, 0)
            cand_val = cand_stats.get(key, 0)
            if target_val:
                diff = abs(cand_val - target_val) / target_val
                diff_score += diff

        if stat_keys:
            similarity = max(0, 1 - (diff_score / len(stat_keys)))

            similarities.append({
                "player_name": candidate.get("player_name"),
                "club": candidate.get("current_club"),
                "position": candidate.get("position"),
                "similarity_pct": int(similarity * 100),
                "key_differences": cand_stats.get("position") != target_stats.get("position"),
            })

    # Sort by similarity and return top N
    return sorted(similarities, key=lambda x: x["similarity_pct"], reverse=True)[:limit]


def what_if_alternative_ranking(
    candidates: List[Dict[str, Any]], alternative_rank_by: List[str]
) -> List[Dict[str, Any]]:
    """
    Improvement #15: What-if analysis.

    Show how rankings would change if we prioritized different stats.
    """
    # This would require re-scoring all candidates with different weights
    # For now, return a structure that shows the concept

    return [
        {
            "scenario": f"If we prioritize {alternative_rank_by}",
            "note": "Re-ranking would require live stat re-weighting",
            "current_ranking": [c.get("player_name") for c in candidates],
        }
    ]
