import os
import re
import time
from typing import Any, Dict, List, Optional

import dspy
import numpy as np

from backend.app.database import (
    BLENDED_COLLECTION,
    CAREER_COLLECTION,
    EMBEDDING_MODEL_NAME,
    STATS_COLLECTION,
    VECTOR_DIM,
    fetch_player_history,
    fetch_player_vectors,
    get_embedding,
    get_embeddings,
    query_blended,
    query_career,
    query_stats,
    query_theory,
)
from backend.app.enrichment.progression import calculate_progression, format_progression
from backend.app.query_understanding import (
    detect_query_concept,
    is_career_query,
    mentions_cost,
    parse_query_constraints,
)
from backend.app.query_mapper import (
    map_query_to_filters,
    analyze_tactical_suitability,
    extract_manager_tactics,
    compare_alignment,
)
from backend.app.tactical_scorer import build_metrics_checklist
from backend.app.enrichment.impact_analysis import (
    build_ui_response_envelope,
    extract_impact_metrics,
)
from backend.app.enrichment.visualization import build_similarity_plot
from backend.app.reference import (
    detect_club,
    get_estimated_cost,
    get_manager_profile,
    normalize_club,
)
from backend.app.tactical_reference import search_tactical_systems
from backend.app.explainability import (
    confidence_score_intent,
    rank_alternatives,
    build_filter_relaxation_ladder,
    scalar_vs_vector_balance,
    confidence_bands_for_rankings,
    risk_flags_and_context,
    comparison_alternatives,
    similarity_breakdown_by_dimension,
    explainability_ledger,
    uncertainty_quantification,
    player_clustering_similar_players,
    what_if_alternative_ranking,
)


def _lm_name() -> str:
    """Human-readable name of the currently configured DSPy language model."""
    lm = getattr(dspy.settings, "lm", None)
    return getattr(lm, "model", "no LM configured") if lm else "no LM configured"


_STAT_LABELS = {
    "goals": "goals",
    "assists": "assists",
    "xg": "xG",
    "xag": "xAG",
    "prgc": "progressive carries",
    "prgp": "progressive passes",
    "tackles": "tackles",
    "tackles_won": "tackles won",
    "interceptions": "interceptions",
    "recoveries": "recoveries",
    "big_chances_created": "big chances created",
    "saves": "saves",
    "clean_sheets": "clean sheets",
    "improvement_score": "improvement",
    "stability_score": "stability",
    "consistency_pct": "consistency %",
}


def _focused_profile(candidate, columns):
    """Aspect-scoped player context: identity + ONLY the stat facets the query
    asked about (the rank_by columns). Removes off-topic stats so the context
    fits the query. Falls back to the full prose when there are no concept cues.
    """
    if not columns:
        return candidate.get("stats_summary", "")
    stats = candidate.get("stats") or {}
    base = f"{candidate.get('player_name', 'Unknown')} ({candidate.get('position', '?')}, {candidate.get('current_club', 'Unknown')}, {candidate.get('season', 'N/A')})"
    mins, apps = stats.get("minutes"), stats.get("appearances")
    if mins:
        base += f", {int(mins)} mins"
    elif apps:
        base += f", {int(apps)} apps"

    facets = []
    for col in columns:
        v = None
        # Check stats dictionary first
        if col in stats and stats[col] is not None:
            v = stats[col]
        # For progression metrics, also check candidate-level data (for backward compat)
        elif col in ("improvement_score", "stability_score", "consistency_pct") and col in candidate and candidate[col] is not None:
            v = candidate.get(col)

        if v is not None:
            # Format based on metric type
            if col == "consistency_pct":
                facets.append(f"{int(v)}% {_STAT_LABELS.get(col, col)}")
            elif col in ("improvement_score", "stability_score"):
                # These are 0-1 floats, format as percentage
                facets.append(f"{v:.0%} {_STAT_LABELS.get(col, col)}")
            else:
                # Other stats are integers or large decimals
                facets.append(f"{int(v)} {_STAT_LABELS.get(col, col)}")

    return base + (" — " + ", ".join(facets) + "." if facets else ".")


def _compress_theory(query_vector, snippets, keep_fraction=0.6):
    """Sentence-level contextual compression: within each theory chunk, keep only
    the sentences most relevant to the query (by cosine), in original order. This
    strips off-topic prose so the passed context fits the query (raising
    contextual relevancy and reducing noise to the LLM).
    """
    qv = np.asarray(query_vector, dtype=float)
    qn = np.linalg.norm(qv) + 1e-9
    out = []
    for snip in snippets:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snip.strip()) if s.strip()]
        if len(sents) <= 2:
            out.append(snip)
            continue
        embs = np.asarray(get_embeddings(sents), dtype=float)
        sims = (embs @ qv) / ((np.linalg.norm(embs, axis=1) + 1e-9) * qn)
        k = max(1, int(round(len(sents) * keep_fraction)))
        keep_idx = sorted(np.argsort(sims)[-k:].tolist())
        out.append(" ".join(sents[i] for i in keep_idx))
    return out


class ScoutingReportSignature(dspy.Signature):
    """
    You are an expert Sporting Director producing a recruitment recommendation.

    Your recommendation MUST be grounded in available data. When tactical metrics are listed in
    metrics_checklist, cite ONLY those. When the checklist indicates metrics aren't available,
    use the player's actual stats (from player_stats) to justify the recommendation.

    RULES (Grounding-First):
      1. Recommend the single best-fit player (name them) and, briefly, a runner-up.

      2. Prioritize metrics from metrics_checklist (use exact values with ✓/✗).
         If the checklist lists metrics → cite those with thresholds.
         If the checklist is empty/missing tactics → fall back to player_stats for justification.

      3. For each claim, show explicit verification:
         CORRECT (with checklist): "Tackles 160 (target > 150) ✓"
         CORRECT (without checklist): "João has 156 tackles, 45 interceptions → strong ball-winner"
         WRONG:   "Excellent defensive positioning" (too generic, unverifiable)

      4. Build the brief step-by-step:
         Step 1: Check metrics_checklist. Are metrics listed? (Yes/No)
         Step 2a (Yes): Cite metrics with ✓. Only include passing grades.
         Step 2b (No): Use player_stats to show why player fits the query.
         Step 3: Connect to query requirements clearly.
         Step 4: Write brief grounded in data (either checklist or raw stats).

      5. When career data is provided in metrics_checklist:
         - For "improving": cite improvement_score ✓ and upward trend explicitly
         - For "consistent": cite stability_score ✓ and consistency_pct to show reliability
         - For "declining": cite negative momentum with warning
         - Always blend current season form WITH career baseline when both are available

      6. CRITICAL RULES (Prevent Hallucination):
         - DO NOT invent metrics or thresholds not in checklist or player stats
         - DO NOT add generic observations like "strong positioning sense"
         - ALWAYS cite numbers: "156 tackles" not "exceptional defensively"
         - When using player_stats, reference specific numbers available
         - Admissible: metrics in checklist OR actual stats in player_stats

      7. Discuss wages ONLY if explicitly in player profile.

      8. No generic football platitudes. Every claim must cite a number.

    FORMAT YOUR REASONING as 4-7 bullets showing verification:
      - Query asks for X. Available metrics show Y.
      - Player achieves W. Why is this fit?

    FORMAT YOUR BRIEF as:
      [Player Name] is the best fit.
      - [Query concept]: [Metric/Stat] [Value] [Verification]
      - (Repeat for each concept)
    """

    tactical_context = dspy.InputField(
        desc="Modern tactical systems with required metrics. "
             "Use for background understanding only; actual thresholds come from metrics_checklist."
    )

    player_stats = dspy.InputField(
        desc="Candidate player profiles with all available statistics. "
             "Use these both to verify metrics_checklist claims AND to justify recommendations when "
             "specific tactical metrics aren't available in the checklist (fallback grounding)."
    )

    tactical_query = dspy.InputField(
        desc="Specific tactical question. "
             "Determines which metrics in the checklist are relevant, or which player_stats to emphasize."
    )

    metrics_checklist = dspy.InputField(
        desc="Available metrics with thresholds (if any). "
             "If populated: cite ONLY these with exact values and ✓/✗. "
             "If empty or missing tactics: fall back to player_stats for grounding. "
             "Format when available: 'CONCEPT: [metric] (target > [threshold]) → [value] ✓/✗'. "
             "Example filled: 'Tackles 160 (target > 150) ✓'. "
             "Example empty: '[No matching metrics for this tactic—use player_stats instead]'."
    )

    reasoning = dspy.OutputField(
        desc="4-7 bullets showing verification for each claim. "
             "For each: cite the metric/stat and its value, then explain fit. "
             "Source from checklist if available; otherwise from player_stats."
    )

    scouting_brief = dspy.OutputField(
        desc="A recommendation grounded in available data (checklist metrics OR raw player_stats). "
             "When checklist has metrics with ✓: cite those. "
             "When checklist is empty: cite actual stats from player_stats. "
             "Example with checklist: 'Tackles 160 (target > 150) ✓ + Interceptions 45 ✓'. "
             "Example without checklist: 'João: 156 tackles, 45 interceptions, 12 key passes—strong creator and defender'. "
             "CRITICAL: Every number must come from checklist or player_stats. No invented claims."
    )


def _hit_field(hit: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Milvus search hit, tolerating flat or nested layouts."""
    try:
        if name in hit:
            return hit[name]
    except TypeError:
        pass
    entity = hit.get("entity") if hasattr(hit, "get") else None
    if isinstance(entity, dict) and name in entity:
        return entity[name]
    return default


def _build_candidate(hit: Any) -> Dict[str, Any]:
    """Turn a player-stats hit into an enriched, human-readable candidate with progression."""
    player_name = _hit_field(hit, "player_name", "Unknown")
    squad = _hit_field(hit, "squad", "Unknown")
    season = _hit_field(hit, "season", "")
    manager_profile = get_manager_profile(squad, season)

    stats_dict = {
        k: _hit_field(hit, k)
        for k in (
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
            "improvement_score",
            "stability_score",
            "consistency_pct",
        )
        if _hit_field(hit, k) is not None
    }

    # Fetch full history and calculate progression
    history = fetch_player_history(player_name)
    progression = calculate_progression(history)
    progression_summary = format_progression(progression)

    # Analyze which tactical systems suit this player
    tactical_fit = analyze_tactical_suitability(stats_dict)

    # Compare with manager's system
    manager_style = manager_profile.get("style")
    manager_tactics = extract_manager_tactics(manager_style) if manager_style else []
    tactical_alignment = compare_alignment(manager_tactics, tactical_fit)

    return {
        "player_name": player_name,
        "position": _hit_field(hit, "position", "Unknown"),
        "current_club": squad,
        "current_manager": manager_profile["manager"],
        "manager_playing_style": manager_profile["style"],
        "manager_season": manager_profile["season"],
        "season": season,
        "stats_summary": _hit_field(hit, "text", ""),
        "stats": stats_dict,
        "progression": progression,
        "progression_summary": progression_summary,
        # Explicit scores for LLM to use when differentiating candidates
        "stability_score": progression.get("stability_score", 0),  # 0-1: consistency across seasons
        "improvement_score": progression.get("improvement_score", 0),  # 0-1: upward trajectory strength
        "consistency_pct": progression.get("consistency_pct", 0),  # 0-100: human-readable consistency
        "tactical_suitability": tactical_fit,
        "manager_tactics": manager_tactics,
        "tactical_alignment": tactical_alignment,
        "estimated_cost": get_estimated_cost(player_name),
        "relevance_score": round(float(_hit_field(hit, "distance", 0.0) or 0.0), 4),
    }


def _enrich_candidate_defaults(c: Dict[str, Any], season: Optional[str] = None) -> Dict[str, Any]:
    """Ensure a candidate has all required enriched fields, filling in defaults for career results."""

    # Ensure player_name and position are always present (should be from source data)
    if "player_name" not in c:
        c["player_name"] = (c.get("name") or "Unknown").strip()
    else:
        # Clean up any whitespace in player_name
        c["player_name"] = (c["player_name"] or "Unknown").strip()

    # Ensure position is set (may come from database or be empty)
    position = (c.get("position") or "").strip()
    if not position or position == "Unknown":
        # Try to fetch from player history if not available
        player_name = c.get("player_name")
        if player_name and player_name != "Unknown":
            try:
                history = fetch_player_history(player_name)
                if history and isinstance(history, list) and len(history) > 0:
                    # Get position from the most recent season
                    for row in reversed(history):
                        if row.get("position") and row.get("position") != "Unknown":
                            position = row.get("position")
                            break
            except Exception:
                pass

    c["position"] = position or "Unknown"

    # Determine the club (season-specific results use "squad", career results use "best_squad")
    # Try season-specific squad first, then fallback to best_squad or current_club
    squad = (c.get("squad") or c.get("best_squad") or c.get("current_club") or "").strip()

    # If no squad found, try to fetch from player history
    if not squad or squad == "Unknown":
        player_name = c.get("player_name")
        if player_name and player_name != "Unknown":
            try:
                history = fetch_player_history(player_name)
                if history and isinstance(history, list) and len(history) > 0:
                    # history is sorted ascending, so last item is most recent
                    latest_row = history[-1]
                    squad = (latest_row.get("squad") or "").strip()
            except Exception:
                pass  # If history fetch fails, continue with None

    # Final fallback to best_squad if we still don't have a squad
    if not squad or squad == "Unknown":
        squad = (c.get("best_squad") or "").strip()

    squad = squad or "Unknown"

    # Set current_club if not already set
    if "current_club" not in c:
        c["current_club"] = squad

    # Get manager profile (try multiple clubs if the primary one fails)
    if "current_manager" not in c:
        # Strip and normalize the squad name before lookup
        normalized_squad = normalize_club(squad) if squad and squad != "Unknown" else squad
        manager_profile = get_manager_profile(squad, season)

        # If manager lookup failed, try alternative squad sources
        if not manager_profile.get("manager"):
            for alt_squad in [c.get("best_squad"), c.get("squad"), c.get("current_club")]:
                if alt_squad and alt_squad != squad:
                    alt_manager = get_manager_profile(alt_squad, season)
                    if alt_manager.get("manager"):
                        manager_profile = alt_manager
                        break

        c["current_manager"] = manager_profile.get("manager") or "manager unknown"
        c["manager_playing_style"] = manager_profile.get("style") or ""
        c["manager_season"] = manager_profile.get("season")

    # Get estimated cost
    if "estimated_cost" not in c:
        player_name = (c.get("player_name") or "").strip()
        if player_name and player_name != "Unknown":
            c["estimated_cost"] = get_estimated_cost(player_name)
        else:
            # Fallback if no player name
            c["estimated_cost"] = {
                "weekly_wages": None,
                "annual_wages": None,
                "basis": "No wage data available for this player.",
            }

    # Ensure progression is populated (for blended records, include all career metrics)
    if "progression" not in c:
        c["progression"] = {
            "momentum": c.get("momentum", 0),
            "trend": c.get("trend", "stable"),
            "improvement_score": c.get("improvement_score", 0),
            "stability_score": c.get("stability_score", 0),
            "consistency_pct": c.get("consistency_pct", 0),
        }

    # Set progression_summary for career results
    if "progression_summary" not in c and c.get("momentum") is not None:
        momentum = c.get("momentum", 0)
        trend_desc = ""
        if momentum > 0.15:
            trend_desc = f"Improving ({momentum:+.0%})"
        elif momentum < -0.15:
            trend_desc = f"Declining ({momentum:+.0%})"
        else:
            trend_desc = "Stable"
        c["progression_summary"] = trend_desc

    # Use career prose as stats_summary if not already set
    if "stats_summary" not in c and c.get("text"):
        c["stats_summary"] = c["text"]

    # Compute tactical suitability for career results if not already set
    if "tactical_suitability" not in c:
        # For career results, compute from available stats
        stats_dict = {
            k: c.get(f"avg_{k}" if k in ("goals", "assists", "prgp", "tackles") else k)
            for k in ("goals", "assists", "xg", "xag", "prgc", "prgp", "tackles", "tackles_won")
            if c.get(f"avg_{k}" if k in ("goals", "assists", "prgp", "tackles") else k) is not None
        }
        c["tactical_suitability"] = analyze_tactical_suitability(stats_dict) if stats_dict else []

    # Extract manager tactics for the current club if not already set
    if "manager_tactics" not in c:
        manager_style = c.get("manager_playing_style")
        c["manager_tactics"] = extract_manager_tactics(manager_style) if manager_style else []

    # Compute tactical alignment if not already set
    if "tactical_alignment" not in c:
        c["tactical_alignment"] = compare_alignment(c.get("manager_tactics", []), c.get("tactical_suitability", []))

    # For career aggregates, populate stats dict from avg_* fields so _focused_profile can find them
    if "stats" not in c or not c.get("stats"):
        c["stats"] = {}
    if not c["stats"]:
        # Map avg_* fields from career aggregates to stats dict
        avg_fields = ["avg_goals", "avg_assists", "avg_prgp", "avg_tackles", "avg_tackles_won", "avg_interceptions"]
        for field in avg_fields:
            if field in c:
                # Map avg_tackles → tackles, avg_tackles_won → tackles_won, etc.
                stat_name = field.replace("avg_", "")
                c["stats"][stat_name] = c[field]

    # Ensure progression metrics from top-level fields are in stats dict (for blended records)
    # This is needed so explainability_ledger, _focused_profile, and ranking functions can access them
    progression_fields = ["improvement_score", "stability_score", "consistency_pct"]
    for field in progression_fields:
        if field in c and c[field] is not None and field not in c["stats"]:
            c["stats"][field] = c[field]

    return c


def _format_candidate_for_prompt(c: Dict[str, Any], include_cost: bool = False) -> str:
    """Render a candidate as a compact, grounded block for the LLM context."""
    # Wage/cost is included ONLY when the query asks about it (include_cost) — for
    # purely tactical queries it's off-topic and dilutes the brief.
    manager = c.get("current_manager") or "manager unknown"
    style = c.get("manager_playing_style") or ""
    manager_line = f"manager: {manager}" + (f", style: {style}" if style else "")
    cost_line = ""
    if include_cost:
        cost = c.get("estimated_cost") or {}
        wage = cost.get("annual_wages") or "wage unknown"
        weekly = cost.get("weekly_wages")
        cost_line = f" | wage cost: {wage}/yr" + (f" ({weekly}/wk)" if weekly else "")
    # Use the aspect-scoped profile when available (query had concept cues).
    profile = c.get("context_profile") or c.get("stats_summary", "No stats available")
    return (
        f"- {c.get('player_name', 'Unknown')} | {c.get('position', '?')} | {c.get('current_club', 'Unknown')} "
        f"({manager_line}) | season {c.get('season', 'N/A')}{cost_line}\n"
        f"  Stats: {profile}"
    )


# Rate stats are compared PER 90 MINUTES (so a high-impact rotation player isn't
# beaten purely by someone who played more). Volume stats (minutes, appearances,
# clean sheets) are ranked on their raw totals — there, volume IS the signal.
_RATE_COLUMNS = {
    "goals",
    "assists",
    "xg",
    "xag",
    "prgc",
    "prgp",
    "tackles",
    "tackles_won",
    "interceptions",
    "recoveries",
    "big_chances_created",
    "saves",
}
_MIN_NINETIES = 3.0  # ~270 min: shrinkage floor so tiny samples can't inflate per-90


def _ranking_value(stats: Dict[str, Any], col: str) -> float:
    """Per-90 rate for rate stats; raw total for volume stats.

    Per-90 uses real minutes when available, else falls back to appearances as a
    '90s' proxy (2023/24 has appearances but no minutes). A shrinkage floor damps
    inflation from very small samples (e.g. 1 tackle in 20 minutes).
    """
    raw = float(stats.get(col) or 0.0)
    if col not in _RATE_COLUMNS:
        return raw
    minutes = float(stats.get("minutes") or 0.0)
    apps = float(stats.get("appearances") or 0.0)
    nineties = (minutes / 90.0) if minutes > 0 else apps
    return raw / max(nineties, _MIN_NINETIES)


def _filter_checklist_by_concept(checklist_text: str, concepts: list) -> str:
    """Filter metrics checklist to only show query-relevant metrics.

    Reduces noise and improves answer relevancy by removing metrics
    that aren't related to the query's tactical concepts. However, always
    keep career/progression metrics (improvement, stability, consistency)
    as they're crucial for many queries.
    """
    if not concepts or not checklist_text:
        return checklist_text

    # Map concepts to checklist section keywords
    concept_keywords = {
        "high_press": ["PRESSING", "HIGH PRESS", "PRESS", "WORK RATE"],
        "ball_progression": ["PROGRESSION", "BALL PROGRESSION", "PASSING", "CARRY"],
        "creative": ["CREATIVE", "ASSISTS", "PLAYMAKING", "KEY PASS", "CHANCE"],
        "defensive_transition": ["DEFENSIVE", "DEFENSIVE TRANSITION", "TACKLES", "INTERCEPTIONS", "RECOVERY", "1V1"],
        "goal_threat": ["GOAL", "GOALS", "THREAT", "FINISHING", "CLINICAL"],
        "momentum": ["MOMENTUM", "TRAJECTORY", "TREND", "IMPROVEMENT", "STABILITY", "CONSISTENCY"],
        "player_development": ["IMPROVEMENT", "STABILITY", "CONSISTENCY", "TRAJECTORY", "YOUNG", "EMERGING"],
        "defending": ["TACKLES", "INTERCEPTIONS", "CLEARANCE", "1V1", "DEFENDING"],
        "attacking_fullback": ["ASSISTS", "PROGRESSIVE", "FULLBACK", "CARRIES", "ATTACKING"],
    }

    # Collect relevant keywords from query concepts
    relevant_keywords = set()
    for concept in concepts:
        keywords = concept_keywords.get(concept.lower(), [])
        relevant_keywords.update(kw.upper() for kw in keywords)

    # Always include progression/stability metrics (important for all queries)
    relevant_keywords.update(["IMPROVEMENT", "STABILITY", "CONSISTENCY", "TRAJECTORY", "MOMENTUM"])

    if not relevant_keywords:
        return checklist_text  # No filtering if no concepts matched

    # Filter checklist lines to only relevant sections + their metrics
    lines = checklist_text.split("\n")
    filtered = []
    in_relevant_section = False

    for line in lines:
        upper_line = line.upper()

        # Check if this is a section header matching our concepts
        if any(kw in upper_line for kw in relevant_keywords):
            filtered.append(line)
            in_relevant_section = True
        # Keep metric lines under relevant sections
        elif in_relevant_section and (line.strip().startswith("-") or line.strip().startswith("→")):
            filtered.append(line)
        # Reset if we hit a new section that's not relevant
        elif line.strip() and not line.startswith(" ") and line.strip().startswith(
            ("CAREER", "TACTICAL", "GOAL", "DEFENSIVE", "CREATIVE", "BALL", "MOMENTUM")
        ):
            in_relevant_section = any(kw in upper_line for kw in relevant_keywords)
            if in_relevant_section:
                filtered.append(line)

    return "\n".join(filtered) if filtered else checklist_text


def _rerank_by_stats(candidates, rank_by):
    """Blend semantic similarity with the (per-90) stat columns the query asked for.

    For each rank_by column we compute a per-90 rate (or raw total for volume
    stats), min-max normalize across the pool to [0,1], average into a stat score,
    then combine 50/50 with the normalized cosine score. Columns with no signal
    this season are skipped, so ranking degrades gracefully to pure cosine.
    Returns (reordered_candidates, applied_columns).
    """
    if not rank_by or len(candidates) < 2:
        return candidates, []

    columns = [c["column"] for c in rank_by]
    # Per-90 (or raw) ranking value per candidate, per column.
    applied = []
    col_vals = {}
    for col in columns:
        vals = [_ranking_value(c.get("stats") or {}, col) for c in candidates]
        if max(vals) > min(vals):  # column has signal
            col_vals[col] = vals
            applied.append(col)
    if not applied:
        return candidates, []

    cos = [float(c.get("relevance_score") or 0.0) for c in candidates]
    clo, chi = min(cos), max(cos)

    def _norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    for idx, c in enumerate(candidates):
        stat_scores = [_norm(col_vals[col][idx], min(col_vals[col]), max(col_vals[col])) for col in applied]
        stat_score = sum(stat_scores) / len(stat_scores)
        cos_score = _norm(float(c.get("relevance_score") or 0.0), clo, chi)
        c["blended_score"] = round(0.5 * cos_score + 0.5 * stat_score, 4)

    reordered = sorted(candidates, key=lambda c: c["blended_score"], reverse=True)
    return reordered, applied


def _get_fallback_positions(position):
    """Get fallback positions for a given position (shared logic).

    Used by both season-specific and blended retrieval.
    """
    if position == "LB":
        return ["RB", "LWB", "RWB"]
    elif position == "RB":
        return ["LB", "LWB", "RWB"]
    elif position in ("CM", "DM", "AM"):
        return ["CM", "DM", "AM"]
    elif position in ("CB", "LWB", "RWB"):
        return ["CB", "LB", "RB"]
    elif position in ("ST", "IF", "W"):
        return ["ST", "IF", "W"]
    return None


class ScoutIntelRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_report = dspy.ChainOfThought(ScoutingReportSignature)

    def _retrieve_candidates_blended(
        self,
        query_vector,
        position,
        stat_filters,
        limit=25,
        season=None,
        exclude_club=None,
    ):
        """Retrieve candidates from blended collection.

        One function for both season-specific AND career queries.
        Season parameter is optional:
        - If season specified: returns that season (current form)
        - If season None: returns all seasons (for career analysis)
        """
        pool, filter_note = self._retrieve_with_fallback_blended(
            query_vector,
            position=position,
            stat_filters=stat_filters,
            season=season,
            exclude_club=exclude_club,
            limit=limit,
        )

        return pool, filter_note

    def _retrieve_with_fallback_blended(
        self,
        query_vector,
        position,
        stat_filters,
        season,
        exclude_club,
        limit,
    ):
        """Fallback ladder for blended collection (unified approach)."""
        sf = list(stat_filters or [])

        # Fallback ladder (simpler - just position fallback)
        ladder = [
            (position, sf, season, "all constraints (incl. stat thresholds)"),
            (position, [], season, "stat thresholds relaxed"),
        ]

        # Add granular position fallback if needed
        fallback_positions = _get_fallback_positions(position)
        if fallback_positions:
            for fallback_pos in fallback_positions:
                if fallback_pos != position:
                    ladder.append(
                        (fallback_pos, [], season, f"position relaxed ({position} → {fallback_pos})")
                    )

        # Only unfiltered if NO position was specified
        if not position:
            ladder.append((None, [], season, "position relaxed (no matches)"))

        seen = set()
        for p, f, s, note in ladder:
            key = (p, tuple(f), s)
            if key in seen:
                continue
            seen.add(key)

            hits = query_blended(
                query_vector,
                position=p,
                season=s,
                stat_filters=f,
                exclude_club=exclude_club,
                limit=limit,
            )

            if hits:
                return hits, note

        return [], "no players matched"

    def _retrieve_candidates_career(
        self,
        query_vector,
        position,
        stat_filters,
        limit,
        exclude_club=None,
    ):
        """Retrieve players from career aggregates (multi-season profiles).

        No season filtering — career aggregates span all years. Simpler relaxation
        ladder since we're dealing with consolidated data.

        Player names are NOT explicitly filtered; semantic search matches naturally
        if a specific player is named in the query.
        """
        sf = list(stat_filters or [])
        # Career queries: keep position constraint until last resort
        # Relax stat filters first, then position
        ladder = [
            (position, sf, "all constraints (incl. stat thresholds)"),
            (position, [], "stat thresholds relaxed"),
        ]

        # Add granular position fallback for fullbacks/midfielders
        fallback_positions = None
        if position == "LB":
            fallback_positions = ["RB", "LWB", "RWB"]  # Other fullbacks
        elif position == "RB":
            fallback_positions = ["LB", "LWB", "RWB"]  # Other fullbacks
        elif position in ("CM", "DM", "AM"):
            fallback_positions = ["CM", "DM", "AM"]  # Other midfielders
        elif position in ("CB", "LWB", "RWB"):
            fallback_positions = ["CB", "LB", "RB"]  # Other defenders

        # Add fallback positions if we have them
        if fallback_positions:
            for fallback_pos in fallback_positions:
                if fallback_pos != position:
                    ladder.append((fallback_pos, [], f"position relaxed ({position} → {fallback_pos})"))

        # Only add fully unfiltered fallback if NO position was specified
        if not position:
            ladder.append((None, [], "position relaxed (no matches found with specified position)"))
        seen = set()
        for p, f, note in ladder:
            key = (p, tuple(f))
            if key in seen:
                continue
            seen.add(key)
            hits = query_career(query_vector, position=p, stat_filters=f, exclude_club=exclude_club, limit=limit)
            if hits:
                return hits, note
        return [], "no career profiles matched"

    def _retrieve_candidates(
        self,
        query_vector,
        season,
        position,
        min_minutes,
        stat_filters,
        limit,
        exclude_club=None,
    ):
        """Retrieve players, relaxing filters step-by-step until something matches.

        Hybrid: Milvus applies the scalar filters (season / position /
        min_minutes / stat thresholds) FIRST and ranks only the rows that qualify
        by cosine similarity. We start fully constrained and relax along a ladder —
        dropping the most aggressive constraint first — so the English-derived stat
        requirements are honoured when possible but never cause a silent empty set.

        Player names are NOT explicitly filtered; semantic search matches naturally
        if a specific player is named in the query.
        """
        sf = list(stat_filters or [])
        # Build relaxation ladder with intelligent position relaxation
        # Uses granular positions (CB, LB, RB, CM, DM, AM, ST, IF, W, GK) only
        fallback_positions = None
        if position == "CB":
            fallback_positions = ["LB", "RB"]  # Centre-backs fall back to other defenders
        elif position in ("LB", "RB", "LWB", "RWB"):
            fallback_positions = ["CB", "LB", "RB"]  # Fullbacks fall back to other defenders
        elif position in ("CM", "DM", "AM"):
            fallback_positions = ["CM", "DM", "AM"]  # Midfielders fall back to other mid positions
        elif position in ("ST", "IF", "W"):
            fallback_positions = ["ST", "IF", "W"]  # Forwards fall back to other forward positions
        elif position == "GK":
            fallback_positions = ["GK"]  # Goalkeepers stay GK (no fallback)

        ladder = [
            (
                season,
                position,
                min_minutes,
                sf,
                "all constraints (incl. stat thresholds)",
            ),
            (season, position, min_minutes, [], "stat thresholds relaxed"),
            (season, position, None, [], "min-minutes relaxed"),
            # Before broader position relaxation, try relaxing season
            (None, position, None, [], "season relaxed (multi-season context)"),
        ]

        # Add intelligent position relaxation with granular fallbacks
        if fallback_positions:
            for fallback_pos in fallback_positions:
                if fallback_pos != position:  # Skip if same as original
                    ladder.append(
                        (season, fallback_pos, None, [], f"position relaxed ({position} → {fallback_pos})")
                    )
            for fallback_pos in fallback_positions:
                if fallback_pos != position:
                    ladder.append(
                        (None, fallback_pos, None, [], f"season + position relaxed ({position} → {fallback_pos})")
                    )

        # Final fallback: fully unfiltered ONLY if no position was explicitly requested
        # If user asked for RB/CB/DF etc., don't fall back to all positions (which would include FW)
        if not position:
            ladder.append((None, None, None, [], "unfiltered (all relaxed)"))
        seen = set()
        for s, p, m, f, note in ladder:
            key = (s, p, m, tuple(f))
            if key in seen:
                continue
            seen.add(key)
            hits = query_stats(query_vector, s, p, min_minutes=m, stat_filters=f, exclude_club=exclude_club, limit=limit)
            if hits:
                return [_build_candidate(h) for h in hits], note
        return [], "no matches"

    def forward(
        self,
        query_str: str,
        season: str = None,
    ) -> Dict[str, Any]:
        """
        Retrieves tactical theory + candidate players, enriches each candidate
        with current manager and wage cost, and generates a grounded brief.

        All query intent (position, player name, club to scout for) is inferred from
        the query text during Phase 2 (Query Understanding). This ensures the system
        works end-to-end with natural language only.

        If no players match, it does NOT ask the LLM to write a brief (which would
        invent ungrounded content) — it returns an explicit "no data" result.

        The result includes a ``phases`` trace where every step records its INPUT,
        OUTPUT, and WHY (purpose + mechanism) so the UI can teach how the whole
        system works, end to end.
        """
        phases: List[Dict[str, Any]] = []

        def _phase(name, started, *, input, output, why, **extra):
            phases.append(
                {
                    "name": name,
                    "input": input,
                    "output": output,
                    "why": why,
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    **extra,
                }
            )

        # Phase 1 — embed the query into the shared vector space
        t = time.perf_counter()
        query_vector = get_embedding(query_str)
        _phase(
            "1. Embed query",
            t,
            input=f'Raw text: "{query_str}"',
            output=f"{VECTOR_DIM}-dim float vector",
            why=(
                f"The query is encoded with {EMBEDDING_MODEL_NAME} into the SAME "
                f"{VECTOR_DIM}-dim space as every indexed document. Only vectors in "
                "one space can be compared, so this is what makes semantic (meaning-"
                "based, not keyword) search possible."
            ),
            vector_dim=VECTOR_DIM,
            model=EMBEDDING_MODEL_NAME,
        )

        # Phase 2 — understand the query: extract position + stat intent
        t = time.perf_counter()
        constraints = parse_query_constraints(query_str)
        eff_position = constraints["position"]
        rank_by = constraints["rank_by"]
        stat_filters = list(constraints["stat_filters"])

        # Extract tactical concepts and map to FBref filters
        tactical_concepts, tactical_filters = map_query_to_filters(query_str)
        stat_filters.extend(
            [f"{k}>={v}" for k, v in tactical_filters.items()]
        )
        # min_minutes: parsed from query OR season default
        if constraints["min_minutes"] is not None:
            eff_min_minutes = float(constraints["min_minutes"])
        elif season in ("2024/2025", "2025/2026"):
            eff_min_minutes = 270.0  # ~3 full matches; cuts tiny-sample noise
        else:
            eff_min_minutes = None
        # Club you're scouting FOR — its own players are excluded from results.
        # Inferred from query text via detect_club(). No explicit parameter.
        exclude_key = detect_club(query_str)
        exclude_display = exclude_key.title() if exclude_key else None
        matched = list(constraints["matched"])
        if exclude_key:
            matched.append(f"excluding own club: {exclude_display}")
        _phase(
            "2. Understand query",
            t,
            input=f'"{query_str}"',
            output=(", ".join(matched) or "no explicit stat cues — pure semantic"),
            why=(
                "A lightweight parser maps English to your dataset's columns: explicit "
                "numbers become HARD filters (applied before the search), while "
                "qualitative concepts ('high-volume passing', 'wins the ball back') "
                "become RANKING signals on the matching stat columns. If you name the "
                "club you're scouting for, ITS players are excluded — you want fits "
                "from OTHER teams. This is how a plain-English query 'considers the stats'."
            ),
            constraints={**constraints, "exclude_club": exclude_display},
        )

        # Phase 3 — Index A: concept-targeted theory retrieval, then compression
        t = time.perf_counter()
        # Lookup structured tactical reference (not book indexing)
        tactical_ref_hits = search_tactical_systems(query_str)

        # Also query database for Football Hackers context pack chunks
        football_hackers_hits = query_theory(query_str, limit=3)

        theory_context = ""
        if tactical_ref_hits:
            theory_context = "TACTICAL REFERENCE SYSTEMS:\n"
            for system in tactical_ref_hits[:3]:  # Use top 3 most relevant systems
                theory_context += f"\n{system.get('display_name', 'System')}:\n"
                theory_context += f"  {system.get('description', '')}\n"
                if system.get('key_metrics'):
                    theory_context += f"  Key Metrics: {', '.join(system.get('key_metrics', []))}\n"
                if system.get('player_profile'):
                    theory_context += f"  Player Profile: {system.get('player_profile', '')}\n"

        if football_hackers_hits:
            theory_context += "\n\nFOOTBALL HACKERS PRINCIPLES:\n"
            for hit in football_hackers_hits[:2]:  # Top 2 matching principles
                text = hit.get("text", "")
                source = hit.get("source", "")
                if source == "football-hackers-context":
                    theory_context += f"\n{text[:500]}\n"  # First 500 chars of each chunk

        if not tactical_ref_hits and not football_hackers_hits:
            theory_context = "No specific tactical systems or principles matched the query."

        _phase(
            "3. Retrieve tactical reference (YAML systems + Football Hackers context)",
            t,
            input=f"Query string | tactical concepts: {', '.join(tactical_concepts) if tactical_concepts else 'none — unfiltered'}",
            output=f"{len(tactical_ref_hits)} systems + {len(football_hackers_hits)} context chunks",
            why=(
                "Combines YAML tactical systems with Football Hackers context pack. "
                "Systems provide explicit metrics and player profiles; context provides "
                "scouting principles and operational rules for decision-making."
            ),
            count=len(tactical_ref_hits) + len(football_hackers_hits),
            collection="tactical-reference + football-hackers-context",
            tactical_concepts=tactical_concepts,
        )

        # Phase 4 — Index B: hybrid retrieval from blended collection
        # One unified query returns both season-specific AND career metrics
        t = time.perf_counter()
        is_career = is_career_query(query_str)
        collection_name = "player_profiles_blended"
        search_type = "Blended (season + career)"

        # Query blended collection
        # If career query: no season filter (return all seasons for analysis)
        # If season query: filter to current season for form
        query_season = None if is_career else season

        pool, filter_note = self._retrieve_candidates_blended(
            query_vector,
            eff_position,
            stat_filters,
            limit=25,
            season=query_season,
            exclude_club=exclude_key,
        )

        # Ensure all results have stats dict populated with all stat fields
        stat_fields = [
            "tackles", "tackles_won", "interceptions", "recoveries",
            "assists", "goals", "xg", "xag", "prgc", "prgp",
            "minutes", "appearances", "big_chances_created",
            "improvement_score", "stability_score", "consistency_pct",
        ]
        for hit in pool:
            if "stats" not in hit:
                hit["stats"] = {}
            # Populate stats dict from top-level fields (for blended records)
            for field in stat_fields:
                if field in hit and hit[field] is not None:
                    hit["stats"][field] = hit[field]

        retrieval_msg = (
            f"Blended query (season={query_season}, position={eff_position}, "
            f"stat_filters={stat_filters or '[]'}, exclude={exclude_display})"
        )

        excl_note = f"; excluded from {exclude_display}" if exclude_key else ""
        _phase(
            "4. Retrieve candidate pool (Blended collection)",
            t,
            input=retrieval_msg,
            output=f"{len(pool)} candidate(s) after filtering{excl_note}",
            why=(
                "Milvus hybrid search: scalar filters FIRST, then cosine on results. "
                f"Blended collection returns both season-specific + career metrics in one record. "
                + (
                    "Career query: no season filter, analyzes 3-year trends. "
                    if is_career
                    else f"Season query: filtered to {season} for current form. "
                    "Excludes players from the club you're scouting for. "
                )
                + f"Filters relax step-by-step if needed; applied: {filter_note}."
            ),
            count=len(pool),
            collection=collection_name,
            search_route=search_type,
            is_career_query=is_career,
        )

        # Phase 5 — rank by the requested stat columns (blend semantic + stats)
        t = time.perf_counter()
        ranked, applied_cols = _rerank_by_stats(pool, rank_by)

        # Filter out rapidly declining players (momentum < -0.15 = 15% decline)
        # This surfaces improving/stable talent, not deteriorating players
        filtered = [c for c in ranked if (c.get("progression", {}).get("momentum") or 0) > -0.15]
        # Select top 5 from ranked pool (or unfiltered if all are declining)
        # We retrieve 25 from DB, so 5 gives good coverage without overwhelming the user
        candidates = (filtered or ranked)[:5]  # Expanded to 5 for better coverage

        # Validate candidates have sufficient data for meaningful evaluation
        # Filter out candidates with critical missing data
        valid_candidates = []

        for c in candidates:
            # Check if candidate has meaningful stats or tactical data
            has_stats = bool(c.get("stats") and len(c.get("stats", {})) > 0)
            has_progression = bool(c.get("progression") or c.get("momentum") is not None)
            has_position = c.get("position") and c.get("position") != "Unknown"

            # Only include candidates with at least stats OR progression data AND position
            if (has_stats or has_progression) and has_position:
                valid_candidates.append(c)

        # If we filtered out some candidates, keep the valid ones
        # If all filtered out, fall back to original (with warning)
        if valid_candidates:
            candidates = valid_candidates
        # else: keep original candidates but will show data quality warning

        # Enrich results with missing fields (current_manager, estimated_cost, etc.)
        # Blended collection already has all stats fields, no normalization needed
        for c in candidates:
            _enrich_candidate_defaults(c, season)
        # Aspect-scoped context: pass only the stat facets the query asked about.
        rank_columns = [rb["column"] for rb in rank_by]
        for c in candidates:
            c["context_profile"] = _focused_profile(c, rank_columns)
        # Check if position was relaxed (important for explicit position queries)
        position_was_relaxed = constraints.get("position") and "position relaxed" in filter_note.lower()

        # Check if candidates have insufficient data for evaluation
        candidates_with_incomplete_data = [
            c for c in candidates
            if not (c.get("stats") and len(c.get("stats", {})) > 0)
            and not (c.get("progression") or c.get("momentum") is not None)
        ]
        has_data_quality_issue = bool(candidates_with_incomplete_data)

        retrieval = {
            "search_route": search_type,
            "collection": collection_name,
            "is_career_query": is_career,
            "filter_used": filter_note,
            "position_was_relaxed": position_was_relaxed,
            "position_relaxation_warning": (
                f"⚠️ No {constraints.get('position', '?')} players found matching criteria. "
                "Expanded search to all positions in this category. Results may include different roles."
                if position_was_relaxed else ""
            ),
            "data_quality_warning": (
                f"⚠️ {len(candidates_with_incomplete_data)} of {len(candidates)} candidates lack sufficient stats to evaluate "
                "stability or improvement trajectory. Results may not fully match your criteria."
                if has_data_quality_issue else ""
            ),
            "result_count": len(candidates),
            "pool_size": len(pool),
            "min_minutes": eff_min_minutes,
            "ranked_by": [rb["label"] for rb in rank_by],
            "ranking_columns_applied": applied_cols,
            "excluded_club": exclude_display,
        }
        _phase(
            "5. Rank by stats",
            t,
            input=f"{len(pool)} candidate(s)" + (f"; rank by {', '.join(applied_cols)}" if applied_cols else ""),
            output=f"Top {len(candidates)} after blended ranking",
            why=(
                "The concepts in your query map to stat columns. Rate stats are "
                "compared PER 90 MINUTES (so a high-impact rotation player isn't beaten "
                "just for playing more), with a small-sample guard. We blend the cosine "
                "score 50/50 with those normalized per-90 columns, so a 'high-volume "
                "passing' ask surfaces genuinely high-PrgP/90 players — not just "
                "semantically-similar prose. With no concept cues, pure cosine order is kept."
                + ("" if applied_cols else " (No usable ranking columns for this query/season — using cosine.)")
            ),
            count=len(candidates),
        )

        # Phase 6 — enrichment note (manager/style/cost were joined during retrieval)
        t = time.perf_counter()
        _phase(
            "6. Enrich candidates",
            t,
            input=f"{len(candidates)} player(s): club + name",
            output="Each player + current manager, playing style, and wage cost",
            why=(
                "Manager, playing style and wages don't live in the vector store. "
                "They're joined from reference data (managers.csv, player_wages.csv) "
                "so the recommendation can reason about fit and affordability — "
                "grounded facts, not LLM guesses."
            ),
            count=len(candidates),
        )

        # Phase 7 — similarity projection for the scatter plot
        t = time.perf_counter()
        candidate_names = [c["player_name"] for c in candidates]
        cloud = fetch_player_vectors(season=season, position=eff_position, limit=140)
        if candidate_names:
            cloud += fetch_player_vectors(player_names=candidate_names, limit=10)
        similarity_plot = build_similarity_plot(query_vector, cloud, candidate_names)
        _phase(
            "7. Project similarity (cosine + PCA)",
            t,
            input=f"Query vector + {len(cloud)} player vectors",
            output="2D points with cosine scores",
            why=(
                "Embeddings live in 384 dimensions, which we can't see. PCA flattens "
                "them to 2D so distance ~ dissimilarity, while the cosine score is the "
                "exact number used to rank. This is the search, made visible."
            ),
            count=len(cloud),
        )

        # No grounded data -> be honest instead of letting the model bluff.
        if not candidates:
            phases.append(
                {
                    "name": "8. Generate brief (DSPy)",
                    "input": "—",
                    "output": "Skipped",
                    "why": (
                        "With zero matching players there is nothing to ground a brief "
                        "on, so the LLM is NOT called — preventing confident but invented "
                        "recommendations."
                    ),
                    "ms": 0.0,
                    "skipped": True,
                }
            )
            return {
                "scouting_brief": (
                    "No player records matched this query. The blended collection may "
                    "be empty/unindexed, age filters may be too strict, or season/position/name filters "
                    "matched nothing. Run `python backend/scripts/index_blended.py` to reindex, "
                    "or broaden the filters (e.g., lower age threshold, remove position constraints)."
                ),
                "reasoning": "",
                "candidates": [],
                "context_used": {
                    "tactical_systems": tactical_ref_hits,
                    "player_records": [],
                },
                "retrieval": retrieval,
                "phases": phases,
                "similarity_plot": similarity_plot,
            }

        # Phase 6 — DSPy ChainOfThought generates reasoning + brief
        wants_cost = mentions_cost(query_str)
        stats_context = "\n".join(_format_candidate_for_prompt(c, include_cost=wants_cost) for c in candidates)

        # Build metrics checklist to ground the LLM in actual player stats vs tactical requirements
        metrics_checklist = "\n".join(
            f"{c.get('player_name', 'Unknown')}:\n{build_metrics_checklist(tactical_concepts, c)}"
            for c in candidates
        )

        # DEEPEVAL IMPROVEMENT: Filter checklist to only show query-relevant metrics
        # This improves answer relevancy by removing irrelevant metrics
        filtered_metrics_checklist = _filter_checklist_by_concept(metrics_checklist, tactical_concepts)

        t = time.perf_counter()
        prediction = self.generate_report(
            tactical_context=theory_context or "No tactical theory retrieved.",
            player_stats=stats_context,
            tactical_query=query_str,
            metrics_checklist=filtered_metrics_checklist,  # Use filtered checklist for better relevancy
        )
        _phase(
            "8. Generate brief (DSPy ChainOfThought)",
            t,
            input="Tactical theory + enriched player profiles + the query",
            output="Chain-of-thought reasoning + final scouting brief",
            why=(
                "DSPy's declarative signature (not a hand-written prompt) forces the "
                f"model ({_lm_name()}) to reason step-by-step over ONLY the supplied "
                "context before writing. The reasoning is exposed so every claim is "
                "traceable to the retrieved evidence."
            ),
        )

        # Build response with ALL 15 improvements for UI
        # ============ IMPROVEMENT #1-2: Intent confidence & ambiguity ============
        intent_confidence = confidence_score_intent(constraints)
        ambiguity_warnings = rank_alternatives(query_str)

        # ============ IMPROVEMENT #3: Filter relaxation ladder ============
        filter_ladder = build_filter_relaxation_ladder(season, eff_position, eff_min_minutes, stat_filters)

        # ============ IMPROVEMENT #4: Scalar vs vector balance ============
        scalar_vector_balance = scalar_vs_vector_balance(
            len(pool), len(candidates), 2000  # Approximate corpus size
        )

        # ============ IMPROVEMENT #5: Confidence bands for rankings ============
        ranked_with_confidence = confidence_bands_for_rankings(candidates, rank_by)

        # ============ IMPROVEMENT #6 & #7: Risk flags & comparison alternatives ============
        candidates_with_enrichment = []
        for c in candidates:
            c["risk_flags"] = risk_flags_and_context(c)
            c["similarity_breakdown"] = similarity_breakdown_by_dimension(c, rank_by)
            c["explainability_ledger"] = explainability_ledger(c, tactical_concepts)
            candidates_with_enrichment.append(c)

        # ============ IMPROVEMENT #7: Comparison alternatives ============
        comparison = (
            comparison_alternatives(candidates[0], candidates + pool)
            if candidates else {"top_pick": None, "cheaper_alternative": None, "breakthrough_prospect": None}
        )

        # ============ IMPROVEMENT #14: Player clustering ============
        similar_players_by_candidate = {}
        for c in candidates:
            similar_players_by_candidate[c["player_name"]] = player_clustering_similar_players(
                c, pool, limit=3
            )

        # ============ IMPROVEMENT #15: What-if analysis ============
        what_if_scenarios = what_if_alternative_ranking(candidates, rank_by)

        response = {
            "scouting_brief": prediction.scouting_brief,
            "reasoning": getattr(prediction, "reasoning", ""),
            "candidates": candidates_with_enrichment,
            "context_used": {
                "tactical_systems": tactical_ref_hits,
                "player_records": [c.get("context_profile") or c["stats_summary"] for c in candidates],
            },
            "similarity_plot": similarity_plot,
            "retrieval": retrieval,
            "phases": phases,
            # ============ ALL 15 IMPROVEMENTS ============
            "explainability": {
                # #1: Intent confidence
                "intent_confidence": intent_confidence,
                # #2: Ambiguity warnings
                "query_ambiguities": ambiguity_warnings,
                # #3: Filter relaxation ladder
                "filter_relaxation_ladder": filter_ladder,
                # #4: Scalar vs vector balance
                "hybrid_search_balance": scalar_vector_balance,
                # #5: Confidence bands
                "ranking_confidence_bands": ranked_with_confidence,
                # #6-7: Risk flags & comparison
                "comparison_alternatives": comparison,
                # #14: Similar players clustering
                "player_clustering": similar_players_by_candidate,
                # #15: What-if analysis
                "what_if_scenarios": what_if_scenarios,
            },
        }

        # Add impact analysis showing why these players were chosen
        if candidates:
            response["impact_analysis"] = {
                "query_type": "career_progression" if is_career else "season_form",
                "primary_driver": extract_impact_metrics(candidates[0], is_career),
                "runner_up_driver": extract_impact_metrics(candidates[1], is_career) if len(candidates) > 1 else None,
                "collection_used": collection_name,
            }

        return response


def init_dspy() -> bool:
    """
    Configures DSPy with the default language model.

    Supports either OpenRouter or OpenAI, selected by which API key is present
    (OpenRouter takes precedence). The model can be overridden via the
    SCOUTINTEL_LM_MODEL env var. Returns True if a real LM was configured,
    False otherwise. When no key is present the engine is left unconfigured;
    callers should surface a clear error rather than letting inference fail
    deep inside DSPy.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    model_override = os.environ.get("SCOUTINTEL_LM_MODEL")

    if openrouter_key:
        # LiteLLM routes any "openrouter/..." model to https://openrouter.ai/api/v1.
        model = model_override or "openrouter/openai/gpt-4o"
        lm = dspy.LM(
            model=model,
            api_key=openrouter_key,
            api_base="https://openrouter.ai/api/v1",
        )
        dspy.settings.configure(lm=lm)
        return True

    if openai_key:
        model = model_override or "openai/gpt-4o"
        lm = dspy.LM(model=model, api_key=openai_key)
        dspy.settings.configure(lm=lm)
        return True

    print("Warning: no OPENROUTER_API_KEY or OPENAI_API_KEY found. /query will be unavailable until one is set.")
    return False


def is_dspy_ready() -> bool:
    """Returns True if a language model is configured for inference."""
    return getattr(dspy.settings, "lm", None) is not None
