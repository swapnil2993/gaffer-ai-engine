import os
import re
import time
from typing import Any, Dict, List

import dspy
import numpy as np

from backend.app.database import (
    CAREER_COLLECTION,
    EMBEDDING_MODEL_NAME,
    STATS_COLLECTION,
    THEORY_COLLECTION,
    VECTOR_DIM,
    fetch_player_history,
    fetch_player_vectors,
    get_embedding,
    get_embeddings,
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
}


def _focused_profile(candidate, columns):
    """Aspect-scoped player context: identity + ONLY the stat facets the query
    asked about (the rank_by columns). Removes off-topic stats so the context
    fits the query. Falls back to the full prose when there are no concept cues.
    """
    if not columns:
        return candidate.get("stats_summary", "")
    stats = candidate.get("stats") or {}
    base = f"{candidate['player_name']} ({candidate['position']}, {candidate['current_club']}, {candidate['season']})"
    mins, apps = stats.get("minutes"), stats.get("appearances")
    if mins:
        base += f", {int(mins)} mins"
    elif apps:
        base += f", {int(apps)} apps"
    facets = [f"{int(v)} {_STAT_LABELS.get(col, col)}" for col in columns if (v := stats.get(col)) is not None]
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

    Your recommendation MUST be grounded exclusively in the metrics_checklist.
    Every claim is verified against what appears in that checklist.

    RULES (Verification-First):
      1. Recommend the single best-fit player (name them) and, briefly, a runner-up.

      2. For EACH tactical concept in the query, cite ONLY metrics from the
         metrics_checklist. Do NOT infer thresholds or metrics not listed there.

      3. For EACH claim, show explicit verification:
         CORRECT: "Tackles 160 (target > 150) ✓"
         WRONG:   "Excellent defensive positioning" (not in checklist)

      4. Build the brief step-by-step:
         Step 1: Which metrics are in the metrics_checklist? (Note them)
         Step 2: Which have ✓ (pass)? (These are your evidence)
         Step 3: How do they address the query? (Make the connection)
         Step 4: Write brief citing ONLY metrics with ✓

      5. When career data is provided in metrics_checklist:
         - For "improving": cite momentum ✓ and trend
         - For "consistent": cite consistency metric and variance
         - For "declining": cite negative momentum with warning

      6. CRITICAL RULES (Prevents Faithfulness Issues):
         - DO NOT cite any metric not in the metrics_checklist
         - DO NOT invent thresholds (e.g., "> 150 tackles" if not stated)
         - DO NOT add generic observations like "strong positioning sense"
         - DO NOT infer unstated requirements
         - ONLY use exact numbers from the checklist

      7. Discuss wages ONLY if explicitly in player profile.

      8. No generic football platitudes. ONLY cite verifiable metrics.

    FORMAT YOUR REASONING as 4-7 bullets showing verification:
      - Query asks for X. Checklist shows metric Y with threshold Z.
      - Player achieves value W. ✓ or ✗
      - Why is this player best fit?

    FORMAT YOUR BRIEF as:
      [Player Name] is the best fit.
      - [Query concept]: [Metric] [Value] (target > [Threshold]) ✓
      - [Query concept]: [Metric] [Value] (target > [Threshold]) ✓
      - (Repeat for each concept; only include if ✓)
    """

    tactical_context = dspy.InputField(
        desc="Modern tactical systems with required metrics. "
             "Use for background understanding only; actual thresholds come from metrics_checklist."
    )

    player_stats = dspy.InputField(
        desc="Candidate player profiles with statistics. "
             "Reference only to verify metrics_checklist values."
    )

    tactical_query = dspy.InputField(
        desc="Specific tactical question. "
             "Determines which metrics in the checklist are relevant."
    )

    metrics_checklist = dspy.InputField(
        desc="THE GROUND TRUTH. This is EVERY metric you should cite. "
             "Format: 'CONCEPT: [name] > [threshold] → Player: [value] ✓/✗'. "
             "CRITICAL: You may ONLY cite metrics that appear here with exact values. "
             "If a metric is not in this checklist, do NOT mention it, even if logical. "
             "Example of CORRECT: 'TACKLES: > 150 → Player: 160 ✓'. "
             "Example of WRONG: 'Strong defensive reading' (not in checklist, unverifiable)."
    )

    reasoning = dspy.OutputField(
        desc="4-7 bullets showing verification for each claim. "
             "For each: 'Query concept X requires metric Y. Checklist shows threshold Z. Player has W. ✓/✗'. "
             "Only mention metrics from the checklist."
    )

    scouting_brief = dspy.OutputField(
        desc="A tight recommendation citing ONLY checklist metrics with verification. "
             "Format: '[Player]: [Concept]: [Metric Value] (target > X) ✓'. "
             "CRITICAL RULES: (1) Do NOT cite any value that doesn't appear in metrics_checklist. "
             "(2) Do NOT infer or assume thresholds. (3) Every claim must be verifiable. "
             "(4) No generic observations like 'strong positioning'. (5) Only include metrics with ✓. "
             "Example CORRECT: 'Tackles 160 (target > 150) ✓ + Interceptions 45 (target > 40) ✓'. "
             "Example WRONG: 'Exceptional defensive awareness' (not verifiable against checklist)."
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
        "tactical_suitability": tactical_fit,
        "manager_tactics": manager_tactics,
        "tactical_alignment": tactical_alignment,
        "estimated_cost": get_estimated_cost(player_name),
        "relevance_score": round(float(_hit_field(hit, "distance", 0.0) or 0.0), 4),
    }


def _format_candidate_for_prompt(c: Dict[str, Any], include_cost: bool = False) -> str:
    """Render a candidate as a compact, grounded block for the LLM context."""
    # Wage/cost is included ONLY when the query asks about it (include_cost) — for
    # purely tactical queries it's off-topic and dilutes the brief.
    manager = c["current_manager"] or "manager unknown"
    style = c["manager_playing_style"]
    manager_line = f"manager: {manager}" + (f", style: {style}" if style else "")
    cost_line = ""
    if include_cost:
        cost = c.get("estimated_cost") or {}
        wage = cost.get("annual_wages") or "wage unknown"
        weekly = cost.get("weekly_wages")
        cost_line = f" | wage cost: {wage}/yr" + (f" ({weekly}/wk)" if weekly else "")
    # Use the aspect-scoped profile when available (query had concept cues).
    profile = c.get("context_profile") or c["stats_summary"]
    return (
        f"- {c['player_name']} | {c['position']} | {c['current_club']} "
        f"({manager_line}) | season {c['season']}{cost_line}\n"
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
    that aren't related to the query's tactical concepts.
    """
    if not concepts or not checklist_text:
        return checklist_text

    # Map concepts to checklist section keywords
    concept_keywords = {
        "high_press": ["PRESSING", "HIGH PRESS", "PRESS"],
        "ball_progression": ["PROGRESSION", "BALL PROGRESSION", "PASSING"],
        "creative": ["CREATIVE", "ASSISTS", "PLAYMAKING", "KEY PASS"],
        "defensive_transition": ["DEFENSIVE", "DEFENSIVE TRANSITION", "TACKLES", "INTERCEPTIONS", "RECOVERY"],
        "goal_threat": ["GOAL", "GOALS", "THREAT", "FINISHING"],
        "momentum": ["MOMENTUM", "TRAJECTORY", "TREND"],
    }

    # Collect relevant keywords from query concepts
    relevant_keywords = set()
    for concept in concepts:
        keywords = concept_keywords.get(concept.lower(), [])
        relevant_keywords.update(kw.upper() for kw in keywords)

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


class ScoutIntelRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_report = dspy.ChainOfThought(ScoutingReportSignature)

    def _retrieve_candidates_career(
        self,
        query_vector,
        position,
        player_name,
        stat_filters,
        limit,
    ):
        """Retrieve players from career aggregates (multi-season profiles).

        No season filtering — career aggregates span all years. Simpler relaxation
        ladder since we're dealing with consolidated data.
        """
        sf = list(stat_filters or [])
        ladder = [
            (position, player_name, sf, "all constraints (incl. stat thresholds)"),
            (position, player_name, [], "stat thresholds relaxed"),
            (position, None, [], "name relaxed"),
            (None, None, [], "position relaxed"),
        ]
        seen = set()
        for p, n, f, note in ladder:
            key = (p, n, tuple(f))
            if key in seen:
                continue
            seen.add(key)
            hits = query_career(query_vector, position=p, player_name=n, stat_filters=f, limit=limit)
            if hits:
                return hits, note
        return [], "no career profiles matched"

    def _retrieve_candidates(
        self,
        query_vector,
        season,
        position,
        player_name,
        min_minutes,
        stat_filters,
        limit,
    ):
        """Retrieve players, relaxing filters step-by-step until something matches.

        Hybrid: Milvus applies the scalar filters (season / position / name /
        min_minutes / stat thresholds) FIRST and ranks only the rows that qualify
        by cosine similarity. We start fully constrained and relax along a ladder —
        dropping the most aggressive constraint first — so the English-derived stat
        requirements are honoured when possible but never cause a silent empty set.
        """
        sf = list(stat_filters or [])
        ladder = [
            (
                season,
                position,
                player_name,
                min_minutes,
                sf,
                "all constraints (incl. stat thresholds)",
            ),
            (season, position, player_name, min_minutes, [], "stat thresholds relaxed"),
            (season, position, player_name, None, [], "min-minutes relaxed"),
            (season, position, None, None, [], "name relaxed"),
            (season, None, None, None, [], "position relaxed"),
            (None, None, player_name, None, [], "season relaxed (name kept)"),
            (None, None, None, None, [], "unfiltered (all relaxed)"),
        ]
        seen = set()
        for s, p, n, m, f, note in ladder:
            key = (s, p, n, m, tuple(f))
            if key in seen:
                continue
            seen.add(key)
            hits = query_stats(query_vector, s, p, n, min_minutes=m, stat_filters=f, limit=limit)
            if hits:
                return [_build_candidate(h) for h in hits], note
        return [], "no matches"

    def forward(
        self,
        query_str: str,
        season: str = None,
        position: str = None,
        player_name: str = None,
        min_minutes: float = None,
        scouting_for: str = None,
    ) -> Dict[str, Any]:
        """
        Retrieves tactical theory + candidate players, enriches each candidate
        with current manager and wage cost, and generates a grounded brief.

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
        constraints = parse_query_constraints(query_str, position_hint=position)
        eff_position = constraints["position"]
        rank_by = constraints["rank_by"]
        stat_filters = list(constraints["stat_filters"])

        # Extract tactical concepts and map to FBref filters
        tactical_concepts, tactical_filters = map_query_to_filters(query_str)
        stat_filters.extend(
            [f"{k}>={v}" for k, v in tactical_filters.items()]
        )
        # min_minutes precedence: explicit arg > parsed from query > season default.
        if min_minutes is not None:
            eff_min_minutes = min_minutes
        elif constraints["min_minutes"] is not None:
            eff_min_minutes = float(constraints["min_minutes"])
        elif season in ("2024/2025", "2025/2026"):
            eff_min_minutes = 270.0  # ~3 full matches; cuts tiny-sample noise
        else:
            eff_min_minutes = None
        # Club you're scouting FOR — its own players are excluded from results.
        # Explicit `scouting_for` wins; otherwise infer a club named in the query.
        exclude_key = normalize_club(scouting_for) if scouting_for else detect_club(query_str)
        exclude_display = scouting_for or (exclude_key.title() if exclude_key else None)
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
        # Use tactical concepts extracted from query to pre-filter book chunks
        theory_results = query_theory(query_str, limit=3, concepts=tactical_concepts if tactical_concepts else None)
        
        theory_hits = []
        for r in theory_results:
            hit_text = _hit_field(r, "text", "")
            # Compress for the LLM prompt, but keep metadata for the UI
            compressed = _compress_theory(query_vector, [hit_text])[0]
            theory_hits.append({
                "text": compressed,
                "heading": _hit_field(r, "heading", "General"),
                "era": _hit_field(r, "era"),
                "formation": _hit_field(r, "formation"),
                "style": _hit_field(r, "possession_style_tag"),
                "line_height": _hit_field(r, "defensive_line_height"),
                "weights": _hit_field(r, "fbref_metric_weights", {})
            })

        theory_context = "\n".join(h["text"] for h in theory_hits)
        _phase(
            "3. Retrieve + compress tactical theory (Index A)",
            t,
            input=f"Query vector | tactical concepts: {', '.join(tactical_concepts) if tactical_concepts else 'none — unfiltered'}",
            output=f"{len(theory_hits)} chunk(s), compressed to query-relevant sentences",
            why=(
                "Chunks are tagged with tactical concepts (high_press, ball_progression, etc.) "
                "at index time. We pre-filter by the concepts detected in your query "
                f"({', '.join(tactical_concepts) if tactical_concepts else 'none — unfiltered'}), "
                "so the search hits relevant chapters instead of random blocks. Then we keep only the "
                "sentences in each chunk most relevant to the query (compression)."
            ),
            count=len(theory_hits),
            collection=THEORY_COLLECTION,
            tactical_concepts=tactical_concepts,
        )

        # Phase 4 — Index B: hybrid retrieval. Pull a POOL (not just the final 3)
        # so the stat re-rank in phase 5 has candidates to sort.
        # Route to career aggregates if the query asks for progression/trends.
        t = time.perf_counter()
        is_career = is_career_query(query_str)
        collection_name = CAREER_COLLECTION if is_career else STATS_COLLECTION

        if is_career:
            # Career query: no season-specific filtering
            raw_results, filter_note = self._retrieve_candidates_career(
                query_vector,
                eff_position,
                player_name,
                stat_filters,
                limit=25,
            )
            pool = raw_results  # Career results don't have _build_candidate wrapping
            retrieval_msg = f"Query vector + filters (position={eff_position}, name={player_name})"
        else:
            # Season-specific query
            pool, filter_note = self._retrieve_candidates(
                query_vector,
                season,
                eff_position,
                player_name,
                eff_min_minutes,
                stat_filters,
                limit=25,
            )
            retrieval_msg = (
                f"Query vector + filters (season={season}, position={eff_position}, "
                f"name={player_name}, min_minutes={eff_min_minutes}, "
                f"stat_filters={stat_filters or '[]'}, exclude={exclude_display})"
            )

        # Exclude the club you're scouting for — you want external fits.
        excluded_count = 0
        if exclude_key and not is_career:
            before = len(pool)
            pool = [c for c in pool if normalize_club(c.get("current_club")) != exclude_key]
            excluded_count = before - len(pool)
        excl_note = f"; excluded {excluded_count} from {exclude_display}" if exclude_key and not is_career else ""
        _phase(
            "4. Retrieve candidate pool (Index B, hybrid)",
            t,
            input=retrieval_msg,
            output=f"{len(pool)} candidate(s) after filtering{excl_note}",
            why=(
                "Milvus applies the scalar filters FIRST and computes cosine "
                "similarity only on the rows that qualify. "
                + (
                    "Career aggregates span all seasons, so this retrieves 3-year profiles. "
                    if is_career
                    else f"We then drop any players from the club you're scouting for. "
                )
                + f"Filters relax step-by-step if too strict; here: {filter_note}."
            ),
            count=len(pool),
            collection=collection_name,
        )

        # Phase 5 — rank by the requested stat columns (blend semantic + stats)
        t = time.perf_counter()
        ranked, applied_cols = _rerank_by_stats(pool, rank_by)

        # Filter out rapidly declining players (momentum < -0.15 = 15% decline)
        # This surfaces improving/stable talent, not deteriorating players
        filtered = [c for c in ranked if (c.get("progression", {}).get("momentum") or 0) > -0.15]
        candidates = (filtered or ranked)[:3]  # Fallback to unfiltered if all are declining
        # Aspect-scoped context: pass only the stat facets the query asked about.
        rank_columns = [rb["column"] for rb in rank_by]
        for c in candidates:
            c["context_profile"] = _focused_profile(c, rank_columns)
        retrieval = {
            "filter_used": filter_note,
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
                    "No player records matched this query. The stats database may "
                    "be empty/unindexed, or the season/position/name filters matched "
                    "nothing. Run `uv run python -m backend.scripts.index_data` to "
                    "populate it, or broaden the filters."
                ),
                "reasoning": "",
                "candidates": [],
                "context_used": {
                    "tactical_theory": theory_hits,
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
            f"{c['player_name']}:\n{build_metrics_checklist(tactical_concepts, c)}"
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

        # Build response with impact analysis for UI
        response = {
            "scouting_brief": prediction.scouting_brief,
            "reasoning": getattr(prediction, "reasoning", ""),
            "candidates": candidates,
            "context_used": {
                "tactical_theory": theory_hits,
                "player_records": [c.get("context_profile") or c["stats_summary"] for c in candidates],
            },
            "similarity_plot": similarity_plot,
            "retrieval": retrieval,
            "phases": phases,
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
