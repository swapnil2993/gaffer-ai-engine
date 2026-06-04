"""Turn a plain-English scouting query into structured retrieval intent.

This is what makes the natural-language query "consider the stats". We split
intent into two kinds:

* **Hard filters** — explicit numbers ("more than 10 goals") become Milvus scalar
  filters applied BEFORE the vector search, so similarity is only computed on the
  rows that already qualify (cheap + precise).
* **Ranking signals** — qualitative concepts ("high-volume passing", "wins the
  ball back") map to stat COLUMNS (PrgP, Tackles/Interceptions). Rather than
  inventing a brittle threshold, we rank the semantically-relevant candidates by
  a blend of cosine similarity and those columns. Robust to phrasing.

Deterministic (regex + a curated concept→column map): no latency, no LLM cost,
and fully transparent — every mapping is reported in `matched` for the UI.
"""

import re
from typing import Dict, List, Optional

# Keywords that indicate a query is asking for career/multi-year data
_CAREER_KEYWORDS = (
    "career",
    "improving",
    "declining",
    "progression",
    "trajectory",
    "three year",
    "3 year",
    "across seasons",
    "consistent",
    "consistency",
    "trend",
    "momentum",
    "long-term",
    "multi-year",
    "over time",
    "this season vs",
    "upward trajectory",
    "upward curve",
    "year-on-year",
    "baseline",
    "historically",
)

# Most specific position words first (longest matches first to avoid partial matches).
_POSITION_KEYWORDS = [
    (("goalkeeper", "keeper", "shot-stopper", "goalie", "number 1"), "GK"),
    # Centre backs: be specific (handle both British and American spellings)
    (("centre-back", "center-back", "centre back", "center back", "centre half", "cb"), "CB"),
    # Fullbacks: left side
    (("left-back", "leftback", "left back", "lb"), "LB"),
    # Fullbacks: right side
    (("right-back", "rightback", "right back", "rb"), "RB"),
    # Wing-backs
    (("wing-back", "wingback", "wing back", "lwb", "rwb"), "LWB"),
    # Generic defender (maps to multiple positions via filter relaxation)
    (
        (
            "defender",
            "full-back",
            "fullback",
            "full back",
            "back four",
            "defensive line",
        ),
        "DF",
    ),
    # Specific midfielder types
    # Defensive midfielder / holding midfielder
    (("deep-lying", "holding mid", "regista", "pivot", "number 6", "dm", "defensive mid", "defensive-minded"), "DM"),
    # Attacking midfielder / playmaker
    (("playmaker", "number 10", "attacking mid", "am", "creative"), "AM"),
    # Central midfielder / box-to-box
    (("box-to-box", "box to box", "number 8", "central mid", "cm"), "CM"),
    # Winger (can be MF or FW depending on formation)
    (("left winger", "right winger", "lw", "rw"), "W"),
    # Generic midfielder
    (("midfielder", "midfield"), "MF"),
    # Specific forward types
    # Striker / centre-forward
    (("striker", "centre-forward", "center-forward", "centre forward", "number 9", "st"), "ST"),
    # Inside forward / false nine
    (("inside forward", "false nine", "false-nine", "if"), "IF"),
    # Winger (can be FW or MF)
    (("winger",), "W"),
    # Generic forward/attacker
    (("forward", "attacker", "front man", "frontman"), "FW"),
]

# Qualitative concept -> stat columns to RANK by (label shown in the UI).
# Only columns we actually index are listed; ranking ignores absent ones.
_CONCEPT_COLUMNS = [
    (
        (
            "goalscorer",
            "goal scorer",
            "prolific",
            "clinical",
            "finisher",
            "lethal",
            "poacher",
            "scoring",
            "goal threat",
        ),
        [("goals", "Goals"), ("xg", "xG")],
    ),
    (
        (
            "creative",
            "playmaker",
            "chance creat",
            "creator",
            "provider",
            "assist",
            "vision",
            "through ball",
            "key pass",
        ),
        [
            ("assists", "Assists"),
            ("xag", "xAG"),
            ("big_chances_created", "Big chances created"),
        ],
    ),
    (
        (
            "passing",
            "passer",
            "distribution",
            "ball progression",
            "progressive pass",
            "circulat",
            "tempo",
            "metronom",
        ),
        [("prgp", "Progressive passes")],
    ),
    (
        (
            "carry",
            "carrier",
            "dribbl",
            "drive",
            "run with the ball",
            "progressive carr",
            "ball-carrying",
            "ball carrying",
            "gets forward",
            "get forward",
            "supports attack",
            "attacking contribution",
        ),
        [("prgc", "Progressive carries"), ("prgp", "Progressive passes")],
    ),
    (
        (
            "win the ball",
            "wins the ball",
            "winning the ball",
            "win it back",
            "ball back",
            "ball-winner",
            "ball winner",
            "ball-winning",
            "tackl",
            "break up play",
            "breaks up play",
            "breaking up play",
            "defensive midfield",
            "destroyer",
            "intercept",
            "regain",
            "ball recovery",
            "recoveries",
            "recover possession",
            "recover the ball",
            "recovery",
            "recovery run",
            "screen the defen",
            # pressing & transition vocabulary (defensive transition = win the ball back)
            "press",
            "pressing",
            "high press",
            "high-press",
            "counter-press",
            "counterpress",
            "gegenpress",
            "gegenpressing",
            "defensive transition",
            "transition",
            "transitions",
            "out of possession",
            "off the ball",
            "defensive work",
            "defensive duties",
            "win possession",
            "winning possession",
            "press resistant",
            "work rate",
            "work hard",
            "works hard",
        ),
        [
            ("tackles_won", "Tackles won"),
            ("tackles", "Tackles"),
            ("interceptions", "Interceptions"),
            ("recoveries", "Recoveries"),
        ],
    ),
    (
        ("shot-stopper", "saves", "save", "clean sheet"),
        [("saves", "Saves"), ("clean_sheets", "Clean sheets")],
    ),
    (
        (
            "1v1 defending",
            "1v1",
            "one-on-one",
            "one on one",
            "duel",
            "dueling",
            "individual defending",
            "man-marking",
            "marking",
        ),
        [("tackles", "Tackles"), ("tackles_won", "Tackles won"), ("interceptions", "Interceptions")],
    ),
    (
        (
            "experienced",
            "regular starter",
            "ever-present",
            "game time",
            "minutes",
            "established",
            "mainstay",
        ),
        [("minutes", "Minutes"), ("appearances", "Appearances")],
    ),
    (
        (
            "improving",
            "improvement",
            "improving over time",
            "getting better",
            "development",
            "trajectory",
            "momentum",
            "progress",
            "upward trend",
            "on an upward curve",
            "not regressing",
            "consistency",
            "consistent",
            "stable",
            "reliable",
            "over 3 years",
            "over time",
            "long-term",
            "young",
            "emerging",
            "rising star",
            "rising",
            "growing",
            "growth",
            "improvement trend",
            "improved",
            "getting stronger",
            "development trajectory",
            "craft",
            "form trajectory",
        ),
        [("improvement_score", "Improvement"), ("stability_score", "Stability"), ("consistency_pct", "Consistency")],
    ),
]

_STAT_WORDS = {
    "goals": ("goals", "goal", "strikes"),
    "assists": ("assists", "assist"),
    "minutes": ("minutes", "minute", "mins"),
    "appearances": ("appearances", "apps", "games", "matches"),
    "age": ("over", "aged", "year old", "years old"),  # "over 30", "30+ years old", etc.
}
_SANITY_MAX = {"goals": 60, "assists": 50, "minutes": 4000, "appearances": 60, "age": 50}

# ---------------------------------------------------------------------------
# "Query Target Headers": a controlled tactical-concept taxonomy used to GROUP
# theory chunks at index time AND to target the matching group at query time.
# A search for "defensive transitions" then hits chunks grouped under
# "Pressing & defensive transitions" instead of a random passing-metrics block.
# The SAME taxonomy classifies chunks and queries, so the two line up.
# ---------------------------------------------------------------------------
TACTICAL_CONCEPTS = {
    "Pressing & defensive transitions": (
        "press",
        "pressing",
        "gegenpress",
        "counter-press",
        "counterpress",
        "transition",
        "win the ball",
        "winning the ball",
        "tackle",
        "tackling",
        "interception",
        "intercept",
        "regain",
        "recover",
        "ball-winner",
        "out of possession",
        "off the ball",
        "defensive work",
        "harry",
        "hound",
        "high press",
        "work rate",
        "work hard",
        "recovery run",
        "1v1 defending",
    ),
    "Build-up & possession": (
        "build-up",
        "build up",
        "possession",
        "passing",
        "circulation",
        "progressive",
        "tempo",
        "playmaker",
        "deep-lying",
        "tiki-taka",
        "positional play",
        "rondo",
        "retain",
        "patient",
        "metronome",
    ),
    "Chance creation & creativity": (
        "chance creation",
        "creative",
        "creator",
        "assist",
        "through ball",
        "final third",
        "key pass",
        "vision",
        "playmaking",
        "number 10",
        "between the lines",
        "incisive",
    ),
    "Finishing & goalscoring": (
        "goalscorer",
        "goal scorer",
        "finishing",
        "finisher",
        "striker",
        "shooting",
        "poacher",
        "clinical",
        "centre-forward",
        "centre forward",
    ),
    "Wide play & crossing": (
        "winger",
        "wide",
        "wing",
        "flank",
        "full-back",
        "fullback",
        "overlap",
        "cross",
        "crossing",
        "byline",
        "wing-back",
        "touchline",
    ),
    "Defending & structure": (
        "defend",
        "back line",
        "back four",
        "offside trap",
        "low block",
        "defensive line",
        "clearance",
        "block",
        "marking",
        "catenaccio",
        "sweeper",
        "libero",
        "compact",
        "defensive intensity",
        "defensive solidity",
        "1v1",
    ),
    "Attacking fullback / wing-back": (
        "fullback",
        "full-back",
        "left-back",
        "right-back",
        "wing-back",
        "wingback",
        "gets forward",
        "attacking fullback",
        "progressive",
        "assist",
        "inverted",
    ),
    "Player development & consistency": (
        "young",
        "emerging",
        "rising star",
        "improving",
        "trajectory",
        "growth",
        "consistent",
        "stable",
        "reliable",
        "improvement",
        "development",
        "veteran",
        "experienced",
        "established",
    ),
}


def _concept_scores(text: str) -> dict:
    low = (text or "").lower()
    return {c: sum(low.count(k) for k in kws) for c, kws in TACTICAL_CONCEPTS.items()}


def classify_tactical_concept(text: str) -> str:
    """Assign a theory chunk to its dominant tactical concept (the group header)."""
    scores = _concept_scores(text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General play"


def detect_query_concept(query: str) -> Optional[str]:
    """The tactical concept a query targets, or None (then retrieval isn't filtered)."""
    scores = _concept_scores(query)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


_COST_WORDS = (
    "wage",
    "wages",
    "salary",
    "salaries",
    "cost",
    "budget",
    "afford",
    "affordab",
    "cheap",
    "expensive",
    "value for money",
    "good value",
    "price",
    "fee",
    "/week",
    "per week",
    "/wk",
    "money",
    "financ",
    "£",
    "$",
    "€",
)


def mentions_cost(query: str) -> bool:
    """True if the query asks about wages/budget/value — then the brief may discuss cost."""
    q = (query or "").lower()
    return any(w in q for w in _COST_WORDS)


def is_career_query(query: str) -> bool:
    """True if the query asks about career arc, progression, or multi-year trends.

    Career queries should hit the player_career_collection (3-year aggregates)
    instead of season-specific stats.
    """
    q = (query or "").lower()
    return any(w in q for w in _CAREER_KEYWORDS)


def _detect_position(q: str) -> Optional[str]:
    for words, code in _POSITION_KEYWORDS:
        # Use word boundary matching to avoid false matches (e.g., "rw" in "forward")
        # For multi-word keywords (e.g., "center back"), don't require trailing boundary
        # since the plural form "center backs" would otherwise not match
        for w in words:
            # Start with word boundary; for end, check if it's multi-word
            if " " in w or "-" in w:
                # Multi-word keyword: match start boundary, but allow trailing inflections
                if re.search(rf'\b{re.escape(w)}', q):
                    return code
            else:
                # Single word: use full word boundary matching
                if re.search(rf'\b{re.escape(w)}\b', q):
                    return code
    return None


def _find_threshold(q: str, words) -> Optional[int]:
    for w in words:
        for pat in [
            rf"(?:more than|over|at least|minimum(?: of)?|min|upwards of|north of)\s+(\d+)\+?\s+{w}\b",
            rf"\b(\d+)\s*\+\s*{w}\b",
            rf"(?:{w}\s*)?\((\d+)\+?\)",  # Handle "goals (5+)" format
            rf"\b(\d+)\s+or\s+more\s+{w}\b",
            rf"(?:scored|netted|notched|with|having|registered)\s+(\d+)\s+{w}\b",
            rf"\b(\d+)\s+{w}\b",
        ]:
            m = re.search(pat, q)
            if m:
                return int(m.group(1))
    return None


def parse_query_constraints(query: str, position_hint: Optional[str] = None) -> Dict:
    """Return position, hard stat filters, and `rank_by` columns from the query."""
    q = (query or "").lower()
    matched: List[str] = []

    position = position_hint or _detect_position(q)
    if position and not position_hint:
        matched.append(f"position = {position} (from wording)")

    # Explicit numeric thresholds -> hard filters.
    mins: Dict[str, int] = {}
    for field, words in _STAT_WORDS.items():
        val = _find_threshold(q, words)
        if val is not None and 0 < val <= _SANITY_MAX.get(field, val):
            mins[field] = val
            matched.append(f"{field} ≥ {val} (hard filter)")
    stat_filters = [f"{f} >= {v}" for f, v in mins.items() if f in ("goals", "assists", "appearances", "age")]

    # Qualitative concepts -> ranking columns (weighted, de-duplicated).
    rank_by: List[Dict] = []
    seen_cols = set()
    for phrases, cols in _CONCEPT_COLUMNS:
        if any(p in q for p in phrases):
            labels = []
            for col, label in cols:
                if col not in seen_cols:
                    rank_by.append({"column": col, "label": label})
                    seen_cols.add(col)
                    labels.append(label)
            if labels:
                matched.append(f"rank by {', '.join(labels)} (from wording)")

    return {
        "position": position,
        "min_goals": mins.get("goals"),
        "min_assists": mins.get("assists"),
        "min_minutes": mins.get("minutes"),
        "min_age": mins.get("age"),
        "stat_filters": stat_filters,
        "rank_by": rank_by,
        "matched": matched,
    }
