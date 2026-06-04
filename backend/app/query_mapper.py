"""Map tactical concepts in queries to FBref metric filters.

This is the lightweight knowledge graph — a simple concept → metric mapper.
No LLM at indexing time. Just a curated set of tactical patterns that route
to specific FBref stats filters.
"""

from typing import Dict, List, Tuple

TACTICAL_FILTERS = {
    # Concept: tuple of (keywords_to_match, fbref_filters)
    "high_press": (
        ["high press", "aggressive press", "gegenpressing", "gegenpressure"],
        {"Succ_Press": 0.75, "Tkl_Att_3rd": 0.70},
    ),
    "defensive_transition": (
        ["defensive transition", "transition defense", "recovery", "regain"],
        {"Tkl": 1.0, "Int": 0.80, "Recoveries": 0.75},
    ),
    "ball_progression": (
        ["ball progression", "progressive", "carries", "pass completion", "progression"],
        {"PrgC": 0.60, "PrgP": 0.60},
    ),
    "creativity": (
        ["creative", "assist", "chance creation", "key pass", "key passes", "through ball", "through balls", "unlock", "unlock defense"],
        {"Ast": 0.50, "KP": 0.70},
    ),
    "goal_threat": (
        ["goal scorer", "prolific", "finishing", "shot"],
        {"Gls": 0.50, "xG": 0.60},
    ),
    "physical": (
        ["strong", "powerful", "athletic", "pace", "speed"],
        {"Tkl": 0.70},  # Proxy: defenders who tackle a lot tend to be physical
    ),
}


def extract_tactical_concepts(query: str) -> List[str]:
    """Extract matched tactical concepts from a query string."""
    query_lower = query.lower()
    matched = []
    for concept, (keywords, _) in TACTICAL_FILTERS.items():
        if any(kw in query_lower for kw in keywords):
            matched.append(concept)
    return matched


def build_fbref_filters(concepts: List[str]) -> Dict[str, float]:
    """Merge FBref filters from matched tactical concepts.

    When multiple concepts are matched, take the max weight per metric
    (e.g., if both high_press and defensive_transition mention Tkl, use
    the higher weight).
    """
    merged = {}
    for concept in concepts:
        if concept in TACTICAL_FILTERS:
            _, filters = TACTICAL_FILTERS[concept]
            for metric, weight in filters.items():
                merged[metric] = max(merged.get(metric, 0.0), weight)
    return merged


def map_query_to_filters(query: str) -> Tuple[List[str], Dict[str, float]]:
    """End-to-end: query → tactical concepts → FBref filters."""
    concepts = extract_tactical_concepts(query)
    filters = build_fbref_filters(concepts)
    return concepts, filters


def extract_manager_tactics(manager_style: str) -> List[str]:
    """Extract tactical concepts from a manager's 'Evolution Pyramid Placement' style.

    Examples: 'Gegenpressing', 'Possession-based', 'Counter-attacking', etc.
    """
    if not manager_style:
        return []

    style_lower = manager_style.lower()
    tactics = []

    if any(w in style_lower for w in ["gegenpressing", "pressing", "counter-attack", "gegenpressure"]):
        tactics.append("high_press")
    if any(w in style_lower for w in ["possession", "ball", "control"]):
        tactics.append("ball_progression")
    if any(w in style_lower for w in ["direct", "long ball", "transition"]):
        tactics.append("defensive_transition")
    if any(w in style_lower for w in ["low block", "defensive", "defensive-minded"]):
        tactics.append("low_block")
    if any(w in style_lower for w in ["high line", "aggressive"]):
        tactics.append("high_press")

    return tactics if tactics else []


def compare_alignment(manager_tactics: List[str], player_tactics: List[str]) -> Dict[str, str]:
    """Compare manager's system with player's suitability.

    Returns: {
        'aligned': ['high_press'],      # Both manager and player emphasize this
        'conflict': [],                 # Manager needs X but player doesn't suit it
        'surplus': ['creativity'],      # Player is good at X but manager doesn't use it
    }
    """
    manager_set = set(manager_tactics)
    player_set = set(player_tactics)

    return {
        "aligned": list(manager_set & player_set),  # Both have it
        "conflict": list(manager_set - player_set),  # Manager needs, player lacks
        "surplus": list(player_set - manager_set),   # Player has, manager doesn't use
    }


def analyze_tactical_suitability(stats: Dict[str, float]) -> List[str]:
    """Analyze a player's stats and return which tactical systems suit them best.

    Args:
        stats: dict with keys like 'Succ_Press', 'Tkl', 'Int', 'PrgC', 'PrgP', 'Ast', 'Gls'

    Returns:
        list of tactical systems the player suits (e.g., ['high_press', 'ball_progression'])
    """
    suited = []

    # High press: needs pressing success + tackles/interceptions
    if (stats.get("Succ_Press", 0) > 0.70 or
        (stats.get("Tkl", 0) > 0.7 and stats.get("Int", 0) > 0.7)):
        suited.append("high_press")

    # Defensive transition: tackles + interceptions + recoveries
    if (stats.get("Tkl", 0) > 0.7 and
        stats.get("Int", 0) > 0.6):
        suited.append("defensive_transition")

    # Ball progression: progressive carries/passes
    if (stats.get("PrgC", 0) > 0.6 or
        stats.get("PrgP", 0) > 0.6):
        suited.append("ball_progression")

    # Creativity: assists + key passes
    if stats.get("Ast", 0) > 0.5:
        suited.append("creativity")

    # Goal threat: goals + xG
    if (stats.get("Gls", 0) > 0.5 or
        stats.get("xG", 0) > 0.6):
        suited.append("goal_threat")

    # Physical (proxy: high tackles + recovery)
    if (stats.get("Tkl", 0) > 0.7 or
        stats.get("Recoveries", 0) > 0.7):
        suited.append("physical")

    return suited if suited else ["versatile"]  # fallback if no strong signal
