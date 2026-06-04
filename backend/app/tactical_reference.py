"""Tactical systems reference loaded from YAML (instead of book indexing).

This provides a lightweight tactical knowledge base with structured systems,
player profiles, and key metrics—used directly in queries without vector indexing.
"""

import os
import yaml
from functools import lru_cache
from typing import Dict, List, Optional

TACTICAL_REFERENCE_PATH = os.environ.get(
    "TACTICAL_REFERENCE_PATH",
    "data/historical/tactical_reference_extended.yaml"
)


@lru_cache(maxsize=1)
def load_tactical_reference() -> Dict:
    """Load tactical systems from YAML."""
    if not os.path.exists(TACTICAL_REFERENCE_PATH):
        return {"tactical_systems": {}}

    with open(TACTICAL_REFERENCE_PATH, 'r') as f:
        return yaml.safe_load(f) or {"tactical_systems": {}}


def get_tactical_systems() -> Dict[str, Dict]:
    """Return all tactical systems."""
    ref = load_tactical_reference()
    return ref.get("tactical_systems", {})


def get_tactical_system(system_name: str) -> Optional[Dict]:
    """Get a specific tactical system by name."""
    systems = get_tactical_systems()
    return systems.get(system_name)


def search_tactical_systems(query: str) -> List[Dict]:
    """Search tactical systems by concept keywords.

    Returns list of matching systems with their profiles.
    """
    query_lower = query.lower()
    systems = get_tactical_systems()
    matches = []

    for system_name, system_data in systems.items():
        keywords = system_data.get("concept_keywords", [])
        if any(kw in query_lower for kw in keywords):
            matches.append({
                "name": system_name,
                "display_name": system_data.get("display_name"),
                "description": system_data.get("description"),
                "player_profile": system_data.get("player_profile"),
                "key_metrics": system_data.get("required_metrics") or system_data.get("key_metrics"),
            })

    return matches


def get_system_info() -> Dict:
    """Return info about tactical reference for UI/system endpoint."""
    systems = get_tactical_systems()
    return {
        "source": "tactical_reference_extended.yaml",
        "type": "Structured tactical systems (no vector indexing)",
        "count": len(systems),
        "systems": list(systems.keys()),
        "description": "Comprehensive tactical systems with player profiles and key metrics",
    }
