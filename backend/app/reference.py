"""Reference + enrichment data that isn't present in the scouting vector store.

Two things the system needs but the indexed CSVs don't cleanly provide:

1. **Manager + playing style** — loaded from ``data/epl_seasons/managers.csv``,
   a multi-season history giving each club's manager per season, their final
   points/status, and a tactical "Evolution Pyramid Placement" (playing style).
   Lookups are season-aware and resolve to the manager who finished the season.
   This drives the "which manager would the player be under, and does the player
   fit that manager's system" reasoning. ``get_manager`` returns ``None`` for
   unknown clubs.

2. **Cost (wages)** — `player_wages.csv` is malformed as CSV (the figures contain
   unquoted commas, e.g. ``£ 525,000``), so we parse it with a regex instead of
   pandas. Wages are the only cost signal in the dataset; transfer fees are not
   available, so "cost" here means current wage outlay.
"""

import csv
import os
import re
from functools import lru_cache
from typing import Dict, Optional

WAGES_PATH = os.environ.get("WAGES_CSV_PATH", "data/epl_seasons/player_wages.csv")
MANAGERS_PATH = os.environ.get("MANAGERS_CSV_PATH", "data/epl_seasons/managers.csv")

# Map the dataset's various club spellings onto a single canonical key so that
# stats records, wages, and managers.csv all line up.
_CLUB_ALIASES: Dict[str, str] = {
    "manchester utd": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "newcastle utd": "newcastle united",
    "newcastle": "newcastle united",
    "nott'ham forest": "nottingham forest",
    "forest": "nottingham forest",
    "west ham united": "west ham",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    "brighton & hove albion": "brighton",
    "man united": "manchester united",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "ipswich": "ipswich town",
}

# Phrases used to detect a club mentioned in a free-text query. Longest first so
# "manchester united" wins over "manchester". normalize_club canonicalizes them.
_CLUB_SEARCH_TERMS = [
    "manchester united",
    "manchester utd",
    "man united",
    "man utd",
    "man city",
    "manchester city",
    "newcastle united",
    "newcastle utd",
    "newcastle",
    "nottingham forest",
    "nott'ham forest",
    "forest",
    "tottenham hotspur",
    "tottenham",
    "spurs",
    "west ham united",
    "west ham",
    "wolverhampton wanderers",
    "wolves",
    "brighton & hove albion",
    "brighton",
    "aston villa",
    "crystal palace",
    "leeds united",
    "leeds",
    "leicester city",
    "leicester",
    "ipswich town",
    "ipswich",
    "arsenal",
    "chelsea",
    "liverpool",
    "everton",
    "fulham",
    "brentford",
    "bournemouth",
    "burnley",
    "sunderland",
    "southampton",
]


def detect_club(text: Optional[str]) -> Optional[str]:
    """Return the canonical club key mentioned in free text, or None."""
    if not text:
        return None
    q = text.lower()
    for term in sorted(_CLUB_SEARCH_TERMS, key=len, reverse=True):
        if term in q:
            return normalize_club(term)
    return None


def normalize_club(squad: Optional[str]) -> str:
    """Canonicalize a club name: drop parentheticals like '(Incoming)', lowercase, alias."""
    if not squad:
        return ""
    key = re.sub(r"\(.*?\)", "", squad).strip().lower()
    return _CLUB_ALIASES.get(key, key)


_EMPTY_MANAGER: Dict[str, Optional[object]] = {
    "manager": None,
    "style": None,
    "status": None,
    "points": None,
    "season": None,
}


def _season_key(season: Optional[str]) -> str:
    """Normalize any season form to 'YYYY-YY'. '2024/2025' / '2024-25' -> '2024-25'."""
    if not season:
        return ""
    m = re.match(r"(\d{4})\D+(\d{2,4})", season.strip())
    return f"{m.group(1)}-{m.group(2)[-2:]}" if m else season.strip()


def _extract_points(status: str) -> Optional[int]:
    """'78 (2nd)' -> 78; 'Sacked (Jan 2026)' / 'Relegated' -> None."""
    m = re.match(r"\s*(\d+)", status or "")
    return int(m.group(1)) if m else None


def _is_settled(entry: Dict[str, Optional[object]]) -> bool:
    """True unless the row represents a mid-season exit (sacked/relegated)."""
    status = (entry.get("status") or "").lower()
    return not (status.startswith("sacked") or status.startswith("relegated"))


@lru_cache(maxsize=1)
def load_managers(
    path: str = MANAGERS_PATH,
) -> Dict[str, Dict[str, Dict[str, Optional[object]]]]:
    """Parse managers.csv into ``club -> {season_key -> profile}``.

    The file holds multiple rows per club (one per season, plus extra rows when a
    manager is sacked mid-season). For each club+season we keep the *settled*
    manager — the one who finished the season — preferring a non-sacked row and,
    on a tie, the later row (the successor).
    """
    table: Dict[str, Dict[str, Dict[str, Optional[object]]]] = {}
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            club = normalize_club(row.get("Club"))
            season = _season_key(row.get("Season"))
            if not club or not season:
                continue
            status = (row.get("Final Points/Status") or "").strip()
            entry: Dict[str, Optional[object]] = {
                "manager": (row.get("Manager") or "").strip() or None,
                "style": (row.get("Evolution Pyramid Placement") or "").strip() or None,
                "status": status or None,
                "points": _extract_points(status),
                "season": season,
            }
            per_season = table.setdefault(club, {})
            prev = per_season.get(season)
            # Prefer a settled row over an unsettled one; otherwise later wins.
            if prev is None or _is_settled(entry) or not _is_settled(prev):
                per_season[season] = entry
    return table


def get_manager_profile(squad: Optional[str], season: Optional[str] = None) -> Dict[str, Optional[object]]:
    """Return ``{manager, style, status, points, season}`` for a club.

    Looks up the requested season; falls back to the most recent season on record
    for that club (the returned ``season`` field shows which one was used).
    """
    per_season = load_managers().get(normalize_club(squad))
    if not per_season:
        return dict(_EMPTY_MANAGER)
    key = _season_key(season)
    if key and key in per_season:
        return per_season[key]
    latest = max(per_season)  # season keys sort chronologically as 'YYYY-YY'
    return per_season[latest]


def get_manager(squad: Optional[str], season: Optional[str] = None) -> Optional[str]:
    """Return the manager for a club in a given season, or None if unknown."""
    return get_manager_profile(squad, season)["manager"]


def _to_int_gbp(raw: str) -> Optional[int]:
    """'525,000' / '27 300 000' -> 525000 / 27300000."""
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def format_gbp(amount: Optional[int]) -> Optional[str]:
    """27300000 -> '£27,300,000'."""
    return f"£{amount:,}" if amount is not None else None


@lru_cache(maxsize=1)
def load_player_wages(path: str = WAGES_PATH) -> Dict[str, Dict[str, Optional[int]]]:
    """Load wages into ``name -> {weekly_gbp, annual_gbp}``.

    Prefers the cleaned schema (``Weekly Wages (GBP)`` / ``Annual Wages (GBP)``
    integer columns produced by ``backend.scripts.clean_csvs``). Falls back to the
    original malformed format (unquoted commas inside ``£`` currency strings),
    parsed line by line via regex, so it still works on an un-cleaned file.
    """
    wages: Dict[str, Dict[str, Optional[int]]] = {}
    if not os.path.exists(path):
        return wages

    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        clean = "Weekly Wages (GBP)" in header and "Annual Wages (GBP)" in header
        if clean:
            idx = {col: header.index(col) for col in header}
            p_i = idx["Player"]
            w_i = idx["Weekly Wages (GBP)"]
            a_i = idx["Annual Wages (GBP)"]
            for row in reader:
                if len(row) <= a_i or not row[p_i].strip():
                    continue
                wages[row[p_i].strip().lower()] = {
                    "weekly_gbp": _to_int_gbp(row[w_i]),
                    "annual_gbp": _to_int_gbp(row[a_i]),
                }
        else:
            # Legacy malformed file: pull the two £ amounts per raw line.
            fh.seek(0)
            next(fh, None)  # skip header
            for line in fh:
                parts = line.split(",")
                if len(parts) < 3 or not parts[1].strip():
                    continue
                name = parts[1].strip()  # fbref names contain no commas
                gbp = re.findall(r"£\s*([\d,\s]+?)\s*\(", line)
                wages[name.lower()] = {
                    "weekly_gbp": _to_int_gbp(gbp[0]) if len(gbp) >= 1 else None,
                    "annual_gbp": _to_int_gbp(gbp[1]) if len(gbp) >= 2 else None,
                }
    return wages


def get_estimated_cost(player_name: Optional[str]) -> Dict[str, Optional[str]]:
    """Return a human-readable wage cost for a player.

    Cost is approximated by current wages (transfer fees are not in the dataset).
    """
    record = load_player_wages().get((player_name or "").lower())
    if not record:
        return {
            "weekly_wages": None,
            "annual_wages": None,
            "basis": "No wage data available for this player.",
        }
    return {
        "weekly_wages": format_gbp(record.get("weekly_gbp")),
        "annual_wages": format_gbp(record.get("annual_gbp")),
        "basis": "Current wages (transfer fee not available in dataset).",
    }
