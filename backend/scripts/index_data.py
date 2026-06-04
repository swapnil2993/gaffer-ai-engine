import os
import re
import sys
from typing import List

import pandas as pd
import yaml

# Ensure backend is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.scripts.data_cleaning.career_aggregation import (
    aggregate_player_seasons,
    describe_career_aggregate,
)
from backend.app.database import (
    fetch_player_history,
    get_embedding,
    insert_player_career,
    insert_player_stat,
    insert_theory_chunk,
)
from backend.app.parser import chunk_tactical_book
from backend.app.query_understanding import classify_tactical_concept
from backend.app.schemas import TacticalTheoryPayload
from backend.scripts.data_cleaning.common import num, pct
from backend.scripts.data_cleaning.data_transformer import (
    describe_outfield_player,
    describe_2023_24_player,
)

# Per-season cap. Default 0 = index every player; set MAX_PLAYERS_PER_SEASON to a
# positive number for a quick partial demo index.
MAX_PLAYERS_PER_SEASON = int(os.environ.get("MAX_PLAYERS_PER_SEASON", "0"))

# player_overview.csv uses full position words; map them to the FW/MF/DF/GK
# codes used by the fbref season files so the position filter is consistent.
POSITION_CODE = {
    "goalkeeper": "GK",
    "defender": "DF",
    "midfielder": "MF",
    "forward": "FW",
}




def describe_outfield(name, squad, position, season, stats):
    """Wrapper for describe_outfield_player from data_transformer module."""
    return describe_outfield_player(name, squad, position, season, stats)


def describe_2023_24(name, squad, position, season, stats):
    """Wrapper for describe_2023_24_player from data_transformer module."""
    return describe_2023_24_player(name, squad, position, season, stats)


# Front/back-matter and TOC/index markers — chunks dominated by these aren't
# tactical content; indexing them only dilutes retrieval relevancy.
_NOISE_MARKERS = (
    "all rights reserved",
    "table of contents",
    "also by",
    "isbn",
    "first published",
    "typeset",
    "printed and bound",
    "penguin books",
    "copyright",
    "acknowledgment",
    "bibliography",
)


def _is_noise_chunk(chunk: str) -> bool:
    """True for non-content chunks (front/back matter, table of contents, index)."""
    text = chunk.strip()
    if len(text) < 200:  # too short to be useful prose
        return True
    low = text.lower()
    if any(m in low for m in _NOISE_MARKERS) and len(text) < 700:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and sum(1 for ln in lines if len(ln.strip()) < 45) / len(lines) > 0.6:
        return True  # TOC/index-like: many short lines
    if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.15:
        return True  # page numbers / index / dense figures, not prose
    return False


def chunk_text(text: str, target_size: int = 600) -> List[str]:
    """Paragraph-aware chunking that drops non-content (front/back matter, TOC).

    Smaller, topically-coherent chunks score far better on retrieval relevancy
    than fixed 1000-char windows that straddle unrelated sections. We pack whole
    paragraphs up to a target size, then filter out obvious noise.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 5:  # markdown used single newlines — fall back
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= target_size:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if not _is_noise_chunk(c)]


def index_tactical_reference():
    """Indexes the comprehensive tactical reference guide.

    Uses tactical_reference_extended.yaml which includes:
    - Modern tactical systems with metrics
    - Historical evolution chains
    - Manager evolution profiles
    - Case studies with real examples
    """
    ref_path = "data/historical/tactical_reference_extended.yaml"

    if not os.path.exists(ref_path):
        print(f"Tactical reference not found: {ref_path}")
        return

    print(f"Indexing {ref_path}...")
    with open(ref_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    systems = data.get("tactical_systems", {})
    for system_name, system_data in systems.items():
        # Create a chunk for each tactical system with explicit metrics for grounding
        metrics_section = ""
        if "required_metrics" in system_data:
            metrics_section = "\n## Required Metrics\n"
            for metric, details in system_data["required_metrics"].items():
                threshold = details.get("threshold", "N/A")
                meaning = details.get("meaning", "")
                metrics_section += f"- {metric}: > {threshold} ({meaning})\n"

        managers_section = ""
        if "key_managers" in system_data:
            managers_section = f"\n## Key Managers\n{', '.join(system_data['key_managers'])}\n"

        era_section = ""
        if "era" in system_data:
            era_section = f"\n**Era:** {system_data['era']}"

        text = f"""# {system_data['display_name']}{era_section}

{system_data['description']}{managers_section}

## Player Requirements
{system_data['player_profile']}{metrics_section}
"""
        payload = TacticalTheoryPayload(
            title="Tactical Reference Guide",
            chapter=system_data["display_name"],
            content=text,
        )

        metadata = {
            "title": payload.title,
            "chapter": system_data["display_name"],
            "heading": system_name,
            "concept": system_name,
            "era": system_data.get("era", ""),
            "source": "tactical-reference",
        }

        insert_theory_chunk(payload.content, metadata)

    print(f"Indexed {len(systems)} tactical systems from comprehensive reference guide.")

    # Index Football Hackers context pack as addon to tactical reference
    context_pack_path = "data/historical/Football_Hackers.md"
    if os.path.exists(context_pack_path):
        print(f"Indexing {context_pack_path} as tactical reference addon...")
        with open(context_pack_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into sections for better chunking
        sections = content.split("\n## ")
        for section_text in sections:
            if not section_text.strip():
                continue

            section_title = section_text.split("\n")[0] if "\n" in section_text else "Football Hackers"
            full_section = f"## {section_text}" if section_text and section_text[0] != "#" else section_text

            payload = TacticalTheoryPayload(
                title="Football Hackers",
                chapter=section_title,
                content=full_section,
            )

            metadata = {
                "title": "Football Hackers",
                "chapter": section_title,
                "heading": section_title,
                "concept": section_title,
                "source": "football-hackers-context",
            }

            insert_theory_chunk(payload.content, metadata)

        print(f"Indexed Football Hackers context pack into tactical reference collection.")


def index_tactical_book(limit: int = None):
    """Indexes tactical books (The Inverted Pyramid + Football Hackers) without LLM enrichment.

    Fast indexing. Tactical concept mapping happens at query time via a separate
    lightweight query→FBref filter mapper.
    """
    books = [
        # ("data/historical/the-inverted-pyramid.epub", "the-inverted-pyramid"),
        ("data/historical/Football_hackers.epub", "football-hackers"),
    ]

    total_kept = 0
    for book_path, source_name in books:
        if not os.path.exists(book_path):
            print(f"Book not found: {book_path}")
            continue

        print(f"Chunking {source_name}...")
        chunks = chunk_tactical_book(book_path)
        kept = 0
        for i, ch in enumerate(chunks):
            if limit and total_kept >= limit:
                break
            text = ch["text"]
            if _is_noise_chunk(text):
                continue

            heading = ch.get("heading", "Unknown")
            payload = TacticalTheoryPayload(content=text, chapter=heading)

            metadata = {
                "title": payload.title,
                "chunk_id": i,
                "chapter": payload.chapter or heading,
                "heading": heading,
                "concept": classify_tactical_concept(text),
                "source": source_name,
            }

            insert_theory_chunk(payload.content, metadata)
            kept += 1
            total_kept += 1

            if kept % 20 == 0:
                print(f"  {source_name}: {kept} chunks...")

        print(f"  {source_name}: {kept} chunks indexed.")

    print(f"Indexed {total_kept} total content chunks into theory collection.")


def index_player_stats():
    """Indexes player stats from CSVs."""
    csv_paths = [
        ("data/epl_seasons/fbref_PL_2024-25.csv", "2024/2025"),
        ("data/epl_seasons/raw/merged_25_26.csv", "2025/2026"),  # Use raw merged file to avoid file corruption
    ]

    for path, season in csv_paths:
        if not os.path.exists(path):
            print(f"Skipping {path}, file not found.")
            continue

        print(f"Indexing stats from {path} (Season: {season})...")
        df = pd.read_html(path)[0] if path.endswith(".html") else pd.read_csv(path)

        # Basic cleaning
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        for i, (idx, row) in enumerate(df.iterrows()):
            if MAX_PLAYERS_PER_SEASON and i >= MAX_PLAYERS_PER_SEASON:
                break
            player_name = str(row.get("Player", "Unknown"))
            position = str(row.get("Pos", "Unknown"))
            squad = str(row.get("Squad", "Unknown"))

            # Numeric stats — stored as scalar fields so they can be FILTERED on
            # (hybrid search), not just embedded.
            stats = {
                "goals": num(row.get("Gls")),
                "assists": num(row.get("Ast")),
                "xg": num(row.get("xG")),
                "xag": num(row.get("xAG")),
                "prgc": num(row.get("PrgC")),
                "prgp": num(row.get("PrgP")),
                "minutes": num(row.get("Min")),
                "appearances": num(row.get("MP")),
                # Estimated defensive actions (see mock_defensive_stats.py).
                "tackles": num(row.get("Tkl")),
                "tackles_won": num(row.get("TklW")),
                "interceptions": num(row.get("Int")),
                "recoveries": num(row.get("Recoveries")),
            }
            # Prose description for the embedding (semantic match) — see helper.
            stats_text = describe_outfield(player_name, squad, position, season, stats)

            metadata = {
                "player_name": player_name,
                "position": position,
                "squad": squad,
                "season": season,
                "type": "stats",
                **stats,
            }

            vector = get_embedding(stats_text)
            insert_player_stat(stats_text, vector, metadata)

    print("Player stats indexing complete.")


def _load_overview_lookup(path: str = "data/epl_seasons/player_overview.csv") -> dict:
    """Build a Name -> {club, position_code} lookup from player_overview.csv.

    The 2023/24 stats file has no club or position, so we join on player name to
    recover them (position words are normalized to FW/MF/DF/GK codes).
    """
    lookup = {}
    if not os.path.exists(path):
        return lookup
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name:
            continue
        club = row.get("Club")
        position = str(row.get("Position", "")).strip().lower()
        lookup[name] = {
            "club": str(club).strip() if pd.notna(club) else "Unknown",
            "position": POSITION_CODE.get(position, "Unknown"),
        }
    return lookup


def index_player_stats_2023_24():
    """Indexes the 2023/24 player-profile stats, joined with club/position."""
    path = "data/epl_seasons/player_stats_23-24.csv"
    season = "2023/2024"
    if not os.path.exists(path):
        print(f"Skipping {path}, file not found.")
        return

    overview = _load_overview_lookup()
    print(f"Indexing stats from {path} (Season: {season})...")
    df = pd.read_csv(path)

    for i, (idx, row) in enumerate(df.iterrows()):
        if MAX_PLAYERS_PER_SEASON and i >= MAX_PLAYERS_PER_SEASON:
            break
        player_name = str(row.get("Name", "Unknown"))
        info = overview.get(player_name, {"club": "Unknown", "position": "Unknown"})
        squad = info["club"]
        position = info["position"]

        # This dataset has no xG/progressive metrics, but is rich in counting and
        # defensive stats. minutes=0 (not in this dataset) so a min_minutes filter
        # naturally excludes 2023/24 unless relaxed.
        stats = {
            "goals": num(row.get("Goals")),
            "assists": num(row.get("Assists")),
            "appearances": num(row.get("Appearances")),
            "minutes": 0.0,
            "big_chances_created": num(row.get("Big Chances Created")),
            "tackles": num(row.get("Tackles")),
            "tackles_won": round(num(row.get("Tackles")) * pct(row.get("Tackle success %"))),
            "interceptions": num(row.get("Interceptions")),
            "recoveries": num(row.get("Recoveries")),  # real for 2023/24
            "clean_sheets": num(row.get("Clean sheets")),
            "saves": num(row.get("Saves")),
        }
        stats_text = describe_2023_24(player_name, squad, position, season, stats)

        metadata = {
            "player_name": player_name,
            "position": position,
            "squad": squad,
            "season": season,
            "type": "stats",
            **stats,
        }

        vector = get_embedding(stats_text)
        insert_player_stat(stats_text, vector, metadata)

    print("2023/24 player stats indexing complete.")


def index_player_career_aggregates():
    """Generate and index multi-season career profiles for all players.

    After all seasons are indexed, fetch each player's history, aggregate their
    stats across seasons, and create a career profile document. These are indexed
    in a separate collection for queries about "improving talent" or "career arc".
    """
    from backend.app.database import STATS_COLLECTION, get_client

    client = get_client()

    # Get all unique player names from the stats collection
    results = client.query(
        collection_name=STATS_COLLECTION,
        filter="player_name != ''",
        output_fields=["player_name"],
        limit=10000,
    )
    unique_names = set()
    for r in results:
        name = r.get("player_name")
        if name:
            unique_names.add(name)

    print(f"Generating career aggregates for {len(unique_names)} unique players...")

    aggregated = 0
    for player_name in sorted(unique_names):
        # Fetch all seasons for this player
        history = fetch_player_history(player_name)
        if len(history) < 2:
            # Skip players with less than 2 seasons of data
            continue

        # Aggregate their stats
        agg = aggregate_player_seasons(player_name, history)
        if not agg:
            continue

        # Generate prose description
        prose = describe_career_aggregate(agg, history)
        if not prose:
            continue

        # Embed and insert
        vector = get_embedding(prose)
        metadata = {
            "player_name": player_name,
            "position": agg["position"],
            "best_squad": agg["best_squad"],
            "avg_goals": agg["aggregates"].get("goals", {}).get("avg", 0.0),
            "avg_assists": agg["aggregates"].get("assists", {}).get("avg", 0.0),
            "avg_prgp": agg["aggregates"].get("prgp", {}).get("avg", 0.0),
            "avg_tackles": agg["aggregates"].get("tackles", {}).get("avg", 0.0),
            "avg_tackles_won": agg["aggregates"].get("tackles_won", {}).get("avg", 0.0),
            "avg_interceptions": agg["aggregates"].get("interceptions", {}).get("avg", 0.0),
            "best_season": agg["best_season"],
            "trend": agg["trend"],
            "momentum": agg["momentum"],
            "improvement_score": agg.get("improvement_score", 0.0),
            "stability_score": agg.get("stability_score", 0.0),
            "consistency_pct": agg.get("consistency_pct", 0.0),
            "type": "career",
        }

        insert_player_career(prose, vector, metadata)
        aggregated += 1

        if aggregated % 50 == 0:
            print(f"  {aggregated} career profiles generated...")

    print(f"Career aggregates indexing complete: {aggregated} profiles.")


if __name__ == "__main__":
    from backend.app.database import init_collections
    from backend.scripts.data_cleaning import run_cleaning_pipeline

    print("Step 1: Running data cleaning pipeline...")
    run_cleaning_pipeline()
    print()

    print("Step 2: Initializing collections...")
    init_collections()
    print()

    print("Step 3: Indexing tactical reference...")
    index_tactical_reference()
    print()

    # print("Step 4: Indexing tactical books...")
    # index_tactical_book()
    # print()

    print("Step 5: Indexing player stats...")
    index_player_stats()
    index_player_stats_2023_24()
    print()

    print("Step 6: Generating career aggregate profiles...")
    index_player_career_aggregates()
    print()

    print("✓ All data cleaned and indexed successfully.")
