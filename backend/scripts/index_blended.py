"""Reindex player data into single blended collection.

Combines season-specific stats + career metrics into one collection.
Each (player, season) tuple gets both per-season AND career-level metrics.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.database import (
    get_client,
    get_embedding,
    get_embeddings,
    BLENDED_COLLECTION,
    BLENDED_OUTPUT_FIELDS,
    init_collections,
)
from backend.app.enrichment.progression import calculate_progression

# Data paths
DATA_DIR = Path("data/epl_seasons")
CLEANED_FILES = {
    "2023/24": DATA_DIR / "player_stats_23-24.csv",
    "2024/25": DATA_DIR / "fbref_PL_2024-25.csv",  # Complete FBRef data with all stats
}
POSITION_REFERENCE = DATA_DIR / "player_overview.csv"
FBREF_FILE = DATA_DIR / "fbref_PL_2024-25.csv"  # For age data


def load_position_reference():
    """Load player→position mapping from 2023/24 (authoritative source)."""
    positions = {}
    if not POSITION_REFERENCE.exists():
        print(f"⚠️  Position reference not found: {POSITION_REFERENCE}")
        return positions

    with open(POSITION_REFERENCE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_name = row.get("player_name", "").strip()
            position = row.get("Pos", "").strip()
            if player_name and position:
                positions[player_name] = position
    return positions


def load_age_reference():
    """Load player→age mapping from fbref 2024/25 data."""
    ages = {}
    if not FBREF_FILE.exists():
        print(f"⚠️  FBRef file not found: {FBREF_FILE}")
        return ages

    with open(FBREF_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_name = row.get("Player", "").strip()
            age_str = row.get("Age", "").strip()
            if player_name and age_str:
                try:
                    age = int(age_str)
                    ages[player_name] = age
                except (ValueError, TypeError):
                    pass
    return ages


def load_season_stats(filepath, season):
    """Load cleaned CSV for a season (handles both custom and FBRef formats)."""
    stats_by_player = defaultdict(dict)

    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        return stats_by_player

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both "player_name" and "Player" column names
            player_name = (row.get("player_name") or row.get("Player") or "").strip()
            if not player_name:
                continue

            # Map column names (custom format vs FBRef format)
            col_map = {
                # Custom format columns
                "tackles": "tackles",
                "tackles_won": "tackles_won",
                "interceptions": "interceptions",
                "recoveries": "recoveries",
                "assists": "assists",
                "goals": "goals",
                "xg": "xg",
                "xag": "xag",
                "prgc": "prgc",
                "prgp": "prgp",
                "minutes": "minutes",
                "appearances": "appearances",
                "big_chances_created": "big_chances_created",
                # FBRef format columns
                "Tkl": "tackles",
                "TklW": "tackles_won",
                "Int": "interceptions",
                "Recoveries": "recoveries",
                "Ast": "assists",
                "Gls": "goals",
                "xG": "xg",
                "xAG": "xag",
                "PrgC": "prgc",
                "PrgP": "prgp",
                "Min": "minutes",
                "MP": "appearances",  # Matches played
            }

            # Numeric field types
            stat_dtypes = {
                "tackles": float,
                "tackles_won": float,
                "interceptions": float,
                "recoveries": float,
                "assists": float,
                "goals": float,
                "xg": float,
                "xag": float,
                "prgc": float,
                "prgp": float,
                "minutes": int,
                "appearances": int,
                "big_chances_created": float,
            }

            stats = {"season": season, "player_name": player_name}

            # Try to extract each stat (checking both custom and FBRef column names)
            for custom_col, stat_name in {
                "tackles": "tackles", "Tkl": "tackles",
                "tackles_won": "tackles_won", "TklW": "tackles_won",
                "interceptions": "interceptions", "Int": "interceptions",
                "recoveries": "recoveries", "Recoveries": "recoveries",
                "assists": "assists", "Ast": "assists",
                "goals": "goals", "Gls": "goals",
                "xg": "xg", "xG": "xg",
                "xag": "xag", "xAG": "xag",
                "prgc": "prgc", "PrgC": "prgc",
                "prgp": "prgp", "PrgP": "prgp",
                "minutes": "minutes", "Min": "minutes",
                "appearances": "appearances", "MP": "appearances",
                "big_chances_created": "big_chances_created",
            }.items():
                if stat_name not in stats:  # Don't override if already found
                    val = row.get(custom_col)
                    if val and str(val).strip():
                        try:
                            stats[stat_name] = stat_dtypes.get(stat_name, float)(val)
                        except (ValueError, TypeError):
                            pass

            # Get squad (try both column names)
            stats["squad"] = (row.get("Squad") or row.get("squad") or "").strip()
            stats_by_player[player_name] = stats

    return stats_by_player


def calculate_career_metrics(player_seasons_dict):
    """Calculate career aggregates for each player.

    player_seasons_dict: {player_name: [{season: ..., tackles: ...}, ...]}
    returns: {player_name: {avg_tackles: ..., improvement_score: ...}}
    """
    careers = {}

    for player_name, seasons_list in player_seasons_dict.items():
        if not seasons_list:
            continue

        # Calculate averages
        metrics = [
            "tackles",
            "tackles_won",
            "interceptions",
            "recoveries",
            "assists",
            "goals",
            "xg",
            "xag",
            "prgc",
            "prgp",
        ]

        career = {"player_name": player_name}
        for metric in metrics:
            values = [s.get(metric, 0) for s in seasons_list if metric in s]
            if values:
                career[f"avg_{metric}"] = sum(values) / len(values)

        # Use progression module to get improvement/stability
        history_format = [
            {
                "player_name": player_name,
                "season": s["season"],
                "squad": s.get("squad"),
                "position": None,
                "tackles": s.get("tackles", 0),
                "assists": s.get("assists", 0),
                "goals": s.get("goals", 0),
                "minutes": s.get("minutes", 0),
            }
            for s in seasons_list
        ]

        progression = calculate_progression(history_format)
        career.update(
            {
                "improvement_score": progression.get("improvement_score", 0),
                "stability_score": progression.get("stability_score", 0),
                "consistency_pct": progression.get("consistency_pct", 0),
                "best_season": progression.get("best_season"),
                "trend": progression.get("trend"),
            }
        )

        careers[player_name] = career

    return careers


def build_blended_records(stats_by_season, positions, careers, ages=None):
    """Combine season stats + career metrics into blended records.

    Returns list of records ready to index.
    """
    if ages is None:
        ages = {}
    records = []

    for season, season_stats in stats_by_season.items():
        for player_name, season_data in season_stats.items():
            # Get authoritative position from 2023/24
            position = positions.get(player_name, "Unknown")

            # Get career metrics
            career_data = careers.get(player_name, {})

            # Get age (use fbref data)
            age = ages.get(player_name, None)

            # Combine
            record = {
                "player_name": player_name,
                "season": season,
                "position": position,
                "squad": season_data.get("squad", "Unknown"),
                "age": age,  # From fbref 2024/25 data
                # Season-specific
                **{k: v for k, v in season_data.items() if k not in ["player_name", "season", "squad"]},
                # Career metrics (denormalized)
                **{
                    k: v
                    for k, v in career_data.items()
                    if k.startswith("avg_") or k in ["improvement_score", "stability_score", "consistency_pct", "best_season", "trend"]
                },
            }

            # Build embedding text
            record["text"] = (
                f"{player_name}, {position}, {record.get('squad', 'Unknown')} ({season}). "
                f"Tackles: {record.get('tackles', 0)}, "
                f"Assists: {record.get('assists', 0)}, "
                f"Goals: {record.get('goals', 0)}, "
                f"Minutes: {record.get('minutes', 0)}"
            )

            records.append(record)

    return records


def index_blended_records(records, batch_size=100):
    """Index all records into Milvus blended collection."""
    client = get_client()

    print(f"\n📊 Indexing {len(records)} records into {BLENDED_COLLECTION}...")

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        # Get embeddings for text
        texts = [r["text"] for r in batch]
        embeddings = get_embeddings(texts)

        # Add vectors to records
        for record, vector in zip(batch, embeddings):
            record["vector"] = vector

        # Insert batch
        try:
            client.insert(
                collection_name=BLENDED_COLLECTION,
                data=batch,
            )
        except Exception as e:
            print(f"  ❌ Error inserting batch {i//batch_size}: {e}")
            return False

        print(f"  ✅ Indexed {min(i+batch_size, len(records))}/{len(records)}")

    print(f"✅ Complete! {len(records)} records indexed")
    return True


def run():
    """Full reindexing pipeline."""
    print("\n" + "=" * 80)
    print("BLENDED COLLECTION REINDEXING")
    print("=" * 80)

    # Initialize collections
    print("\n0️⃣  Initializing Milvus collections...")
    try:
        init_collections()
        print("  ✅ Collections initialized")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    # Load position reference
    print("\n1️⃣  Loading position reference...")
    positions = load_position_reference()
    print(f"  ✅ Loaded {len(positions)} player positions from 2023/24")

    # Load age reference
    print("\n1b️⃣  Loading age reference...")
    ages = load_age_reference()
    print(f"  ✅ Loaded {len(ages)} player ages from fbref 2024/25")

    # Load season stats
    print("\n2️⃣  Loading season-specific stats...")
    stats_by_season = {}
    for season, filepath in CLEANED_FILES.items():
        stats = load_season_stats(filepath, season)
        stats_by_season[season] = stats
        print(f"  ✅ {season}: {len(stats)} players")

    # Organize by player for career calculation
    print("\n3️⃣  Organizing by player for career metrics...")
    player_seasons = defaultdict(list)
    for season, season_stats in stats_by_season.items():
        for player_name, stats in season_stats.items():
            player_seasons[player_name].append({**stats, "season": season})
    print(f"  ✅ {len(player_seasons)} unique players")

    # Calculate career metrics
    print("\n4️⃣  Calculating career metrics...")
    careers = calculate_career_metrics(player_seasons)
    print(f"  ✅ Calculated improvement/stability/consistency for {len(careers)} players")

    # Build blended records
    print("\n5️⃣  Building blended records...")
    records = build_blended_records(stats_by_season, positions, careers, ages)
    print(f"  ✅ Created {len(records)} blended records ({len(player_seasons)} players × seasons)")

    # Index
    print("\n6️⃣  Indexing into Milvus...")
    success = index_blended_records(records)
    if not success:
        print("  ❌ Indexing failed")
        return False

    # Verify
    print("\n7️⃣  Verifying...")
    from backend.app.database import count_entities

    count = count_entities(BLENDED_COLLECTION)
    print(f"  ✅ {count} records in {BLENDED_COLLECTION}")

    if count != len(records):
        print(f"  ⚠️  Expected {len(records)} records, got {count}")

    print("\n" + "=" * 80)
    print("✅ REINDEXING COMPLETE")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
