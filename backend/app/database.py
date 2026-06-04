import os
import time as _time

# Milvus Lite spawns its embedded server via fork(). torch (pulled in by
# sentence_transformers) spawns threads, and forking after threads exist crashes
# natively on macOS. Two defenses:
#   1) make gRPC fork-aware (set BEFORE pymilvus/grpc is imported), and
#   2) import sentence_transformers lazily (see get_model), so torch is not
#      loaded until AFTER Milvus has forked its server.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Multiple native libs (grpc, OpenCV, spaCy/blis, torch) can each bundle an
# OpenMP runtime; loading two aborts/segfaults the process. Allow duplicates.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pymilvus import MilvusClient

# Internal cache for client and model
_client = None
_model = None

THEORY_COLLECTION = "tactical_theory_collection"
STATS_COLLECTION = "player_stats_collection"
CAREER_COLLECTION = "player_career_collection"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

DB_PATH = "scoutintel_local.db"


def get_client(retries: int = 8, delay: float = 0.75):
    """Returns the singleton Milvus Lite client.

    Creating the client forks Milvus Lite's embedded server. We do this as early
    as possible (see the eager call at import, below) so the fork happens while
    only pymilvus is loaded — before dspy/litellm/torch spawn threads, which would
    otherwise make fork() crash natively (SIGSEGV) on macOS.

    Milvus Lite allows only ONE process to open the DB. If another instance holds
    the lock we retry briefly (covers a uvicorn --reload handoff) and then fail
    with a clear, actionable message instead of a raw traceback.
    """
    global _client
    if _client is not None:
        return _client
    last = None
    for attempt in range(max(1, retries)):
        try:
            _client = MilvusClient(DB_PATH)
            return _client
        except Exception as e:
            last = e
            if attempt < retries - 1:
                _time.sleep(delay)
    raise RuntimeError(
        f"Could not open {DB_PATH}: another process is holding the Milvus lock. "
        "Stop the other instance and retry — e.g.  pkill -f 'uvicorn backend.app.main'"
    ) from last


# NOTE: we deliberately do NOT open the client at import time. Under
# `uvicorn --reload` the reloader process also imports this module, and an
# import-time open would make the reloader hold the single-process Milvus lock,
# starving the actual worker. Instead the server's lifespan calls get_client()
# (via init_collections) FIRST — before init_dspy — so the fork happens in the
# one worker process and before dspy/torch spawn threads (which would make
# fork() crash on macOS). See backend/app/main.py:lifespan.


def get_model():
    """Returns the singleton SentenceTransformer model (lazily imported)."""
    global _model
    if _model is None:
        # Imported here (not at module top) so torch loads only on first use,
        # after Milvus Lite has already forked its embedded server.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def init_collections():
    """Initializes and loads the Milvus collections."""
    client = get_client()

    # Tactical Theory Collection
    if not client.has_collection(THEORY_COLLECTION):
        client.create_collection(
            collection_name=THEORY_COLLECTION,
            dimension=384,
            metric_type="COSINE",
            enable_dynamic_field=True,
            auto_id=True,
        )

    # Player Stats Collection
    if not client.has_collection(STATS_COLLECTION):
        client.create_collection(
            collection_name=STATS_COLLECTION,
            dimension=384,
            metric_type="COSINE",
            enable_dynamic_field=True,
            auto_id=True,
        )

    # Player Career Aggregate Collection (3-year profiles)
    if not client.has_collection(CAREER_COLLECTION):
        client.create_collection(
            collection_name=CAREER_COLLECTION,
            dimension=384,
            metric_type="COSINE",
            enable_dynamic_field=True,
            auto_id=True,
        )

    # Explicitly load collections for searching
    client.load_collection(THEORY_COLLECTION)
    client.load_collection(STATS_COLLECTION)
    client.load_collection(CAREER_COLLECTION)


def count_entities(collection_name: str) -> int:
    """Return the row count for a collection (0 if it doesn't exist)."""
    client = get_client()
    if not client.has_collection(collection_name):
        return 0
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count", 0))


def get_embedding(text: str) -> list:
    """Generates a 384-dimensional embedding for the given text."""
    model = get_model()
    return model.encode(text).tolist()


def get_embeddings(texts: list) -> list:
    """Batch-embed a list of texts (used for sentence-level relevance scoring)."""
    if not texts:
        return []
    model = get_model()
    return model.encode(list(texts)).tolist()


def insert_theory_chunk(text: str, metadata: dict):
    """Inserts a tactical theory chunk into Milvus."""
    client = get_client()
    vector = get_embedding(text)
    data = [{"vector": vector, "text": text, **metadata}]
    client.insert(collection_name=THEORY_COLLECTION, data=data)


def insert_player_stat(text: str, vector: list, metadata: dict):
    """Inserts a player stat record into Milvus."""
    client = get_client()
    data = [{"vector": vector, "text": text, **metadata}]
    client.insert(collection_name=STATS_COLLECTION, data=data)


def insert_player_career(text: str, vector: list, metadata: dict):
    """Inserts a player career aggregate record into Milvus."""
    client = get_client()
    data = [{"vector": vector, "text": text, **metadata}]
    client.insert(collection_name=CAREER_COLLECTION, data=data)


def query_theory(query_text: str, limit: int = 5, concept: str = None, concepts: list = None) -> list:
    """Queries the tactical theory collection, optionally filtering by concept(s).

    When `concept` (single string) or `concepts` (list) is given, the search is
    restricted to chunks matching those tactical concepts. If no matches, falls
    back to unfiltered search.
    """
    client = get_client()
    vector = get_embedding(query_text)
    output_fields = [
        "text",
        "title",
        "chapter",
        "heading",
        "concept",
        "source",
    ]

    def _search(filter_expr):
        results = client.search(
            collection_name=THEORY_COLLECTION,
            data=[vector],
            filter=filter_expr,
            limit=limit,
            output_fields=output_fields,
        )
        return results[0] if results else []

    # Build filter: single concept or list of concepts
    target_concepts = concepts or ([concept] if concept else [])
    if target_concepts:
        # Filter by ANY matching concept (OR logic)
        filter_expr = " || ".join(f"concept == '{_escape_literal(c)}'" for c in target_concepts)
        hits = _search(filter_expr)
        if hits:
            return hits
    return _search("")


def _escape_literal(value: str) -> str:
    """Escapes a string for safe use inside a Milvus filter expression literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


STAT_OUTPUT_FIELDS = [
    "text",
    "player_name",
    "position",
    "squad",
    "season",
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

CAREER_OUTPUT_FIELDS = [
    "text",
    "player_name",
    "position",
    "best_squad",
    "avg_goals",
    "avg_assists",
    "avg_prgp",
    "avg_tackles",
    "best_season",
    "trend",
    "momentum",
]


def query_stats(
    query_vector: list,
    season: str = None,
    position: str = None,
    player_name: str = None,
    min_minutes: float = None,
    stat_filters: list = None,
    limit: int = 5,
) -> list:
    """Queries the player stats collection — hybrid: vector + scalar filters.

    Semantic ranking comes from the vector; scalar clauses narrow the pool:
      - season: exact match
      - position / player_name: substring (`like`), so 'MF' matches 'MF,FW'
      - min_minutes: numeric floor (filters out tiny-sample noise)
      - stat_filters: extra Milvus expressions, e.g. ["goals >= 10"]
    All are optional.
    """
    client = get_client()
    clauses = []
    if season:
        clauses.append(f"season == '{_escape_literal(season)}'")
    if position:
        clauses.append(f"position like '%{_escape_literal(position)}%'")
    if player_name:
        clauses.append(f"player_name like '%{_escape_literal(player_name)}%'")
    if min_minutes:
        clauses.append(f"minutes >= {float(min_minutes)}")
    if stat_filters:
        clauses.extend(stat_filters)
    filter_expr = " and ".join(clauses)

    results = client.search(
        collection_name=STATS_COLLECTION,
        data=[query_vector],
        filter=filter_expr,
        limit=limit,
        output_fields=STAT_OUTPUT_FIELDS,
    )
    return results[0] if results else []


def query_career(
    query_vector: list,
    position: str = None,
    player_name: str = None,
    stat_filters: list = None,
    limit: int = 5,
) -> list:
    """Queries the player career aggregate collection — hybrid: vector + scalar filters.

    Similar to query_stats but without season filtering (career aggregates span all years).
    Use for queries asking about "improving talent", "consistent performers", "career arc".
    """
    client = get_client()
    clauses = []
    if position:
        clauses.append(f"position like '%{_escape_literal(position)}%'")
    if player_name:
        clauses.append(f"player_name like '%{_escape_literal(player_name)}%'")
    if stat_filters:
        clauses.extend(stat_filters)
    filter_expr = " and ".join(clauses)

    results = client.search(
        collection_name=CAREER_COLLECTION,
        data=[query_vector],
        filter=filter_expr,
        limit=limit,
        output_fields=CAREER_OUTPUT_FIELDS,
    )
    return results[0] if results else []


def fetch_player_history(player_name: str) -> list:
    """Fetch a player's stats across all seasons (for progression tracking).

    Returns list of rows sorted by season (ascending).
    """
    client = get_client()
    results = client.query(
        collection_name=STATS_COLLECTION,
        filter=f"player_name == '{_escape_literal(player_name)}'",
        output_fields=STAT_OUTPUT_FIELDS,
    )
    # Sort by season ascending so 2023/24 comes before 2024/25
    return sorted(results, key=lambda r: r.get("season", ""))


def fetch_player_vectors(
    season: str = None,
    position: str = None,
    player_names: list = None,
    limit: int = 140,
) -> list:
    """Fetch player rows *including their embedding vectors* for visualization.

    Used to build the cosine-similarity scatter plot. Returns dicts with
    player_name, position, squad, season and the raw `vector`.
    """
    client = get_client()
    clauses = []
    if season:
        clauses.append(f"season == '{_escape_literal(season)}'")
    if position:
        clauses.append(f"position like '%{_escape_literal(position)}%'")
    if player_names:
        joined = ", ".join('"' + n.replace("\\", "\\\\").replace('"', '\\"') + '"' for n in player_names)
        clauses.append(f"player_name in [{joined}]")
    # Milvus query() needs a non-empty expression; use an always-true one.
    filter_expr = " and ".join(clauses) if clauses else "player_name != ''"

    raw = client.query(
        collection_name=STATS_COLLECTION,
        filter=filter_expr,
        limit=limit,
        output_fields=["player_name", "position", "squad", "season", "vector"],
    )
    # Materialize into plain dicts. pymilvus returns a special result object whose
    # internal bitmap breaks if it's concatenated (e.g. with `+=`); converting to
    # plain dicts here makes the result safe to combine and iterate.
    fields = ("player_name", "position", "squad", "season", "vector")
    return [{f: r.get(f) for f in fields} for r in raw]
