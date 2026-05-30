from pymilvus import MilvusClient, DataType

# Initialize Milvus Lite client
# This points to a local .db file for persistence
client = MilvusClient("scoutintel_local.db")

COLLECTION_NAME = "epl_scouting_collection"

# Create collection if it doesn't exist
# dimension=384 is common for many lightweight embedding models (e.g., all-MiniLM-L6-v2)
if not client.has_collection(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=384,
        metric_type="COSINE",
        enable_dynamic_field=True
    )

def insert_scouting_vector(vector: list, text: str, metadata: dict):
    """
    Inserts a scouting vector and its associated metadata into Milvus.
    """
    data = [
        {
            "vector": vector,
            "text": text,
            **metadata
        }
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)

def query_scouting_vectors(vector: list, season: str, position: str) -> list:
    """
    Queries the vector database using a vector and strict scalar filters.
    """
    # Build strict SQL-like string filter expression
    filter_expr = f"season == '{season}' and position == '{position}'"
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        filter=filter_expr,
        limit=5,
        output_fields=["text", "player_name", "season", "position"]
    )
    return results
