#!/usr/bin/env python
"""Reindex tactical theory collections (reference + books).

Usage:
    python -m backend.scripts.reindex_theory

This clears and re-indexes the theory collection with:
1. Tactical reference (modern systems)
2. The Inverted Pyramid (historical context)
3. Football Hackers (modern metrics)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.database import (
    THEORY_COLLECTION,
    get_client,
    init_collections,
)
from backend.scripts.index_data import (
    index_tactical_reference,
    index_tactical_book,
)


def main():
    print("Initializing collections...")
    init_collections()

    print(f"Clearing {THEORY_COLLECTION}...")
    client = get_client()
    client.delete(collection_name=THEORY_COLLECTION, filter="chunk_id >= 0")

    print("\nIndexing tactical theory...")
    print("-" * 50)
    index_tactical_reference()
    print()
    index_tactical_book()
    print("-" * 50)
    print("\n✓ Theory collection re-indexed successfully.")


if __name__ == "__main__":
    main()
