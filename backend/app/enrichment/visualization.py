"""Visualization enrichment - generate plots and embeddings visualizations."""

import numpy as np


def build_similarity_plot(query_vector, rows, candidate_names):
    """Project query + player embeddings to 2D (PCA) and score cosine similarity.

    Returns points the UI can scatter-plot: the gold query point, the retrieved
    candidates, and a background corpus cloud — each with its raw cosine score.

    Args:
        query_vector: 384-dim query embedding
        rows: List of player records with vectors
        candidate_names: Set of player names that were ranked as candidates

    Returns:
        Dict with metric, points, and explanation for scatter plot visualization
    """
    rows = [r for r in rows if r.get("vector") is not None]
    if not rows:
        return None

    qv = np.asarray(query_vector, dtype=float)
    vecs = np.asarray([r["vector"] for r in rows], dtype=float)

    # Cosine similarity of every point to the query (the ranking signal).
    q_norm = np.linalg.norm(qv) + 1e-9
    v_norms = np.linalg.norm(vecs, axis=1) + 1e-9
    sims = (vecs @ qv) / (v_norms * q_norm)

    # PCA to 2D over [query + corpus] so positions are comparable.
    matrix = np.vstack([qv, vecs])
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    candidates = set(candidate_names or [])
    points = [
        {
            "label": "Your query",
            "x": float(coords[0, 0]),
            "y": float(coords[0, 1]),
            "similarity": 1.0,
            "type": "query",
        }
    ]
    for i, r in enumerate(rows):
        name = r.get("player_name", "?")
        points.append(
            {
                "label": name,
                "x": float(coords[i + 1, 0]),
                "y": float(coords[i + 1, 1]),
                "similarity": round(float(sims[i]), 4),
                "type": "candidate" if name in candidates else "corpus",
                "club": r.get("squad"),
                "position": r.get("position"),
                "season": r.get("season"),
            }
        )
    return {
        "metric": "cosine",
        "points": points,
        "explanation": (
            "2D PCA of the 384-dim sentence embeddings. The gold point is your "
            "query; points nearer to it are more semantically similar. 'similarity' "
            "is the raw cosine score (−1…1) used to rank candidates."
        ),
    }
