"""Tactical theory document service."""

import json
from backend.app.database import THEORY_COLLECTION, get_client
from backend.app.services.markdown import md_to_html


def fetch_theory_chunks() -> list[dict]:
    """Fetch all enriched chunks from Milvus theory collection."""
    client = get_client()
    client.load_collection(THEORY_COLLECTION)

    chunks = client.query(
        collection_name=THEORY_COLLECTION,
        filter="chunk_id >= 0",
        limit=2000,
        output_fields=[
            "chunk_id",
            "chapter",
            "era",
            "formation",
            "possession_style_tag",
            "defensive_line_height",
            "fbref_metric_weights",
            "text",
            "heading",
        ],
    )

    return sorted(chunks, key=lambda x: x.get("chunk_id", 0))


def build_theory_html() -> str:
    """Build complete theory document HTML."""
    chunks = fetch_theory_chunks()
    html_chunks = []

    for c in chunks:
        weights = c.get("fbref_metric_weights", {})
        active_metrics = {k: v for k, v in weights.items() if v > 0}

        meta_html = f"""
        <div class="chunk-meta">
            <span class="meta-tag era">Era: {c.get('era') or 'Unknown'}</span>
            <span class="meta-tag formation">Formation: {c.get('formation') or 'Unknown'}</span>
            <span class="meta-tag style">Style: {c.get('possession_style_tag') or 'Unknown'}</span>
            <span class="meta-tag line">Line: {c.get('defensive_line_height') or 'Unknown'}</span>
        </div>
        """
        if active_metrics:
            metrics_json = json.dumps(active_metrics, indent=2)
            meta_html += f"<pre class='metrics-box'>Metric Weights: {metrics_json}</pre>"

        content_html = md_to_html(c.get("text", ""))
        html_chunks.append(f"<div class='theory-chunk'>{meta_html}{content_html}</div>")

    return _build_page("".join(html_chunks))


def _build_page(chunks_html: str) -> str:
    """Build the complete HTML page with styling."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structured Tactical DNA — The Inverted Pyramid</title>
<style>
  body {{ background:#0d1b15; color:#e5e7eb; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
          max-width:900px; margin:0 auto; padding:32px 24px; line-height:1.6; }}
  h1 {{ color:#d4af37; border-bottom:1px solid #ffffff22; padding-bottom:.3em; }}
  .theory-chunk {{ background: #1a2e26; border: 1px solid #ffffff11; border-radius: 8px;
                  padding: 20px; margin-bottom: 24px; }}
  .chunk-meta {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }}
  .meta-tag {{ font-size: 11px; padding: 4px 8px; border-radius: 4px; background: #00000044;
               border: 1px solid #ffffff22; color: #9ca3af; }}
  .meta-tag.era {{ color: #d4af37; border-color: #d4af3744; }}
  .metrics-box {{ font-size: 11px; background: #00000066; padding: 10px; border-radius: 4px;
                  color: #22c55e; border-left: 3px solid #22c55e; overflow-x: auto; }}
  h2, h3, h4 {{ color: #d4af37; margin-top: 0; }}
  p {{ margin-bottom: 0; color: #d1d5db; }}
</style></head><body>
<h1>Structured Tactical DNA</h1>
<p style="color:#9ca3af; font-size:13px; margin-bottom:30px;">
  This view shows the raw structured output from the Docling + LLM Enrichment pipeline.
  Every chunk has been translated into machine-readable scouting parameters.
</p>
{chunks_html}
</body></html>"""
