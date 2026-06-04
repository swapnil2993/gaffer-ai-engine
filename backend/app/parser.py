import os
import tempfile
from typing import List, Optional

import ebooklib
from ebooklib import epub

from backend.app.privacy import scrub_sensitive_data
from backend.app.schemas import TacticalTheoryPayload

# Docling pulls in torch and loads layout models. Instantiate it lazily (only
# when a document is actually parsed) so importing this module — and therefore
# booting the API server — stays fast and doesn't load torch before Milvus Lite
# forks its embedded server.
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Enable high-fidelity table structural analysis for PDFs
        pdf_options = PdfPipelineOptions()
        pdf_options.do_table_structure = True
        pdf_options.table_structure_options.do_cell_matching = True

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                # HTML/EPUB uses default options which are usually sufficient
            }
        )
    return _converter



def _epub_to_html(epub_path: str) -> str:
    """Converts EPUB chapters to a single semantic HTML string."""
    book = epub.read_epub(epub_path)
    # Start with a proper HTML5 structure to help Docling's parser
    full_html = "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # We keep the raw HTML from the EPUB because it contains
            # the semantic <h1>, <h2>, and <table> tags Docling needs.
            full_html += item.get_content().decode("utf-8")
    full_html += "</body></html>"
    return full_html


def _convert_to_document(file_path: str):
    """Convert a file to a DoclingDocument (EPUB → temp HTML → Docling)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".epub":
        print(f"Detected EPUB. Converting {file_path} to HTML for Docling...")
        html_content = _epub_to_html(file_path)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
            tmp.write(html_content)
            tmp_path = tmp.name
        try:
            return _get_converter().convert(tmp_path).document
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    print(f"Converting {file_path} using Docling...")
    return _get_converter().convert(file_path).document


def ingest_tactical_book(file_path: str) -> List[TacticalTheoryPayload]:
    """Ingest a tactical theory book and return its full Markdown (one payload).

    Used by the /theory/document viewer. For indexing, prefer
    ``chunk_tactical_book`` which preserves Docling's section structure.
    """
    document = _convert_to_document(file_path)
    return [
        TacticalTheoryPayload(
            content=document.export_to_markdown(),
            metadata={"source": file_path},
        )
    ]


def chunk_tactical_book(file_path: str) -> List[dict]:
    """Structure-aware chunks of a tactical book using Docling's HybridChunker.

    Instead of flattening to Markdown and char-splitting, we chunk along the
    document's heading hierarchy and PREPEND each chunk's heading path
    ('contextualize'). This keeps chunks topically coherent and gives both the
    embedding and the LLM the section context — improving retrieval relevancy
    and grounding. Token-sized to the embedding model so nothing is truncated.

    Returns: [{"text": <heading path + chunk text>, "heading": "A > B", "source": ...}]
    """
    from docling.chunking import HybridChunker

    document = _convert_to_document(file_path)
    # The tokenizer matches our Milvus embedding model (limit 512).
    # We use 320 as the "sweet spot" to maximize tactical context 
    # while leaving enough room for enriched DNA headers without truncation.
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=320,
        merge_peers=True,
    )
    chunks = []
    for ch in chunker.chunk(document):
        headings = list(getattr(ch.meta, "headings", None) or [])
        chunks.append(
            {
                "text": chunker.contextualize(chunk=ch),
                "heading": " > ".join(headings),
                "source": file_path,
            }
        )
    return chunks
