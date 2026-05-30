from docling.document_converter import DocumentConverter
from backend.app.schemas import ScoutingPayload
from backend.app.privacy import scrub_sensitive_data

# Initialize Docling converter
converter = DocumentConverter()

def ingest_scouting_pdf(
    file_path: str, 
    player_name: str, 
    position: str, 
    season: str, 
    metrics: dict
) -> ScoutingPayload:
    """
    Ingests a scouting PDF, extracts text and tables as Markdown, 
    scrubs sensitive data, and returns a ScoutingPayload.
    """
    # Convert PDF to layout-aware markdown
    result = converter.convert(file_path)
    markdown_output = result.document.export_to_markdown()
    
    # Scrub sensitive data from the extracted markdown
    clean_text = scrub_sensitive_data(markdown_output)
    
    # Return the validated ScoutingPayload
    return ScoutingPayload(
        player_name=player_name,
        position=position,
        season=season,
        raw_text=clean_text,
        metrics=metrics
    )
