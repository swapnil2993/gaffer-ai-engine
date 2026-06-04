"""Privacy/PII scrubbing wrapper for data cleaning pipeline."""

from backend.app.privacy import scrub_sensitive_data


def scrub_text(text: str) -> str:
    """Scrub sensitive data (phone, email, passport, credit card) from text.

    Preserves person/location names (player and club names needed for context).
    """
    return scrub_sensitive_data(text)
