from typing import List, Optional

# Presidio + spaCy are heavy native-backed libraries. We build the engines LAZILY
# (on first scrub) rather than at import, so merely importing this module — and
# therefore booting the API server — stays light and avoids loading several
# native libs at once (a cause of intermittent fork/OpenMP segfaults at startup).
_nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}
_analyzer = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp_engine = NlpEngineProvider(nlp_configuration=_nlp_configuration).create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


# Default to *contact* PII only. We intentionally do NOT scrub PERSON / LOCATION
# here: in a scouting dossier those are the player and club names we want to
# retain for embedding and retrieval. Pass `entities` to override.
DEFAULT_ENTITIES: List[str] = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_PASSPORT",
    "IBAN_CODE",
    "CREDIT_CARD",
]


def scrub_sensitive_data(text: str, entities: Optional[List[str]] = None) -> str:
    """
    Detects and masks contact PII (phone numbers, emails, passport/IBAN, etc.)
    from the input text.

    Args:
        text: The raw markdown text to be scrubbed.
        entities: Optional override of the entity types to detect/mask.

    Returns:
        The anonymized text with sensitive entities masked.
    """
    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(
        text=text,
        entities=entities or DEFAULT_ENTITIES,
        language="en",
    )

    # Default behavior is to replace each match with <ENTITY_TYPE>.
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text
