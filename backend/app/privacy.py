from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import spacy

# Initialize engines
# We use spaCy for the analyzer engine backend
analyzer = AnalyzerEngine(default_score_threshold=0.4)
anonymizer = AnonymizerEngine()

def scrub_sensitive_data(text: str) -> str:
    """
    Detects and masks PII (names, phone numbers, passport numbers, etc.) from the input text.
    
    Args:
        text: The raw markdown text to be scrubbed.
        
    Returns:
        The anonymized text with sensitive entities masked.
    """
    # Analyze the text for PII
    results = analyzer.analyze(
        text=text, 
        entities=["PERSON", "PHONE_NUMBER", "LOCATION", "EMAIL_ADDRESS", "PASSPORT"],
        language='en'
    )
    
    # Anonymize the identified PII
    # Default behavior is to replace with <ENTITY_TYPE>
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    return anonymized_result.text
