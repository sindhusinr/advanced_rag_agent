from presidio_analyzer import AnalyzerEngine

# Create analyzer once during application startup
# Avoids reloading NLP models on every request
analyzer = AnalyzerEngine()

# Restrict detection to only the entities required by the business
PII_TYPES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "PERSON",
]

def detect_pii(text: str):
    """
    Detect PII entities from user input.

    Args:
        text (str): User query

    Returns:
        list: Presidio detection results
    """

    results = analyzer.analyze(
        text=text,
        entities=PII_TYPES,
        language="en"
    )

    return results