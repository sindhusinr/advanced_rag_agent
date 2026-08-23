from advanced_rag_agent.guardrails.prompt_injection import (
    detect_prompt_injection
)

from advanced_rag_agent.guardrails.pii_detector import (
    detect_pii
)


def validate_input(text: str) -> dict:
    """
    Validate user input before sending it to the LLM.

    Checks:
    1. Prompt Injection
    2. PII Detection

    Args:
        text (str): User input

    Returns:
        dict: Validation result
    """

    # Prompt Injection Check
    if detect_prompt_injection(text):
        return {
            "allowed": False,
            "reason": "Prompt Injection Detected"
        }

    # PII Detection Check
    pii_results = detect_pii(text)

    if pii_results:
        return {
            "allowed": False,
            "reason": "PII Detected",
            "entities": list(
                {
                    entity.entity_type
                    for entity in pii_results
                }
            )
        }

    return {
        "allowed": True,
        "reason": None
    }