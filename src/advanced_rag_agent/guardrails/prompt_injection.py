import re

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"forget all instructions",
    r"act as system",
    r"reveal system prompt",
    r"developer message",
    r"jailbreak",
    r"bypass security",
    r"disable safeguards",
]

def detect_prompt_injection(query: str) -> bool:
    query_lower = query.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    return False