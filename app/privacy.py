import hashlib
import re
from app import config

# Order matters: card before phone before account.
PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[PHONE]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT]"),
]


def scrub(text: str) -> str:
    for pattern, token in PATTERNS:
        text = pattern.sub(token, text)
    return text


def pseudonym(value: str) -> str:
    return "id_" + hashlib.sha256((config.PII_SALT + value).encode()).hexdigest()[:10]


def suppress_small(rows: list[dict], key: str = "complaints") -> list[dict]:
    """Drop aggregate rows below the minimum group size (re-identification guard)."""
    return [r for r in rows if r[key] >= config.MIN_GROUP_SIZE]
