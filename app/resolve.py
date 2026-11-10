import re
from difflib import SequenceMatcher
import numpy as np
from app import llm
from app.schema import RESERVED_NAMES

# Common English short stopwords that should not prevent valid entity merges
SHORT_STOPWORDS = {
    "in", "to", "of", "on", "at", "by", "for", "is", "a", "an", "the",
    "or", "as", "it", "so", "if", "my"
}


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 .\-]", "", s.lower())
    return " ".join(s.split())


def _blocked(a: str, b: str) -> bool:
    """True = must NOT merge. Stops 'release 4.2' ~ '4.3' and 'gateway x' ~ 'gateway y'.

    Preserves meaningful abbreviations and identifiers (e.g. 'v1', 'v2', 'x', 'y', 'db')
    while filtering out common short stopwords (e.g. 'in', 'to', 'by') that would otherwise
    falsely block legitimate merges.
    """
    da, db = re.findall(r"\d+", a), re.findall(r"\d+", b)
    sa = {t for t in re.findall(r"[a-z0-9.]+", a) if len(t) <= 2 and t not in SHORT_STOPWORDS}
    sb = {t for t in re.findall(r"[a-z0-9.]+", b) if len(t) <= 2 and t not in SHORT_STOPWORDS}
    return da != db or sa != sb


class EntityResolver:
    """Collapses aliases to one canonical node per entity type with cross-type conflict detection."""

    def __init__(self, fuzzy: float = 0.93, cosine: float = 0.95):
        self.fuzzy, self.cosine = fuzzy, cosine
        self.canon: dict[str, list[tuple[str, np.ndarray | None]]] = {}
        self.entity_types: dict[str, str] = {}

    def resolve(self, name: str, etype: str) -> str:
        n = _norm(name)
        if not n:
            raise ValueError("Cannot resolve empty entity name")
        if n in RESERVED_NAMES:
            raise ValueError(f"Cannot resolve reserved ontology name '{name}' as an entity")

        # Cross-type conflict detection: prevent same entity name across conflicting types
        if n in self.entity_types and self.entity_types[n] != etype:
            prev_type = self.entity_types[n]
            raise ValueError(
                f"Cross-type entity conflict: entity '{n}' is already registered as '{prev_type}', "
                f"cannot register as conflicting type '{etype}'"
            )

        items = self.canon.setdefault(etype, [])
        for cn, _ in items:
            if cn.replace(" ", "") == n.replace(" ", ""):
                self.entity_types[n] = etype
                return cn
        for cn, _ in items:
            if not _blocked(cn, n) and SequenceMatcher(None, cn, n).ratio() >= self.fuzzy:
                self.entity_types[n] = etype
                return cn
        vec = np.array(llm.embed([n])[0])
        for i, (cn, cv) in enumerate(items):
            if cv is None:
                cv = np.array(llm.embed([cn])[0])
                items[i] = (cn, cv)
            sim = float(vec @ cv / (np.linalg.norm(vec) * np.linalg.norm(cv)))
            if not _blocked(cn, n) and sim >= self.cosine:
                self.entity_types[n] = etype
                return cn
        items.append((n, vec))
        self.entity_types[n] = etype
        return n
