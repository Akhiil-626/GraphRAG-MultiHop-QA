import re
from difflib import SequenceMatcher
import numpy as np
from app import llm


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 .\-]", "", s.lower())
    return " ".join(s.split())


def _blocked(a: str, b: str) -> bool:
    """True = must NOT merge. Stops 'release 4.2' ~ '4.3' and 'gateway x' ~ 'gateway y'."""
    da, db = re.findall(r"\d+", a), re.findall(r"\d+", b)
    sa = {t for t in re.findall(r"[a-z0-9.]+", a) if len(t) <= 2}
    sb = {t for t in re.findall(r"[a-z0-9.]+", b) if len(t) <= 2}
    return da != db or sa != sb


class EntityResolver:
    """Collapses aliases to one canonical node per entity type."""

    def __init__(self, fuzzy: float = 0.93, cosine: float = 0.95):
        self.fuzzy, self.cosine = fuzzy, cosine
        self.canon: dict[str, list[tuple[str, np.ndarray | None]]] = {}

    def resolve(self, name: str, etype: str) -> str:
        n = _norm(name)
        items = self.canon.setdefault(etype, [])
        for cn, _ in items:
            if cn.replace(" ", "") == n.replace(" ", ""):
                return cn
        for cn, _ in items:
            if not _blocked(cn, n) and SequenceMatcher(None, cn, n).ratio() >= self.fuzzy:
                return cn
        vec = np.array(llm.embed([n])[0])
        for i, (cn, cv) in enumerate(items):
            if cv is None:
                cv = np.array(llm.embed([cn])[0])
                items[i] = (cn, cv)
            sim = float(vec @ cv / (np.linalg.norm(vec) * np.linalg.norm(cv)))
            if not _blocked(cn, n) and sim >= self.cosine:
                return cn
        items.append((n, vec))
        return n
