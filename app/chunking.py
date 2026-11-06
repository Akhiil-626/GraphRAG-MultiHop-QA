import re


def chunk_text(text: str, max_chars: int = 400, overlap: int = 1) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], []
    for s in sentences:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
