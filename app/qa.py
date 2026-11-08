import re
from app import llm
from app.graph_store import GraphStore
from app.retrieve import retrieve

GRAPH_PROMPT = """Answer the question using ONLY the provided evidence. Follow these rules:
1. Base factual claims strictly on the text in SOURCE CHUNKS.
2. Treat GRAPH FACTS as structural pathways and hypotheses of co-occurrence, not definitive proof of causation. Multi-hop connections must be framed as hypotheses unless the source text explicitly asserts a direct causal link.
3. Do not assume an extracted relationship label (e.g. DEPLOYED_ON, AFFECTS, CHANGED_IN) is factually accurate if the underlying source text does not support it.
4. Clearly separate directly stated facts from inferred or hypothesized connections.
5. Cite the supporting chunk id for each claim using brackets, e.g. [ENG-1-0].
6. If the evidence is insufficient to fully establish a conclusion, state what is known and what remains unverified.

GRAPH FACTS:
{facts}
SOURCE CHUNKS:
{chunks}
Question: {q}"""

VECTOR_PROMPT = """Answer using ONLY these chunks. If insufficient, say so.
{chunks}
Question: {q}"""


def validate_citations(answer_text: str, supplied_chunk_ids: list[str] | set[str]) -> dict:
    """Validate that chunk citations in the answer belong to supplied evidence chunks.

    Validates chunk-ID membership (syntactic presence), not semantic evidence support.
    """
    supplied_set = set(supplied_chunk_ids)
    raw_matches = re.findall(r"\[([A-Za-z0-9_.-]+)\]", answer_text)
    cited_ids = []
    for cid in raw_matches:
        if cid not in cited_ids:
            cited_ids.append(cid)
    valid_ids = [cid for cid in cited_ids if cid in supplied_set]
    unsupported_ids = [cid for cid in cited_ids if cid not in supplied_set]
    return {
        "cited_chunk_ids": cited_ids,
        "valid_chunk_ids": valid_ids,
        "unsupported_chunk_ids": unsupported_ids,
        "membership_valid": bool(cited_ids and not unsupported_ids),
        "validation_type": "chunk_id_membership",
    }


def answer(store: GraphStore, question: str, **kw) -> dict:
    r = retrieve(store, question, **kw)
    facts = "\n".join(f"{f['s']} -[{f['rel']}]-> {f['o']} (chunks: {', '.join(f['chunks'])})"
                      for f in r["facts"])
    chunks = "\n".join(f"[{c['id']}] {c['text']}" for c in r["chunks"])
    raw_answer = llm.chat(GRAPH_PROMPT.format(facts=facts, chunks=chunks, q=question))
    supplied_ids = [c["id"] for c in r["chunks"]]
    citations = validate_citations(raw_answer, supplied_ids)
    return {"answer": raw_answer, "citations": citations, **r}


def baseline_answer(store: GraphStore, question: str, k: int = 5) -> str:
    chunks = store.similar_chunks(llm.embed([question])[0], k)
    ctx = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return llm.chat(VECTOR_PROMPT.format(chunks=ctx, q=question))
