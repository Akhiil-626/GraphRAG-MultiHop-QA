import re
from app import llm
from app.graph_store import GraphStore
from app.retrieve import retrieve

GRAPH_PROMPT = """Answer the question using ONLY the provided evidence. Follow these rules:
1. Answer the question directly in the first sentence. Keep the answer concise and avoid unnecessary background.
2. Base factual claims strictly on the text in SOURCE CHUNKS. Every factual claim derived from a source chunk must cite its exact chunk ID in square brackets, e.g. [ENG-1-0]. Every non-empty answer must include at least one valid chunk citation.
3. Never cite chunk IDs in parentheses. Never omit square brackets around chunk IDs.
4. Only cite chunk IDs actually provided in the retrieved evidence (SOURCE CHUNKS). Never invent or fabricate chunk IDs.
5. Never put graph relationship names (such as AFFECTS, VIA, REPORTS, or CAUSED_BY) in square brackets. Write relationship names as plain text.
6. Do not contradict evidence. If a source explicitly identifies a release or gateway, state it directly rather than claiming it is unspecified.
7. Treat GRAPH FACTS as structural pathways and hypotheses of co-occurrence, not definitive proof of causation. Multi-hop connections must be framed as hypotheses unless the source text explicitly asserts a direct causal link. Distinguish explicit evidence from hypotheses; do not present a possible explanation as an established root cause.
8. If evidence does not establish the complete answer, state exactly what is known and what remains uncertain or unverified.
9. Format your response using this exact structure:
Answer: <direct answer citing source chunk IDs in square brackets, e.g. [ENG-1-0]>
Evidence: <one or more exact chunk IDs from the provided evidence, e.g. [ENG-1-0]>

EXAMPLES OF REQUIRED FORMAT:
Example 1:
Answer: Service Alpha complaints report duplicate debits [C-001-0]. Engineering evidence attributes duplicate debits to callback timeout, which affects Gateway Alpha [ENG-1-0]. Release Alpha changed Gateway Alpha's retry logic [ENG-2-0], but the evidence does not establish that this change caused the incidents.
Evidence: [C-001-0], [ENG-1-0], [ENG-2-0]

Example 2:
Answer: Release Alpha changed Gateway Alpha's retry logic [ENG-2-0]. The Service Beta complaint involved Gateway Alpha [C-003-0].
Evidence: [ENG-2-0], [C-003-0]

Example 3:
Answer: Gateway Alpha is affected by callback timeout [ENG-1-0].
Evidence: [ENG-1-0]

GRAPH FACTS:
{facts}
SOURCE CHUNKS:
{chunks}
Question: {q}"""

CITATION_REPAIR_PROMPT = """Your previous response did not include required chunk citations in square brackets (e.g. [ENG-1-0]).
Rewrite your answer using ONLY the retrieved evidence below. Follow this exact structure:
Answer: <direct answer citing supporting source chunk IDs in square brackets, e.g. [ENG-1-0]>
Evidence: <one or more exact chunk IDs from the provided evidence, e.g. [ENG-1-0]>

Question: {q}

SOURCE CHUNKS:
{chunks}"""

VECTOR_PROMPT = """Answer using ONLY these chunks. If insufficient, say so.
{chunks}
Question: {q}"""


def validate_citations(answer_text: str, supplied_chunk_ids: list[str] | set[str]) -> dict:
    """Validate that chunk citations in the answer belong to supplied evidence chunks.

    Validates chunk-ID membership (syntactic presence), not semantic evidence support.
    """
    supplied_set = set(supplied_chunk_ids)
    raw_matches = re.findall(r"\[([A-Z][A-Z0-9]*(?:-\d+)+)\]", answer_text)
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
    if not citations["membership_valid"] and supplied_ids:
        retry_prompt = CITATION_REPAIR_PROMPT.format(q=question, chunks=chunks)
        retry_answer = llm.chat(retry_prompt)
        retry_citations = validate_citations(retry_answer, supplied_ids)
        if retry_citations["membership_valid"]:
            raw_answer = retry_answer
            citations = retry_citations
    return {"answer": raw_answer, "citations": citations, **r}


def baseline_answer(store: GraphStore, question: str, k: int = 5) -> str:
    chunks = store.similar_chunks(llm.embed([question])[0], k)
    ctx = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return llm.chat(VECTOR_PROMPT.format(chunks=ctx, q=question))
