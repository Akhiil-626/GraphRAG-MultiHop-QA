from app import llm
from app.graph_store import GraphStore
from app.retrieve import retrieve

GRAPH_PROMPT = """Answer using ONLY the evidence below. Trace the relationship path hop by hop and cite chunk ids like [C-001-0]. Treat paths as hypotheses (co-occurrence), not proven causation. If evidence is insufficient, say so.
GRAPH FACTS:
{facts}
SOURCE CHUNKS:
{chunks}
Question: {q}"""

VECTOR_PROMPT = """Answer using ONLY these chunks. If insufficient, say so.
{chunks}
Question: {q}"""


def answer(store: GraphStore, question: str, **kw) -> dict:
    r = retrieve(store, question, **kw)
    facts = "\n".join(f"{f['s']} -[{f['rel']}]-> {f['o']} (chunks: {', '.join(f['chunks'])})"
                      for f in r["facts"])
    chunks = "\n".join(f"[{c['id']}] {c['text']}" for c in r["chunks"])
    return {"answer": llm.chat(GRAPH_PROMPT.format(facts=facts, chunks=chunks, q=question)), **r}


def baseline_answer(store: GraphStore, question: str, k: int = 5) -> str:
    chunks = store.similar_chunks(llm.embed([question])[0], k)
    ctx = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return llm.chat(VECTOR_PROMPT.format(chunks=ctx, q=question))
