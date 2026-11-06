from app import llm
from app.graph_store import GraphStore


def retrieve(store: GraphStore, question: str, k_seeds: int = 3,
             hops: int = 2, k_chunks: int = 3, max_chunks: int = 12) -> dict:
    qv = llm.embed([question])[0]
    seeds = store.seed_entities(qv, k_seeds)
    facts = store.expand(seeds, hops)
    vec_chunks = store.similar_chunks(qv, k_chunks)
    ids = sorted({c for f in facts for c in f["chunks"]} | {c["id"] for c in vec_chunks})
    return {"seeds": seeds, "facts": facts, "chunks": store.chunk_texts(ids[:max_chunks])}
