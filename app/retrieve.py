from app import llm
from app.graph_store import GraphStore


def retrieve(store: GraphStore, question: str, k_seeds: int = 3,
             hops: int = 3, k_chunks: int = 3, max_chunks: int = 12) -> dict:
    qv = llm.embed([question])[0]
    seeds = store.seed_entities(qv, k_seeds)
    facts = store.expand(seeds, hops)
    vec_chunks = store.similar_chunks(qv, k_chunks)

    # Preserve vector ranking
    vec_ids = [c["id"] for c in vec_chunks if "id" in c]

    # Preserve graph discovery order
    graph_ids = []
    for f in facts:
        for cid in f.get("chunks", []):
            if cid and cid not in graph_ids:
                graph_ids.append(cid)

    # Prioritize shared evidence, followed by vector rank, then graph discovery
    shared_ids = [cid for cid in vec_ids if cid in graph_ids]
    vec_only_ids = [cid for cid in vec_ids if cid not in shared_ids]
    graph_only_ids = [cid for cid in graph_ids if cid not in shared_ids]

    ordered_ids = shared_ids + vec_only_ids + graph_only_ids
    selected_ids = ordered_ids[:max_chunks]

    return {
        "seeds": seeds,
        "facts": facts,
        "chunks": store.chunk_texts(selected_ids),
        "graph_chunk_ids": graph_ids,
        "vector_chunk_ids": vec_ids,
    }
