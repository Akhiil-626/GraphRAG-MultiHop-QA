import collections
from app import llm
from app.graph_store import GraphStore


def _evidence_role(cid_facts: list[dict]) -> int:
    """Classify graph evidence role for ordering:
    0: Complaint context (e.g. AGAINST, ABOUT, VIA, REPORTS)
    1: Release / change engineering evidence (e.g. CHANGED_IN, DEPLOYED_ON, Release entity)
    2: Causal engineering evidence (e.g. CAUSED_BY, AFFECTS, Failure entity)
    3: Other / general evidence
    """
    is_release = False
    is_complaint = False
    is_causal = False
    for f in cid_facts:
        rel = f.get("rel") or f.get("r")
        st = f.get("s_type")
        ot = f.get("o_type")
        if rel in {"CHANGED_IN", "DEPLOYED_ON"} or st == "Release" or ot == "Release":
            is_release = True
        elif rel in {"AGAINST", "ABOUT", "VIA", "REPORTS"} or st == "Complaint" or ot == "Complaint":
            is_complaint = True
        elif rel in {"CAUSED_BY", "AFFECTS"} or st == "Failure" or ot == "Failure":
            is_causal = True

    if is_complaint:
        return 0
    if is_release:
        return 1
    if is_causal:
        return 2
    return 3


def retrieve(store: GraphStore, question: str, k_seeds: int = 3,
             hops: int = 3, k_chunks: int = 3, max_chunks: int = 12) -> dict:
    qv = llm.embed([question])[0]
    seeds = store.seed_entities(qv, k_seeds)
    facts = store.expand(seeds, hops)
    vec_chunks = store.similar_chunks(qv, k_chunks)

    # Preserve vector ranking without duplicates
    vec_ids = []
    for c in vec_chunks:
        cid = c.get("id")
        if cid and cid not in vec_ids:
            vec_ids.append(cid)

    # Count distinct facts supporting each graph-derived chunk ID and record first-seen order
    first_seen_graph_ids = []
    fact_support = collections.defaultdict(set)
    chunk_to_facts = collections.defaultdict(list)

    for idx, f in enumerate(facts):
        s = f.get("s")
        rel = f.get("rel") or f.get("r")
        o = f.get("o")
        if s is not None or rel is not None or o is not None:
            fact_key = (f.get("s_type"), s, rel, f.get("o_type"), o)
        else:
            fact_key = idx

        for cid in f.get("chunks", []):
            if cid:
                fact_support[cid].add(fact_key)
                chunk_to_facts[cid].append(f)
                if cid not in first_seen_graph_ids:
                    first_seen_graph_ids.append(cid)

    # Rank graph chunk IDs:
    # 1. Evidence role: complaint context -> release/change -> causal engineering -> other
    # 2. Descending distinct fact support count
    # 3. Original first-seen order
    first_seen_index = {cid: idx for idx, cid in enumerate(first_seen_graph_ids)}
    graph_ids = sorted(
        first_seen_graph_ids,
        key=lambda cid: (
            _evidence_role(chunk_to_facts.get(cid, [])),
            -len(fact_support[cid]),
            first_seen_index[cid],
        ),
    )

    # Prioritize shared evidence (vector rank), vector-only (vector rank), then graph-only (support count rank)
    shared_set = set(vec_ids) & set(graph_ids)
    shared_ids = [cid for cid in vec_ids if cid in shared_set]
    vec_only_ids = [cid for cid in vec_ids if cid not in shared_set]
    graph_only_ids = [cid for cid in graph_ids if cid not in shared_set]

    ordered_ids = shared_ids + vec_only_ids + graph_only_ids
    selected_ids = ordered_ids[:max_chunks]

    return {
        "seeds": seeds,
        "facts": facts,
        "chunks": store.chunk_texts(selected_ids),
        "graph_chunk_ids": graph_ids,
        "vector_chunk_ids": vec_ids,
    }
