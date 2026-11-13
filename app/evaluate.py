import collections
import datetime
import json
import os
import re
import string
import sys
import time
from collections import Counter
from app import qa, llm
from app.graph_store import GraphStore
from app.retrieve import retrieve
from app.schema import RELATIONS


def norm(s: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, articles, and extra spaces."""
    s = "".join(c for c in str(s).lower() if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(pred: str, gold: str) -> bool:
    """Check whether normalized gold answer appears as a substring of normalized prediction."""
    return norm(gold) in norm(pred)


def f1(pred: str, gold: str) -> float:
    """Calculate token-level unigram F1 score between prediction and gold answer."""
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def path_hit(facts: list[dict], gold_entities: list[str]) -> bool:
    """Check if all gold entity names are present in retrieved facts (case-insensitive set membership).

    Note: This measures entity set coverage, NOT path connectivity.
    """
    ents = {norm(x) for f in facts for x in (f.get("s", ""), f.get("o", ""))}
    return all(norm(g) in ents for g in gold_entities)


def recall_at_k(retrieved: list[str], gold: list[str], k: int | None = None) -> float:
    """Calculate Recall@K: fraction of gold chunks retrieved within top-K.

    Denominator: len(set(gold)). Returns 0.0 if gold is empty.
    """
    if not gold:
        return 0.0
    ret_k = retrieved[:k] if (k is not None and k > 0) else retrieved
    hits = len(set(ret_k) & set(gold))
    return hits / len(set(gold))


def precision_at_k(retrieved: list[str], gold: list[str], k: int | None = None) -> float:
    """Calculate Precision@K.

    Denominator:
    - If k is explicitly provided and > 0, denominator is k (standard IR Precision@K budget).
    - If k is None, denominator is len(retrieved) (precision over the returned set).
    Returns 0.0 if denominator is 0.
    """
    ret_k = retrieved[:k] if (k is not None and k > 0) else retrieved
    denom = k if (k is not None and k > 0) else len(ret_k)
    if denom == 0:
        return 0.0
    hits = len(set(ret_k) & set(gold))
    return hits / denom


def hit_rate_at_k(retrieved: list[str], gold: list[str], k: int | None = None) -> float:
    """Calculate HitRate@K: 1.0 if at least one gold chunk is in top-K, else 0.0."""
    if not gold:
        return 0.0
    ret_k = retrieved[:k] if (k is not None and k > 0) else retrieved
    return 1.0 if len(set(ret_k) & set(gold)) > 0 else 0.0


def connected_path_exists(facts: list[dict], source: str, target: str,
                          directed: bool = False, validate_schema: bool = True,
                          allowed_relations: set[str] | list[str] | None = None) -> tuple[bool, list[str]]:
    """Determine if a continuous valid path exists between source and target in retrieved facts using BFS.

    - Normalizes entity names case-insensitively.
    - If directed=False (default): explores edges bidirectionally (reflecting graph expansion).
    - If directed=True: explores only outgoing edges (startNode -> endNode).
    - If validate_schema=True (default): validates that each edge uses an ontology relationship
      and conforms to allowed endpoint types (s_type and o_type when present).
    - If allowed_relations is specified: restricts traversal to that subset of relations.
    Returns:
        (path_found: bool, path: list[str])
    """
    src = norm(source)
    tgt = norm(target)
    if not src or not tgt:
        return False, []
    if src == tgt:
        return True, [src]

    allowed_rel_set = set(allowed_relations) if allowed_relations is not None else None

    adj = collections.defaultdict(list)
    for f in facts:
        rel = f.get("rel")
        if validate_schema:
            if not rel or rel not in RELATIONS:
                continue
            if allowed_rel_set is not None and rel not in allowed_rel_set:
                continue
            s_type = f.get("s_type")
            o_type = f.get("o_type")
            allowed_src, allowed_tgt = RELATIONS[rel]
            if s_type and s_type not in allowed_src:
                continue
            if o_type and o_type not in allowed_tgt:
                continue

        u = norm(f.get("s", ""))
        v = norm(f.get("o", ""))
        if not u or not v or u == v:
            continue

        adj[u].append(v)
        if not directed:
            adj[v].append(u)

    queue = collections.deque([[src]])
    visited = {src}

    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == tgt:
            return True, path
        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return False, []


def assess_hypothesis_calibration(answer_text: str) -> dict:
    """Assess whether the answer frames multi-hop causal connections as hypotheses

    rather than asserting unproven definitive causation.
    """
    ans_lower = answer_text.lower()
    hypothesis_markers = [
        "hypothesis", "hypothesized", "possible", "potential", "suggests",
        "likely", "correlat", "unproven", "unverified", "investigation",
        "may be", "might be", "not proven", "not definitive", "co-occurrence"
    ]
    absolute_markers = [
        "definitively caused", "proven to cause", "conclusively caused", "undoubtedly caused"
    ]
    found_hypothesis = [m for m in hypothesis_markers if m in ans_lower]
    found_absolute = [m for m in absolute_markers if m in ans_lower]

    is_calibrated = bool(found_hypothesis and not found_absolute)
    return {
        "is_calibrated": is_calibrated,
        "hypothesis_markers_found": found_hypothesis,
        "absolute_markers_found": found_absolute,
        "summary": "Calibrated as hypothesis" if is_calibrated else "Not calibrated or definitive claim detected"
    }


def evaluate_single(store: GraphStore, item: dict, hops: int = 3, k_baseline: int = 5) -> dict:
    """Evaluate a single question across hybrid GraphRAG and vector-only baseline."""
    q = item["question"]
    gold_ans = item["answer"]
    gold_entities = item.get("gold_entities", [])
    primary_chunks = item.get("primary_chunks", [])
    contextual_chunks = item.get("contextual_chunks", [])
    all_gold_chunks = primary_chunks + contextual_chunks

    # ---------------------------------------------------------
    # 1. Hybrid Retrieval & QA (with internal timing)
    # ---------------------------------------------------------
    t_ret_start = time.perf_counter()
    r = retrieve(store, q, hops=hops)
    retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

    t_gen_start = time.perf_counter()
    facts_str = "\n".join(f"{f['s']} -[{f['rel']}]-> {f['o']} (chunks: {', '.join(f['chunks'])})"
                          for f in r["facts"])
    chunks_str = "\n".join(f"[{c['id']}] {c['text']}" for c in r["chunks"])
    raw_answer = llm.chat(qa.GRAPH_PROMPT.format(facts=facts_str, chunks=chunks_str, q=q))
    supplied_chunk_ids = [c["id"] for c in r["chunks"]]
    citations = qa.validate_citations(raw_answer, supplied_chunk_ids)

    if not citations["membership_valid"] and supplied_chunk_ids:
        retry_prompt = qa.CITATION_REPAIR_PROMPT.format(q=q, chunks=chunks_str)
        retry_answer = llm.chat(retry_prompt)
        retry_citations = qa.validate_citations(retry_answer, supplied_chunk_ids)
        if retry_citations["membership_valid"]:
            raw_answer = retry_answer
            citations = retry_citations

    generation_ms = (time.perf_counter() - t_gen_start) * 1000
    total_ms = retrieval_ms + generation_ms

    # Retrieval metrics for hybrid
    k_retrieved = len(supplied_chunk_ids)
    hybrid_retrieval_metrics = {
        "k_budget": k_retrieved,
        "primary_recall": recall_at_k(supplied_chunk_ids, primary_chunks),
        "primary_precision": precision_at_k(supplied_chunk_ids, primary_chunks, k=k_retrieved),
        "primary_hit_rate": hit_rate_at_k(supplied_chunk_ids, primary_chunks),
        "overall_recall": recall_at_k(supplied_chunk_ids, all_gold_chunks),
        "overall_precision": precision_at_k(supplied_chunk_ids, all_gold_chunks, k=k_retrieved),
        "overall_hit_rate": hit_rate_at_k(supplied_chunk_ids, all_gold_chunks),
    }

    # Graph path connectivity
    src_ent = gold_entities[0] if gold_entities else ""
    tgt_ent = gold_entities[-1] if len(gold_entities) > 1 else src_ent
    undirected_conn, undirected_path = connected_path_exists(r["facts"], src_ent, tgt_ent, directed=False)
    directed_conn, directed_path = connected_path_exists(r["facts"], src_ent, tgt_ent, directed=True)

    connectivity_report = {
        "source_entity": src_ent,
        "target_entity": tgt_ent,
        "undirected_connected": undirected_conn,
        "undirected_path": undirected_path,
        "directed_connected": directed_conn,
        "directed_path": directed_path,
        "path_hit_entity_set": path_hit(r["facts"], gold_entities),
        "notes": "Graph expansion explores undirected paths. Directed paths fail when edges point towards the source node (e.g. Complaint -> Bank A)."
    }

    # Answer metrics
    ans_contains = contains(raw_answer, gold_ans)
    ans_f1 = f1(raw_answer, gold_ans)
    calib = assess_hypothesis_calibration(raw_answer) if "duplicate debits" in q.lower() else None

    # Audit chunk texts for citations
    supplied_chunk_map = {c["id"]: c["text"] for c in r["chunks"]}
    citation_audit = [
        {"chunk_id": cid, "text": supplied_chunk_map.get(cid, "<not found>"), "is_valid": cid in supplied_chunk_map}
        for cid in citations["cited_chunk_ids"]
    ]

    hybrid_result = {
        "answer": raw_answer,
        "contains": ans_contains,
        "f1": round(ans_f1, 4),
        "hypothesis_calibration": calib,
        "citations": citations,
        "citation_audit": citation_audit,
        "retrieved_chunk_ids": supplied_chunk_ids,
        "retrieval_metrics": hybrid_retrieval_metrics,
        "connectivity": connectivity_report,
        "latency": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2)
        }
    }

    # ---------------------------------------------------------
    # 2. Vector Baseline (Equitable Comparison)
    # ---------------------------------------------------------
    qv = llm.embed([q])[0]
    # Controlled budget matching hybrid chunk count, plus standard k=5
    b_chunks_equal = store.similar_chunks(qv, k=k_retrieved)
    b_ids_equal = [c["id"] for c in b_chunks_equal]

    t_b_start = time.perf_counter()
    b_chunks_std = store.similar_chunks(qv, k=k_baseline)
    b_ret_ms = (time.perf_counter() - t_b_start) * 1000

    b_ids_std = [c["id"] for c in b_chunks_std]
    t_b_gen = time.perf_counter()
    ctx_std = "\n".join(f"[{c['id']}] {c['text']}" for c in b_chunks_std)
    b_answer_std = llm.chat(qa.VECTOR_PROMPT.format(chunks=ctx_std, q=q))
    b_gen_ms = (time.perf_counter() - t_b_gen) * 1000

    baseline_metrics_std = {
        "k_budget": k_baseline,
        "primary_recall": recall_at_k(b_ids_std, primary_chunks),
        "primary_precision": precision_at_k(b_ids_std, primary_chunks, k=k_baseline),
        "primary_hit_rate": hit_rate_at_k(b_ids_std, primary_chunks),
        "overall_recall": recall_at_k(b_ids_std, all_gold_chunks),
        "overall_precision": precision_at_k(b_ids_std, all_gold_chunks, k=k_baseline),
        "overall_hit_rate": hit_rate_at_k(b_ids_std, all_gold_chunks),
    }

    baseline_metrics_equal = {
        "k_budget": k_retrieved,
        "primary_recall": recall_at_k(b_ids_equal, primary_chunks),
        "primary_precision": precision_at_k(b_ids_equal, primary_chunks, k=k_retrieved),
        "primary_hit_rate": hit_rate_at_k(b_ids_equal, primary_chunks),
        "overall_recall": recall_at_k(b_ids_equal, all_gold_chunks),
        "overall_precision": precision_at_k(b_ids_equal, all_gold_chunks, k=k_retrieved),
        "overall_hit_rate": hit_rate_at_k(b_ids_equal, all_gold_chunks),
    }

    baseline_result = {
        "answer": b_answer_std,
        "contains": contains(b_answer_std, gold_ans),
        "f1": round(f1(b_answer_std, gold_ans), 4),
        "retrieved_chunk_ids_std_k5": b_ids_std,
        "retrieval_metrics_std_k5": baseline_metrics_std,
        "retrieved_chunk_ids_equal_k": b_ids_equal,
        "retrieval_metrics_equal_k": baseline_metrics_equal,
        "latency": {
            "retrieval_ms": round(b_ret_ms, 2),
            "generation_ms": round(b_gen_ms, 2),
            "total_ms": round(b_ret_ms + b_gen_ms, 2)
        }
    }

    return {
        "question": q,
        "gold_answer": gold_ans,
        "gold_entities": gold_entities,
        "primary_chunks": primary_chunks,
        "contextual_chunks": contextual_chunks,
        "hybrid": hybrid_result,
        "vector_baseline": baseline_result
    }


def run(path: str = "data/eval_questions.jsonl", output_dir: str = "artifacts",
        hops: int = 3, k_baseline: int = 5) -> dict:
    """Execute the Phase 3D evaluation suite across benchmark questions and save structured results."""
    store = GraphStore()
    rows = [json.loads(line) for line in open(path) if line.strip()]

    print(f"\n================================================================================")
    print(f"GraphRAG Phase 3D Evaluation Suite (Questions: {len(rows)}, Hops: {hops}, Baseline K: {k_baseline})")
    print(f"================================================================================\n")

    per_question_results = []
    for idx, item in enumerate(rows):
        print(f"Evaluating Q{idx+1}: {item['question']}")
        res = evaluate_single(store, item, hops=hops, k_baseline=k_baseline)
        per_question_results.append(res)

        h = res["hybrid"]
        b = res["vector_baseline"]
        print(f"  Hybrid: Contains={h['contains']} | F1={h['f1']} | "
              f"Primary Recall={h['retrieval_metrics']['primary_recall']:.2f} | "
              f"Undirected Connected={h['connectivity']['undirected_connected']} | "
              f"Citations Valid={h['citations']['membership_valid']} ({len(h['citations']['valid_chunk_ids'])} valid)")
        print(f"  Vector (k={k_baseline}): Contains={b['contains']} | F1={b['f1']} | "
              f"Primary Recall={b['retrieval_metrics_std_k5']['primary_recall']:.2f}")
        print(f"  Latency: Hybrid Retrieval={h['latency']['retrieval_ms']}ms, Gen={h['latency']['generation_ms']}ms | "
              f"Vector Total={b['latency']['total_ms']}ms\n")

    # Aggregate statistics
    n = len(per_question_results)
    agg = {
        "n_questions": n,
        "hybrid_contains_rate": sum(1 for r in per_question_results if r["hybrid"]["contains"]) / n,
        "vector_contains_rate": sum(1 for r in per_question_results if r["vector_baseline"]["contains"]) / n,
        "hybrid_avg_f1": sum(r["hybrid"]["f1"] for r in per_question_results) / n,
        "vector_avg_f1": sum(r["vector_baseline"]["f1"] for r in per_question_results) / n,
        "hybrid_primary_recall_avg": sum(r["hybrid"]["retrieval_metrics"]["primary_recall"] for r in per_question_results) / n,
        "vector_primary_recall_avg": sum(r["vector_baseline"]["retrieval_metrics_std_k5"]["primary_recall"] for r in per_question_results) / n,
        "hybrid_path_connectivity_rate": sum(1 for r in per_question_results if r["hybrid"]["connectivity"]["undirected_connected"]) / n,
        "hybrid_citation_validity_rate": sum(1 for r in per_question_results if r["hybrid"]["citations"]["membership_valid"]) / n,
        "hybrid_avg_retrieval_ms": sum(r["hybrid"]["latency"]["retrieval_ms"] for r in per_question_results) / n,
        "hybrid_avg_generation_ms": sum(r["hybrid"]["latency"]["generation_ms"] for r in per_question_results) / n,
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"eval_results_{ts}.json")

    full_output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "questions_file": path,
            "hops": hops,
            "k_baseline": k_baseline,
            "precision_k_denominator": "k_budget (fixed budget slots)",
            "citation_validation_type": "chunk_id_format_and_membership (syntactic presence only)"
        },
        "aggregate": agg,
        "questions": per_question_results
    }

    with open(report_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print("================================================================================")
    print("Aggregate Summary:")
    print(f"  Hybrid Contains Match:       {agg['hybrid_contains_rate'] * 100:.1f}% ({int(agg['hybrid_contains_rate']*n)}/{n})")
    print(f"  Vector Baseline Contains:    {agg['vector_contains_rate'] * 100:.1f}% ({int(agg['vector_contains_rate']*n)}/{n})")
    print(f"  Hybrid Primary Chunk Recall: {agg['hybrid_primary_recall_avg'] * 100:.1f}%")
    print(f"  Vector Primary Chunk Recall: {agg['vector_primary_recall_avg'] * 100:.1f}%")
    print(f"  Path Connectivity Rate:      {agg['hybrid_path_connectivity_rate'] * 100:.1f}%")
    print(f"  Citation Validity Rate:      {agg['hybrid_citation_validity_rate'] * 100:.1f}%")
    print(f"  Avg Hybrid Retrieval Time:   {agg['hybrid_avg_retrieval_ms']:.1f} ms")
    print(f"  Avg Hybrid Generation Time:  {agg['hybrid_avg_generation_ms'] / 1000:.1f} s")
    print(f"Structured report saved to: {report_file}")
    print("================================================================================\n")

    store.close()
    return full_output


if __name__ == "__main__":
    args = sys.argv[1:]
    target_path = args[0] if len(args) > 0 else "data/eval_questions.jsonl"
    run(path=target_path)
