import json
import re
import string
import sys
from collections import Counter
from app import qa
from app.graph_store import GraphStore


def norm(s: str) -> str:
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(pred: str, gold: str) -> bool:
    return norm(gold) in norm(pred)


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def path_hit(facts: list[dict], gold_entities: list[str]) -> bool:
    ents = {x for f in facts for x in (f["s"], f["o"])}
    return all(g in ents for g in gold_entities)


def run(path: str = "data/eval_questions.jsonl"):
    store = GraphStore()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    stats = {"graph_contains": 0, "vector_contains": 0, "path_hit": 0}
    for q in rows:
        g = qa.answer(store, q["question"])
        b = qa.baseline_answer(store, q["question"])
        stats["graph_contains"] += contains(g["answer"], q["answer"])
        stats["vector_contains"] += contains(b, q["answer"])
        stats["path_hit"] += path_hit(g["facts"], q["gold_entities"])
        print(f"- {q['question']}\n  graph f1={f1(g['answer'], q['answer']):.2f} "
              f"vector f1={f1(b, q['answer']):.2f}")
    n = len(rows)
    print({k: f"{v}/{n}" for k, v in stats.items()})
    store.close()


if __name__ == "__main__":
    run(*(sys.argv[1:2]))
