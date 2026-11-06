import json
import sys
from pathlib import Path
from app import privacy
from app.chunking import chunk_text
from app.extract import extract_triples
from app.graph_store import GraphStore
from app.resolve import EntityResolver


def main(path: str = "data/sample_corpus.jsonl", reset: bool = False):
    store = GraphStore()
    store.setup()
    if reset:
        store.clear()
    resolver = EntityResolver()
    n_triples = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        cid = doc["id"].lower() if doc.get("kind") == "complaint" else "none"
        for i, ch in enumerate(chunk_text(privacy.scrub(doc["text"]))):
            chunk_id = f"{doc['id']}-{i}"
            store.add_chunk(chunk_id, ch)  # only the scrubbed text is stored
            for t in extract_triples(ch, cid):
                t["s"] = resolver.resolve(t["s"], t["s_type"])
                t["o"] = resolver.resolve(t["o"], t["o_type"])
                store.add_triple(t, chunk_id)
                n_triples += 1
        print(f"ingested {doc['id']}")
    store.embed_entities()
    store.close()
    print(f"done: {n_triples} triples")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
