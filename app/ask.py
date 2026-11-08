import sys
from app import qa
from app.graph_store import GraphStore

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why are Bank A customers seeing duplicate debits?"
    store = GraphStore()
    r = qa.answer(store, q)
    print(r["answer"])
    print("\nEDGES USED:")
    for f in r["facts"]:
        print(f"  {f['s']} -[{f['rel']}]-> {f['o']}  {f['chunks']}")
    if "citations" in r:
        c = r["citations"]
        print(f"\nCITATIONS (membership valid: {c['membership_valid']}):")
        print(f"  Valid: {c['valid_chunk_ids']}")
        if c["unsupported_chunk_ids"]:
            print(f"  Unsupported: {c['unsupported_chunk_ids']}")
    store.close()
