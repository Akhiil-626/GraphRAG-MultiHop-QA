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
    store.close()
