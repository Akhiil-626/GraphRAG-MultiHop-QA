from fastapi import FastAPI
from pydantic import BaseModel
from app import privacy, qa
from app.graph_store import GraphStore

app = FastAPI(title="GraphRAG")
store = GraphStore()


class Ask(BaseModel):
    question: str
    hops: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    r = qa.answer(store, body.question, hops=body.hops)
    return {
        "answer": r["answer"],
        "edges": r["facts"],
        "chunks": r["chunks"],
        "citations": r.get("citations"),
    }


@app.get("/report/shared-gateways")
def shared_gateways():
    return privacy.suppress_small(store.shared_gateway_report())
