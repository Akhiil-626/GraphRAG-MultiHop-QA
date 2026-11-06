# GraphRAG

GraphRAG for multi-hop question answering over anonymized bank complaints.

## Pipeline

```
Text -> PII Scrub -> Chunk -> Triple Extraction (Ollama) -> Entity Resolution
                                                                   |
User Question -> Hybrid Retrieval (Seeds + k-hop Graph + Chunks) <- Neo4j
                          |
                   LLM Cited Answer
```

## Prerequisites
- Docker & Docker Compose
- Ollama (running locally)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
ollama pull llama3.1 && ollama pull nomic-embed-text
pytest
python -m app.ingest
python -m app.ask "Why are Bank A customers seeing duplicate debits?"
python -m app.evaluate
uvicorn app.api:app --reload
```

## Notes
- **Hypothesis only**: Graph paths show co-occurrence, not proven causation.
- **Entity resolution**: Tune resolver thresholds in `app/resolve.py`.
- **Privacy**: The privacy layer is regex-based and should be upgraded (e.g. Presidio) before real data.
