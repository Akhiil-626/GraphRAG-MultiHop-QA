import inspect
from unittest.mock import MagicMock
from app.retrieve import retrieve
from app.qa import validate_citations, GRAPH_PROMPT, baseline_answer
from app.graph_store import GraphStore


def test_default_retrieval_hops():
    """Verify that retrieve defaults to 3 hops as established in Phase 3b."""
    sig = inspect.signature(retrieve)
    assert sig.parameters["hops"].default == 3
    assert sig.parameters["k_seeds"].default == 3
    assert sig.parameters["k_chunks"].default == 3
    assert sig.parameters["max_chunks"].default == 12


def test_evidence_ordering_preserves_vector_and_graph_order():
    """Verify that evidence combination preserves vector ranking and graph discovery order

    without falling back to naive alphabetical sorting.
    """
    mock_store = MagicMock()
    # Mock vector chunks with specific vector rank: V1, SHARED, V2
    mock_store.seed_entities.return_value = ["Entity:a"]
    mock_store.expand.return_value = [
        {"s": "a", "rel": "REL", "o": "b", "chunks": ["SHARED", "G1"]},
        {"s": "b", "rel": "REL", "o": "c", "chunks": ["G2"]},
    ]
    mock_store.similar_chunks.return_value = [
        {"id": "V1", "text": "Vector 1 text"},
        {"id": "SHARED", "text": "Shared chunk text"},
        {"id": "V2", "text": "Vector 2 text"},
    ]
    mock_store.chunk_texts.side_effect = lambda ids: [{"id": cid, "text": f"Text for {cid}"} for cid in ids]

    res = retrieve(mock_store, "test question", k_seeds=1, hops=3, k_chunks=3, max_chunks=10)

    # Expected order:
    # 1. SHARED (in both, vector rank order)
    # 2. V1, V2 (remaining vector rank)
    # 3. G1, G2 (remaining graph discovery order)
    # If alphabetical, G1 and G2 would precede SHARED, V1, V2.
    expected_ids = ["SHARED", "V1", "V2", "G1", "G2"]
    retrieved_ids = [c["id"] for c in res["chunks"]]
    assert retrieved_ids == expected_ids
    assert res["graph_chunk_ids"] == ["SHARED", "G1", "G2"]
    assert res["vector_chunk_ids"] == ["V1", "SHARED", "V2"]


def test_citation_validation_valid():
    """Verify citation validation passes when all citations belong to supplied chunks."""
    text = "Gateway X was modified in [ENG-2-0] following duplicate debits reported in [C-001-0]."
    supplied = ["C-001-0", "ENG-1-0", "ENG-2-0"]
    result = validate_citations(text, supplied)
    assert result["membership_valid"] is True
    assert result["cited_chunk_ids"] == ["ENG-2-0", "C-001-0"]
    assert result["valid_chunk_ids"] == ["ENG-2-0", "C-001-0"]
    assert result["unsupported_chunk_ids"] == []
    assert result["validation_type"] == "chunk_id_membership"


def test_citation_validation_unsupported():
    """Verify citation validation flags citations not in supplied chunks."""
    text = "The issue originated in [FABRICATED-1-0] as seen in [C-001-0]."
    supplied = ["C-001-0", "ENG-1-0"]
    result = validate_citations(text, supplied)
    assert result["membership_valid"] is False
    assert result["cited_chunk_ids"] == ["FABRICATED-1-0", "C-001-0"]
    assert result["valid_chunk_ids"] == ["C-001-0"]
    assert result["unsupported_chunk_ids"] == ["FABRICATED-1-0"]


def test_citation_validation_no_citations():
    """Verify answers with zero citations are marked as not membership-valid."""
    text = "There is no evidence available to answer."
    supplied = ["C-001-0"]
    result = validate_citations(text, supplied)
    assert result["membership_valid"] is False
    assert result["cited_chunk_ids"] == []
    assert result["valid_chunk_ids"] == []
    assert result["unsupported_chunk_ids"] == []


def test_graph_prompt_is_generic_and_guards_hypotheses():
    """Verify that GRAPH_PROMPT enforces hypothesis treatment without hardcoded benchmark terms."""
    # Must enforce hypothesis treatment
    assert "hypothes" in GRAPH_PROMPT.lower()
    assert "proven causation" in GRAPH_PROMPT.lower() or "definitive proof" in GRAPH_PROMPT.lower()
    assert "source chunks" in GRAPH_PROMPT.lower()

    # Must be generic: no benchmark-specific entities hardcoded in prompt template
    assert "bank a" not in GRAPH_PROMPT.lower()
    assert "bank b" not in GRAPH_PROMPT.lower()
    assert "release 4.2" not in GRAPH_PROMPT.lower()
    assert "gateway x" not in GRAPH_PROMPT.lower()


def test_chunk_texts_preserves_input_id_order():
    """Verify that GraphStore.chunk_texts preserves input ID order."""
    store = GraphStore.__new__(GraphStore)
    # Mock run returning rows in reverse order
    store.run = MagicMock(return_value=[
        {"id": "CHUNK-B", "text": "Text B"},
        {"id": "CHUNK-A", "text": "Text A"},
    ])
    result = store.chunk_texts(["CHUNK-A", "CHUNK-B"])
    assert [r["id"] for r in result] == ["CHUNK-A", "CHUNK-B"]


def test_baseline_answer_compatibility():
    """Verify that baseline_answer accepts k and returns a string."""
    sig = inspect.signature(baseline_answer)
    assert "k" in sig.parameters
    assert sig.parameters["k"].default == 5
