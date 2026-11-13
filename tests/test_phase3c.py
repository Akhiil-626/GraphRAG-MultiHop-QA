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


def test_retrieve_ranks_graph_chunks_by_fact_support_count():
    """Verify that graph chunks supported by more distinct facts rank higher."""
    mock_store = MagicMock()
    mock_store.seed_entities.return_value = ["Entity:a"]
    # Fact 1 supports G_LOW and G_HIGH
    # Fact 2 supports G_MID and G_HIGH
    # Fact 3 supports G_HIGH
    # Total distinct facts:
    # G_HIGH: 3 facts
    # G_LOW: 1 fact (seen first)
    # G_MID: 1 fact (seen second)
    mock_store.expand.return_value = [
        {"s": "a", "s_type": "Failure", "rel": "AFFECTS", "o": "b", "o_type": "Gateway", "chunks": ["G_LOW", "G_HIGH"]},
        {"s": "b", "s_type": "Gateway", "rel": "CHANGED_IN", "o": "c", "o_type": "Release", "chunks": ["G_MID", "G_HIGH"]},
        {"s": "c", "s_type": "Release", "rel": "DEPLOYED_ON", "o": "d", "o_type": "Gateway", "chunks": ["G_HIGH"]},
    ]
    mock_store.similar_chunks.return_value = []
    mock_store.chunk_texts.side_effect = lambda ids: [{"id": cid, "text": f"Text for {cid}"} for cid in ids]

    res = retrieve(mock_store, "test question")

    # G_HIGH has 3 facts, G_MID has 1 (release), G_LOW has 1 (causal)
    # G_HIGH ranks first (higher support count), and G_MID ranks before G_LOW (release before causal)
    assert res["graph_chunk_ids"] == ["G_HIGH", "G_MID", "G_LOW"]
    retrieved_ids = [c["id"] for c in res["chunks"]]
    assert retrieved_ids == ["G_HIGH", "G_MID", "G_LOW"]


def test_retrieve_orders_release_before_causal_engineering_evidence():
    """Verify that when both release and causal engineering chunks are retrieved,

    release/change evidence is ordered before causal engineering evidence,
    and complaint context precedes them.
    """
    mock_store = MagicMock()
    mock_store.seed_entities.return_value = ["Entity:a"]
    # Mock graph facts containing:
    # 1. Complaint chunk C_CHUNK
    # 2. Causal engineering chunk CAUSAL_CHUNK (Failure AFFECTS Gateway, Failure CAUSED_BY Failure)
    # 3. Release engineering chunk RELEASE_CHUNK (Gateway CHANGED_IN Release, Release DEPLOYED_ON Gateway)
    # Both engineering chunks have identical support count (2 facts each)
    mock_store.expand.return_value = [
        {"s": "c-1", "s_type": "Complaint", "rel": "AGAINST", "o": "bank", "o_type": "Bank", "chunks": ["C_CHUNK"]},
        {"s": "c-1", "s_type": "Complaint", "rel": "VIA", "o": "gateway", "o_type": "Gateway", "chunks": ["C_CHUNK"]},
        {"s": "failure", "s_type": "Failure", "rel": "AFFECTS", "o": "gateway", "o_type": "Gateway", "chunks": ["CAUSAL_CHUNK"]},
        {"s": "failure2", "s_type": "Failure", "rel": "CAUSED_BY", "o": "failure", "o_type": "Failure", "chunks": ["CAUSAL_CHUNK"]},
        {"s": "gateway", "s_type": "Gateway", "rel": "CHANGED_IN", "o": "release", "o_type": "Release", "chunks": ["RELEASE_CHUNK"]},
        {"s": "release", "s_type": "Release", "rel": "DEPLOYED_ON", "o": "gateway", "o_type": "Gateway", "chunks": ["RELEASE_CHUNK"]},
    ]
    mock_store.similar_chunks.return_value = []
    mock_store.chunk_texts.side_effect = lambda ids: [{"id": cid, "text": f"Text for {cid}"} for cid in ids]

    res = retrieve(mock_store, "test question")

    # Expected order:
    # 1. Complaint context (C_CHUNK)
    # 2. Release/change evidence (RELEASE_CHUNK) before causal evidence (CAUSAL_CHUNK)
    expected_ids = ["C_CHUNK", "RELEASE_CHUNK", "CAUSAL_CHUNK"]
    assert res["graph_chunk_ids"] == expected_ids
    assert [c["id"] for c in res["chunks"]] == expected_ids


def test_retrieve_breaks_ties_by_first_seen_order():
    """Verify that graph chunks with equal fact support break ties by their original first-seen order."""
    mock_store = MagicMock()
    mock_store.seed_entities.return_value = ["Entity:a"]
    # All chunks have support count = 1
    mock_store.expand.return_value = [
        {"s": "a", "rel": "REL1", "o": "b", "chunks": ["G_FIRST", "G_SECOND"]},
        {"s": "b", "rel": "REL2", "o": "c", "chunks": ["G_THIRD"]},
    ]
    mock_store.similar_chunks.return_value = []
    mock_store.chunk_texts.side_effect = lambda ids: [{"id": cid, "text": f"Text for {cid}"} for cid in ids]

    res = retrieve(mock_store, "test question")

    assert res["graph_chunk_ids"] == ["G_FIRST", "G_SECOND", "G_THIRD"]
    retrieved_ids = [c["id"] for c in res["chunks"]]
    assert retrieved_ids == ["G_FIRST", "G_SECOND", "G_THIRD"]


def test_retrieve_preserves_vector_ranking_and_shared_evidence_priority():
    """Verify vector ranking and shared-evidence priority remain unchanged while graph chunks rank by support count."""
    mock_store = MagicMock()
    mock_store.seed_entities.return_value = ["Entity:a"]
    # Graph facts:
    # SHARED_2: 1 fact
    # SHARED_1: 2 facts
    # G_POPULAR: 3 facts
    # G_RARE: 1 fact
    mock_store.expand.return_value = [
        {"s": "a", "rel": "REL1", "o": "b", "chunks": ["SHARED_2", "SHARED_1", "G_POPULAR"]},
        {"s": "b", "rel": "REL2", "o": "c", "chunks": ["SHARED_1", "G_POPULAR"]},
        {"s": "c", "rel": "REL3", "o": "d", "chunks": ["G_POPULAR", "G_RARE"]},
    ]
    # Vector ranking: V_TOP, SHARED_2, SHARED_1, V_BOTTOM
    mock_store.similar_chunks.return_value = [
        {"id": "V_TOP", "text": "Vector top"},
        {"id": "SHARED_2", "text": "Shared 2"},
        {"id": "SHARED_1", "text": "Shared 1"},
        {"id": "V_BOTTOM", "text": "Vector bottom"},
    ]
    mock_store.chunk_texts.side_effect = lambda ids: [{"id": cid, "text": f"Text for {cid}"} for cid in ids]

    res = retrieve(mock_store, "test question", max_chunks=10)

    # 1. Shared chunks first, in vector rank order: SHARED_2, SHARED_1
    # 2. Vector-only chunks second, in vector rank order: V_TOP, V_BOTTOM
    # 3. Graph-only chunks third, in fact support count order: G_POPULAR, G_RARE
    expected_ids = ["SHARED_2", "SHARED_1", "V_TOP", "V_BOTTOM", "G_POPULAR", "G_RARE"]
    retrieved_ids = [c["id"] for c in res["chunks"]]
    assert retrieved_ids == expected_ids
    assert res["graph_chunk_ids"] == ["G_POPULAR", "SHARED_1", "SHARED_2", "G_RARE"]
    assert res["vector_chunk_ids"] == ["V_TOP", "SHARED_2", "SHARED_1", "V_BOTTOM"]


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


def test_citation_validation_unretrieved_real_chunk_ids():
    """Verify citation validation flags real-format chunk IDs that were not in supplied chunks."""
    text = "The issue is described in [C-001-0] and further detailed in [C-002-0]."
    supplied = ["C-001-0", "ENG-1-0"]  # C-002-0 has valid syntax but was not retrieved/supplied
    result = validate_citations(text, supplied)
    assert result["membership_valid"] is False
    assert result["cited_chunk_ids"] == ["C-001-0", "C-002-0"]
    assert result["valid_chunk_ids"] == ["C-001-0"]
    assert result["unsupported_chunk_ids"] == ["C-002-0"]


def test_citation_validation_ignores_relationship_labels():
    """Graph relationship labels must not be treated as chunk citations."""
    text = (
        "The graph shows callback timeout [AFFECTS] gateway x, "
        "and another path [VIA] gateway x [REPORTS]. "
        "The supporting evidence is [ENG-1-0]."
    )
    supplied = ["ENG-1-0"]

    result = validate_citations(text, supplied)

    assert result["membership_valid"] is True
    assert result["cited_chunk_ids"] == ["ENG-1-0"]
    assert result["valid_chunk_ids"] == ["ENG-1-0"]
    assert result["unsupported_chunk_ids"] == []


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
    assert "first sentence" in GRAPH_PROMPT.lower()
    assert "square brackets" in GRAPH_PROMPT.lower()

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


def test_qa_answer_retries_when_citations_missing(monkeypatch):
    """Verify that qa.answer triggers a retry when initial answer lacks citations."""
    mock_store = MagicMock()
    mock_store.seed_entities.return_value = ["Entity:a"]
    mock_store.expand.return_value = [{"s": "a", "rel": "REL", "o": "b", "chunks": ["ENG-1-0"]}]
    mock_store.similar_chunks.return_value = [{"id": "ENG-1-0", "text": "Gateway Alpha retry logic changed"}]
    mock_store.chunk_texts.return_value = [{"id": "ENG-1-0", "text": "Gateway Alpha retry logic changed"}]

    responses = [
        "Gateway Alpha was updated in the latest release.",  # No citations
        "Answer: Gateway Alpha was updated [ENG-1-0].\nEvidence: [ENG-1-0]",  # Valid citation
    ]
    chat_calls = []

    def mock_chat(prompt):
        chat_calls.append(prompt)
        return responses[len(chat_calls) - 1]

    from app import llm, qa
    monkeypatch.setattr(llm, "chat", mock_chat)

    res = qa.answer(mock_store, "What happened to Gateway Alpha?")

    assert len(chat_calls) == 2
    assert "Your previous response did not include required chunk citations" in chat_calls[1]
    assert res["citations"]["membership_valid"] is True
    assert res["citations"]["valid_chunk_ids"] == ["ENG-1-0"]
    assert res["answer"] == responses[1]
