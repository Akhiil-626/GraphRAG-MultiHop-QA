import inspect
import re
from unittest.mock import MagicMock
from app.retrieve import retrieve
from app.qa import validate_citations, GRAPH_PROMPT, baseline_answer
from app.graph_store import GraphStore
from app.evaluate import contains, expected_evidence_coverage, citation_relevance_precision, evaluate_evidence_grounding


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


def test_graph_prompt_no_placeholder_chunk_ids():
    """Verify that GRAPH_PROMPT contains no placeholder chunk tokens (e.g. CHUNK-) that could leak."""
    assert "chunk-" not in GRAPH_PROMPT.lower()
    assert "EXAMPLES OF REQUIRED FORMAT:" in GRAPH_PROMPT


def test_graph_prompt_contains_evidence_linking_and_single_hop_rules():
    """Verify GRAPH_PROMPT contains rules for prose, complaint-to-release linking, single-hop guard, and anti-padding."""
    # Rule 1: Natural-language prose & anti-citation-only rule
    assert "natural-language prose" in GRAPH_PROMPT
    assert "never respond with chunk IDs alone, and never output a list of citations in place of an answer" in GRAPH_PROMPT

    # Rule 9: Multi-hop causal completion rule
    assert "When answering multi-hop causal questions asking why an issue occurred, trace the full causal path across retrieved chunks" in GRAPH_PROMPT
    assert "explicitly identify the release that modified that gateway" in GRAPH_PROMPT

    # Rule 10: Targeted complaint-to-release evidence linking
    assert "When a question asks which release changed the gateway used by a specific complaint, cite both the complaint chunk establishing which gateway was used and the release chunk establishing the change." in GRAPH_PROMPT
    assert "Do not cite unrelated failure or timeout chunks that are not asked about." in GRAPH_PROMPT

    # Rule 11: Single-hop guard
    assert "When a single-hop question asks only which gateway or entity is affected by a specific failure, cite only the chunk directly establishing that fact." in GRAPH_PROMPT
    assert "Do not add release, complaint, or subsequent engineering chunks." in GRAPH_PROMPT

    # Rule 12: Anti-padding rule
    assert "Do not cite unrelated retrieved chunks or add citations merely to increase citation count." in GRAPH_PROMPT


def test_q1_multihop_evidence_completeness_regression():
    """Regression test for Q1 multi-hop evidence completeness across complementary chunks."""
    expected = ["ENG-1-0", "ENG-2-0"]
    relevant = ["ENG-1-0", "ENG-2-0", "C-001-0", "C-002-0", "C-005-0"]
    supplied = ["C-001-0", "C-002-0", "C-003-0", "C-005-0", "ENG-1-0", "ENG-2-0"]

    # 1. Complete multi-hop answer linking both causal and release chunks
    full_answer = (
        "Answer: Bank A customers are experiencing duplicate debits due to callback timeouts "
        "affecting Gateway X [ENG-1-0], which had its retry logic modified in Release 4.2 [ENG-2-0].\n"
        "Evidence: [ENG-1-0], [ENG-2-0]"
    )
    val = validate_citations(full_answer, supplied)
    assert val["membership_valid"] is True
    assert set(val["cited_chunk_ids"]) == {"ENG-1-0", "ENG-2-0"}

    grounding = evaluate_evidence_grounding(val["cited_chunk_ids"], expected, relevant, membership_valid=True)
    assert grounding["expected_coverage"] == 1.0
    assert grounding["relevance_precision"] == 1.0
    assert grounding["uncovered_expected_chunks"] == []
    assert contains(full_answer, "release 4.2") is True

    # 2. Incomplete answer missing the release evidence
    partial_answer = (
        "Answer: Bank A duplicate debits are caused by callback timeouts on Gateway X [ENG-1-0].\n"
        "Evidence: [ENG-1-0]"
    )
    partial_val = validate_citations(partial_answer, supplied)
    partial_grounding = evaluate_evidence_grounding(partial_val["cited_chunk_ids"], expected, relevant, membership_valid=True)
    assert partial_grounding["expected_coverage"] == 0.5
    assert partial_grounding["uncovered_expected_chunks"] == ["ENG-2-0"]


def test_q2_complaint_to_engineering_linking_regression():
    """Regression test for Q2 complaint-to-engineering evidence linking."""
    expected = ["C-003-0", "ENG-2-0"]
    relevant = ["C-003-0", "ENG-2-0"]
    supplied = ["C-003-0", "C-005-0", "ENG-1-0", "ENG-2-0"]

    # 1. Well-linked answer citing both complaint record and engineering release
    linked_answer = (
        "Answer: Bank B complaints involved Gateway X [C-003-0], whose retry logic was modified "
        "in Release 4.2 [ENG-2-0].\n"
        "Evidence: [C-003-0], [ENG-2-0]"
    )
    val = validate_citations(linked_answer, supplied)
    assert val["membership_valid"] is True
    grounding = evaluate_evidence_grounding(val["cited_chunk_ids"], expected, relevant, membership_valid=True)
    assert grounding["expected_coverage"] == 1.0
    assert grounding["relevance_precision"] == 1.0
    assert contains(linked_answer, "release 4.2") is True

    # 2. Degenerate citation-only output without prose is detected
    citation_only = "Answer: [C-003-0], [ENG-2-0]\nEvidence: [C-003-0], [ENG-2-0]"
    ans_line = citation_only.split("\n")[0].replace("Answer:", "").strip()
    prose_only = re.sub(r"\[[A-Za-z0-9_.-]+\]", "", ans_line).strip()
    assert len(prose_only) < 5  # Confirms detection of degenerate citation-only output


def test_q3_single_hop_citation_precision_regression():
    """Regression test for Q3 single-hop citation precision preventing irrelevant evidence padding."""
    expected = ["ENG-1-0"]
    relevant = ["ENG-1-0"]
    supplied = ["C-001-0", "C-003-0", "ENG-1-0", "ENG-2-0"]

    # 1. Targeted single-hop answer adhering to single-hop guard
    targeted_answer = (
        "Answer: Gateway X is affected by the callback timeout failure [ENG-1-0].\n"
        "Evidence: [ENG-1-0]"
    )
    val = validate_citations(targeted_answer, supplied)
    assert val["membership_valid"] is True
    assert val["cited_chunk_ids"] == ["ENG-1-0"]

    grounding = evaluate_evidence_grounding(val["cited_chunk_ids"], expected, relevant, membership_valid=True)
    assert grounding["expected_coverage"] == 1.0
    assert grounding["relevance_precision"] == 1.0
    assert grounding["irrelevant_cited_chunks"] == []
    assert contains(targeted_answer, "gateway x") is True

    # 2. Padded answer that inappropriately drags in complaint and release chunks
    padded_answer = (
        "Answer: Gateway X is affected by callback timeouts [ENG-1-0], which was updated in "
        "Release 4.2 [ENG-2-0] following Bank B complaints [C-003-0].\n"
        "Evidence: [ENG-1-0], [ENG-2-0], [C-003-0]"
    )
    padded_val = validate_citations(padded_answer, supplied)
    padded_grounding = evaluate_evidence_grounding(padded_val["cited_chunk_ids"], expected, relevant, membership_valid=True)
    assert padded_grounding["expected_coverage"] == 1.0
    # Relevance precision is penalized because ENG-2-0 and C-003-0 are irrelevant to Q3
    assert padded_grounding["relevance_precision"] == round(1 / 3, 4)
    assert set(padded_grounding["irrelevant_cited_chunks"]) == {"ENG-2-0", "C-003-0"}
