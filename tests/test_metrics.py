from app.evaluate import (
    contains, f1, path_hit,
    recall_at_k, precision_at_k, hit_rate_at_k,
    connected_path_exists, assess_hypothesis_calibration, norm
)
from app.qa import validate_citations


def test_basic_metrics():
    """Verify legacy contains, f1, and case-normalized path_hit behavior."""
    assert contains("It was Release 4.2.", "release 4.2")
    assert f1("release 42", "release 42") == 1.0
    facts = [{"s": "Bank A", "o": "Gateway X"}]
    # Case-insensitive path_hit
    assert path_hit(facts, ["bank a", "gateway x"])
    assert not path_hit(facts, ["bank c"])


def test_recall_at_k():
    gold = ["ENG-1-0", "ENG-2-0"]
    retrieved = ["C-001-0", "ENG-1-0", "C-002-0", "ENG-2-0"]

    # Full recall
    assert recall_at_k(retrieved, gold) == 1.0
    # Recall at K=2 (only ENG-1-0 retrieved)
    assert recall_at_k(retrieved, gold, k=2) == 0.5
    # Recall at K=1 (neither retrieved)
    assert recall_at_k(retrieved, gold, k=1) == 0.0


def test_precision_at_k():
    gold = ["ENG-1-0", "ENG-2-0"]
    retrieved = ["ENG-1-0", "C-001-0", "ENG-2-0", "C-002-0"]

    # Precision at K=2 (1 hit out of 2 budget slots)
    assert precision_at_k(retrieved, gold, k=2) == 0.5
    # Precision over entire retrieved set (2 hits out of 4)
    assert precision_at_k(retrieved, gold, k=None) == 0.5
    # Precision at K=4 (2 hits out of 4 budget slots)
    assert precision_at_k(retrieved, gold, k=4) == 0.5


def test_hit_rate_at_k():
    gold = ["ENG-1-0"]
    retrieved = ["C-001-0", "C-002-0", "ENG-1-0"]

    assert hit_rate_at_k(retrieved, gold, k=2) == 0.0
    assert hit_rate_at_k(retrieved, gold, k=3) == 1.0
    assert hit_rate_at_k(retrieved, gold) == 1.0


def test_edge_cases_empty_inputs():
    # Empty gold
    assert recall_at_k(["C-001-0"], []) == 0.0
    assert precision_at_k(["C-001-0"], []) == 0.0
    assert hit_rate_at_k(["C-001-0"], []) == 0.0

    # Empty retrieved
    assert recall_at_k([], ["ENG-1-0"]) == 0.0
    assert precision_at_k([], ["ENG-1-0"], k=5) == 0.0
    assert precision_at_k([], ["ENG-1-0"], k=None) == 0.0
    assert hit_rate_at_k([], ["ENG-1-0"]) == 0.0

    # Both empty
    assert recall_at_k([], []) == 0.0
    assert precision_at_k([], []) == 0.0
    assert hit_rate_at_k([], []) == 0.0


def test_connected_path_exists_connected_and_case_insensitive():
    facts = [
        {"s": "Bank A", "rel": "AGAINST", "o": "c-001"},
        {"s": "c-001", "rel": "VIA", "o": "Gateway X"},
        {"s": "Gateway X", "rel": "CHANGED_IN", "o": "Release 4.2"},
    ]
    # Undirected path connects "bank a" to "release 4.2"
    connected, path = connected_path_exists(facts, "bank a", "release 4.2", directed=False)
    assert connected is True
    # Notice norm("Bank A") strips article "a" to "bank", norm("c-001") strips hyphen to "c001"
    assert path == ["bank", "c001", "gateway x", "release 42"]


def test_connected_path_exists_disconnected():
    facts = [
        {"s": "Bank A", "rel": "AGAINST", "o": "c-001"},
        {"s": "Gateway Y", "rel": "CHANGED_IN", "o": "Release 5.0"},
    ]
    connected, path = connected_path_exists(facts, "bank a", "release 5.0", directed=False)
    assert connected is False
    assert path == []


def test_connected_path_exists_directed_vs_undirected():
    # Edge points from Release 4.2 to Gateway X (DEPLOYED_ON)
    facts = [{"s": "Release 4.2", "rel": "DEPLOYED_ON", "o": "Gateway X"}]

    # Undirected can traverse from Gateway X to Release 4.2
    conn_undir, _ = connected_path_exists(facts, "Gateway X", "Release 4.2", directed=False)
    assert conn_undir is True

    # Directed cannot traverse from Gateway X to Release 4.2 (edge is Release 4.2 -> Gateway X)
    conn_dir, _ = connected_path_exists(facts, "Gateway X", "Release 4.2", directed=True)
    assert conn_dir is False


def test_connected_path_rejects_eng1_invalid_endpoint_anomaly():
    """Regression test: verify ENG-1 anomaly where Failure is used with DEPLOYED_ON is rejected."""
    # Corrupted ENG-1 fact: callback timeout (Failure) DEPLOYED_ON Gateway X
    bad_facts = [
        {"s": "callback timeout", "s_type": "Failure", "rel": "DEPLOYED_ON", "o": "Gateway X", "o_type": "Gateway"}
    ]
    connected, path = connected_path_exists(bad_facts, "callback timeout", "Gateway X")
    assert connected is False
    assert path == []

    # Valid ENG-1 fact: callback timeout (Failure) AFFECTS Gateway X (Gateway)
    good_facts = [
        {"s": "callback timeout", "s_type": "Failure", "rel": "AFFECTS", "o": "Gateway X", "o_type": "Gateway"}
    ]
    connected, path = connected_path_exists(good_facts, "callback timeout", "Gateway X")
    assert connected is True
    assert path == ["callback timeout", "gateway x"]


def test_connected_path_rejects_eng3_self_loop_anomaly():
    """Regression test: verify ENG-3 pseudo-self-loop anomaly is rejected and does not bridge path."""
    bad_facts = [
        {"s": "Gateway Y", "s_type": "Failure", "rel": "AFFECTS", "o": "Gateway Y", "o_type": "Gateway"}
    ]
    # Same name s == o cannot form an edge between distinct entities
    connected, path = connected_path_exists(bad_facts, "declined transaction", "Gateway Y")
    assert connected is False
    assert path == []


def test_connected_path_rejects_invalid_relation_type():
    """Verify unknown relationship types are rejected when validate_schema=True."""
    facts = [
        {"s": "Gateway X", "rel": "FABRICATED_RELATION", "o": "Release 4.2"}
    ]
    connected, _ = connected_path_exists(facts, "Gateway X", "Release 4.2")
    assert connected is False

    # When validate_schema=False, unvalidated graph search is permitted
    connected_unval, _ = connected_path_exists(facts, "Gateway X", "Release 4.2", validate_schema=False)
    assert connected_unval is True


def test_connected_path_allowed_relations_filtering():
    """Verify allowed_relations restricts traversal to approved relations."""
    facts = [
        {"s": "duplicate debit", "rel": "CAUSED_BY", "o": "callback timeout"},
        {"s": "callback timeout", "rel": "AFFECTS", "o": "Gateway X"},
    ]
    # Path exists with all relations
    connected, path = connected_path_exists(facts, "duplicate debit", "Gateway X")
    assert connected is True

    # Path rejected if AFFECTS is excluded from allowed_relations
    connected, path = connected_path_exists(facts, "duplicate debit", "Gateway X", allowed_relations={"CAUSED_BY"})
    assert connected is False


def test_citation_membership_integration():
    answer_text = "The issue was identified in [ENG-1-0] and confirmed in [ENG-2-0]."
    supplied = ["ENG-1-0", "ENG-2-0", "C-001-0"]
    citations = validate_citations(answer_text, supplied)

    assert citations["membership_valid"] is True
    assert set(citations["valid_chunk_ids"]) == {"ENG-1-0", "ENG-2-0"}
    assert citations["unsupported_chunk_ids"] == []

    # With hallucinated citation
    bad_answer = "The issue was identified in [ENG-1-0] and [FABRICATED-9-9]."
    bad_citations = validate_citations(bad_answer, supplied)
    assert bad_citations["membership_valid"] is False
    assert bad_citations["unsupported_chunk_ids"] == ["FABRICATED-9-9"]


def test_assess_hypothesis_calibration():
    calibrated_ans = (
        "Duplicate debits are directly caused by callback timeout on Gateway X [ENG-1-0]. "
        "A potential hypothesis is that changes in Release 4.2 modified retry logic [ENG-2-0], "
        "which suggests a possible correlation, though direct causation is unproven."
    )
    res_cal = assess_hypothesis_calibration(calibrated_ans)
    assert res_cal["is_calibrated"] is True
    assert len(res_cal["hypothesis_markers_found"]) > 0
    assert len(res_cal["absolute_markers_found"]) == 0

    uncalibrated_ans = (
        "Release 4.2 definitively caused the duplicate debits seen by Bank A customers."
    )
    res_uncal = assess_hypothesis_calibration(uncalibrated_ans)
    assert res_uncal["is_calibrated"] is False
