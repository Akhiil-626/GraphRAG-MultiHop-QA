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
    # Edge points from B to A
    facts = [{"s": "Node B", "rel": "POINTS_TO", "o": "Node A"}]

    # Undirected can traverse from A to B
    conn_undir, _ = connected_path_exists(facts, "Node A", "Node B", directed=False)
    assert conn_undir is True

    # Directed cannot traverse from A to B (edge is B -> A)
    conn_dir, _ = connected_path_exists(facts, "Node A", "Node B", directed=True)
    assert conn_dir is False


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
