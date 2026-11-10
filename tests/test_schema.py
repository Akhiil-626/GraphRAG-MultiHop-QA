import pytest
from app.schema import is_valid, RESERVED_NAMES, RELATIONS


def test_valid_and_invalid():
    ok = {"s": "c-1", "s_type": "Complaint", "r": "AGAINST", "o": "bank a", "o_type": "Bank"}
    bad = {"s": "bank a", "s_type": "Bank", "r": "AGAINST", "o": "c-1", "o_type": "Complaint"}
    assert is_valid(ok) and not is_valid(bad)
    assert not is_valid({"r": "UNKNOWN"})


def test_reserved_entity_names_rejected():
    """Verify that reserved ontology types and tokens cannot be used as entity names."""
    # Test ontology type names
    for reserved in ["Release", "Failure", "Gateway", "Product", "Bank", "Complaint"]:
        triple_s = {"s": reserved, "s_type": "Release", "r": "DEPLOYED_ON", "o": "Gateway X", "o_type": "Gateway"}
        triple_o = {"s": "Gateway X", "s_type": "Gateway", "r": "CHANGED_IN", "o": reserved, "o_type": "Release"}
        assert not is_valid(triple_s), f"Should reject reserved subject: {reserved}"
        assert not is_valid(triple_o), f"Should reject reserved object: {reserved}"

    # Test reserved tokens
    for token in ["none", "null", "n/a", "nil", "undefined"]:
        triple = {"s": token, "s_type": "Failure", "r": "AFFECTS", "o": "Gateway X", "o_type": "Gateway"}
        assert not is_valid(triple), f"Should reject reserved token: {token}"


def test_self_loops_rejected():
    """Verify that self-loops where subject and object names match are rejected."""
    # Homonymous self-loop across types (ENG-3 anomaly)
    pseudo_loop = {
        "s": "Gateway Y",
        "s_type": "Failure",
        "r": "AFFECTS",
        "o": "Gateway Y",
        "o_type": "Gateway",
    }
    assert not is_valid(pseudo_loop)

    # Identical name and type
    same_loop = {
        "s": "Duplicate Debit",
        "s_type": "Failure",
        "r": "CAUSED_BY",
        "o": "duplicate debit",
        "o_type": "Failure",
    }
    assert not is_valid(same_loop)


def test_deployed_on_endpoints():
    """Verify DEPLOYED_ON permits both Gateway and Product targets while requiring Release source."""
    ok_gw = {"s": "Release 4.2", "s_type": "Release", "r": "DEPLOYED_ON", "o": "Gateway X", "o_type": "Gateway"}
    ok_prod = {"s": "Release 4.2", "s_type": "Release", "r": "DEPLOYED_ON", "o": "UPI", "o_type": "Product"}
    bad_src = {"s": "callback timeout", "s_type": "Failure", "r": "DEPLOYED_ON", "o": "Gateway X", "o_type": "Gateway"}
    bad_tgt = {"s": "Release 4.2", "s_type": "Release", "r": "DEPLOYED_ON", "o": "Bank A", "o_type": "Bank"}

    assert is_valid(ok_gw)
    assert is_valid(ok_prod)
    assert not is_valid(bad_src)
    assert not is_valid(bad_tgt)


def test_caused_by_endpoints():
    """Verify CAUSED_BY permits Failure target and Release target from Failure source."""
    ok_fail = {"s": "duplicate debit", "s_type": "Failure", "r": "CAUSED_BY", "o": "callback timeout", "o_type": "Failure"}
    ok_rel = {"s": "duplicate debit", "s_type": "Failure", "r": "CAUSED_BY", "o": "Release 4.2", "o_type": "Release"}
    bad_src = {"s": "Release 4.2", "s_type": "Release", "r": "CAUSED_BY", "o": "callback timeout", "o_type": "Failure"}
    bad_tgt = {"s": "duplicate debit", "s_type": "Failure", "r": "CAUSED_BY", "o": "Gateway X", "o_type": "Gateway"}

    assert is_valid(ok_fail)
    assert is_valid(ok_rel)
    assert not is_valid(bad_src)
    assert not is_valid(bad_tgt)


def test_empty_or_whitespace_entities():
    """Verify empty or whitespace-only entity names are rejected."""
    assert not is_valid({"s": "", "s_type": "Complaint", "r": "AGAINST", "o": "Bank A", "o_type": "Bank"})
    assert not is_valid({"s": "c-001", "s_type": "Complaint", "r": "AGAINST", "o": "   ", "o_type": "Bank"})
    assert not is_valid({"s": None, "s_type": "Complaint", "r": "AGAINST", "o": "Bank A", "o_type": "Bank"})
