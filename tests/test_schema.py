from app.schema import is_valid


def test_valid_and_invalid():
    ok = {"s": "c-1", "s_type": "Complaint", "r": "AGAINST", "o": "bank a", "o_type": "Bank"}
    bad = {"s": "bank a", "s_type": "Bank", "r": "AGAINST", "o": "c-1", "o_type": "Complaint"}
    assert is_valid(ok) and not is_valid(bad)
    assert not is_valid({"r": "UNKNOWN"})
