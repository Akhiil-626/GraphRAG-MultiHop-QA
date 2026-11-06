from app.evaluate import contains, f1, path_hit


def test_metrics():
    assert contains("It was Release 4.2.", "release 4.2")
    assert f1("release 42", "release 42") == 1.0
    facts = [{"s": "a", "o": "b"}]
    assert path_hit(facts, ["a", "b"]) and not path_hit(facts, ["c"])
