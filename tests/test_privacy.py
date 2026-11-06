from app import privacy


def test_scrub_email_and_phone():
    out = privacy.scrub("mail a.b@x.com call +91 98765 43210")
    assert "[EMAIL]" in out and "[PHONE]" in out and "@" not in out


def test_suppress_small():
    rows = [{"complaints": 1}, {"complaints": 5}]
    assert privacy.suppress_small(rows) == [{"complaints": 5}]
