import pytest
from unittest.mock import patch
from app.resolve import EntityResolver, _blocked, _norm


def test_short_stopwords_do_not_block_merges():
    """Verify that short stopwords (in, to, by, of) do not block entity alias merging."""
    # 'gateway x in prod' vs 'gateway x': 'in' should be ignored
    assert not _blocked("gateway x in prod", "gateway x prod")
    assert not _blocked("payment by upi", "payment upi")
    assert not _blocked("card to bank", "card bank")


def test_meaningful_identifiers_and_abbreviations_block_merges():
    """Verify that distinct meaningful identifiers (x vs y, v1 vs v2, db) are preserved and block merging."""
    # Gateway identifiers x vs y
    assert _blocked("gateway x", "gateway y")

    # Version abbreviations v1 vs v2
    assert _blocked("release v1", "release v2")

    # Database vs API abbreviation
    assert _blocked("bank db", "bank api")

    # Numeric difference
    assert _blocked("release 4.2", "release 4.3")


def test_reserved_ontology_names_rejected_by_resolver():
    """Verify resolver raises ValueError when attempting to resolve reserved words."""
    resolver = EntityResolver()
    for reserved in ["Release", "Failure", "Gateway", "Product", "Bank", "Complaint", "none", "null"]:
        with pytest.raises(ValueError, match="reserved ontology name"):
            resolver.resolve(reserved, "Release")


def test_empty_entity_name_rejected_by_resolver():
    """Verify resolver raises ValueError for empty or whitespace-only names."""
    resolver = EntityResolver()
    with pytest.raises(ValueError, match="empty entity name"):
        resolver.resolve("   ", "Gateway")


def test_cross_type_conflict_detection():
    """Verify resolver prevents the same entity name from being registered across conflicting types."""
    # Mock llm.embed to avoid real model calls
    with patch("app.llm.embed", return_value=[[0.1] * 384]):
        resolver = EntityResolver()

        # Successfully resolve Gateway X as Gateway
        res1 = resolver.resolve("Gateway X", "Gateway")
        assert res1 == "gateway x"

        # Attempting to resolve the same name as Product must raise ValueError
        with pytest.raises(ValueError, match="Cross-type entity conflict"):
            resolver.resolve("Gateway X", "Product")

        # Resolving another entity as Product succeeds
        res2 = resolver.resolve("UPI", "Product")
        assert res2 == "upi"


def test_valid_fuzzy_resolution_same_type():
    """Verify same-type entities resolve properly to canonical node."""
    with patch("app.llm.embed", return_value=[[0.1] * 384]):
        resolver = EntityResolver()
        c1 = resolver.resolve("Bank A", "Bank")
        c2 = resolver.resolve("bank a", "Bank")
        assert c1 == c2 == "bank a"
