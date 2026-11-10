import pytest
from unittest.mock import MagicMock, patch
from app.graph_store import GraphStore


def test_add_triple_rejects_invalid_triples_before_write():
    """Verify GraphStore.add_triple rejects invalid triples before executing any Cypher writes."""
    store = GraphStore.__new__(GraphStore)
    store.run = MagicMock()

    # Invalid: unknown relation
    bad_rel = {"s": "c-001", "s_type": "Complaint", "r": "INVALID_REL", "o": "Bank A", "o_type": "Bank"}
    with pytest.raises(ValueError, match="Invalid triple for storage"):
        store.add_triple(bad_rel, "chunk-1")

    # Invalid: mismatched source type (Bank cannot be source of AGAINST)
    bad_types = {"s": "Bank A", "s_type": "Bank", "r": "AGAINST", "o": "c-001", "o_type": "Complaint"}
    with pytest.raises(ValueError, match="Invalid triple for storage"):
        store.add_triple(bad_types, "chunk-1")

    # Invalid: reserved ontology type as entity name
    bad_reserved = {"s": "Gateway X", "s_type": "Gateway", "r": "CHANGED_IN", "o": "Release", "o_type": "Release"}
    with pytest.raises(ValueError, match="Invalid triple for storage"):
        store.add_triple(bad_reserved, "chunk-1")

    # Invalid: self-loop (same subject and object name)
    bad_loop = {"s": "Gateway Y", "s_type": "Failure", "r": "AFFECTS", "o": "Gateway Y", "o_type": "Gateway"}
    with pytest.raises(ValueError, match="Invalid triple for storage"):
        store.add_triple(bad_loop, "chunk-1")

    # Ensure no Cypher queries were run
    store.run.assert_not_called()


def test_add_triple_executes_on_valid_triple():
    """Verify GraphStore.add_triple passes valid triples to self.run."""
    store = GraphStore.__new__(GraphStore)
    store.run = MagicMock()

    valid_triple = {
        "s": "c-001",
        "s_type": "Complaint",
        "r": "AGAINST",
        "o": "Bank A",
        "o_type": "Bank"
    }
    store.add_triple(valid_triple, "chunk-1")
    store.run.assert_called_once()
    query_text = store.run.call_args[0][0]
    assert "MERGE (a)-[r:AGAINST]->(b)" in query_text


def test_expand_projects_typed_graph_facts():
    """Verify GraphStore.expand includes s_type and o_type in projected Cypher query."""
    store = GraphStore.__new__(GraphStore)
    mock_rows = [
        {
            "s": "duplicate debit",
            "s_type": "Failure",
            "rel": "CAUSED_BY",
            "o": "callback timeout",
            "o_type": "Failure",
            "chunks": ["ENG-1-0"]
        }
    ]
    store.run = MagicMock(return_value=mock_rows)

    res = store.expand(["Failure:duplicate debit"], hops=2)
    assert len(res) == 1
    assert res[0]["s_type"] == "Failure"
    assert res[0]["o_type"] == "Failure"
    assert res[0]["rel"] == "CAUSED_BY"

    query_text = store.run.call_args[0][0]
    assert "startNode(r).type AS s_type" in query_text
    assert "endNode(r).type AS o_type" in query_text


def test_embed_entities_typed_mode():
    """Verify GraphStore.embed_entities formats type-aware text when typed=True."""
    store = GraphStore.__new__(GraphStore)
    store.run = MagicMock(side_effect=[
        [{"key": "Gateway:gateway x", "name": "gateway x", "type": "Gateway"}],
        None
    ])

    with patch("app.llm.embed", return_value=[[0.1] * 384]) as mock_embed:
        store.embed_entities(typed=True)
        # Verify text passed to embed includes entity type
        mock_embed.assert_called_once_with(["Gateway gateway x"])
