"""Tests for backup serialization, atomic file writes, and temporal type preservation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import neo4j.time
import pytest

from scripts.backup_restore import (
    create_backup,
    decode_neo4j_values,
    encode_neo4j_values,
)


def test_neo4j_datetime_roundtrip():
    """Verify neo4j.time.DateTime round-trips with nanosecond and timezone fidelity."""
    original_dt = neo4j.time.DateTime(2026, 11, 10, 14, 30, 45, 123456000)
    encoded = encode_neo4j_values({"last_read": original_dt})

    assert isinstance(encoded["last_read"], dict)
    assert encoded["last_read"]["__neo4j_type__"] == "DateTime"

    # JSON serialization round-trip
    serialized = json.dumps(encoded)
    deserialized = json.loads(serialized)
    decoded = decode_neo4j_values(deserialized)

    assert isinstance(decoded["last_read"], neo4j.time.DateTime)
    assert decoded["last_read"] == original_dt


def test_neo4j_all_temporal_types_roundtrip():
    """Verify Date, Time, and Duration round-trip to their exact Neo4j types."""
    d = neo4j.time.Date(2026, 11, 10)
    t = neo4j.time.Time(9, 15, 30, 500000000)
    dur = neo4j.time.Duration(months=2, days=5, seconds=10, nanoseconds=5000)

    payload = {
        "event_date": d,
        "event_time": t,
        "duration": dur,
        "regular_str": "valid_string",
        "regular_int": 42,
        "regular_float": 3.14159,
        "regular_bool": True,
        "regular_null": None,
        "embedding": [0.123] * 384,
    }

    encoded = encode_neo4j_values(payload)
    serialized = json.dumps(encoded)
    deserialized = json.loads(serialized)
    decoded = decode_neo4j_values(deserialized)

    assert isinstance(decoded["event_date"], neo4j.time.Date)
    assert decoded["event_date"] == d

    assert isinstance(decoded["event_time"], neo4j.time.Time)
    assert decoded["event_time"] == t

    assert isinstance(decoded["duration"], neo4j.time.Duration)
    assert decoded["duration"] == dur

    assert decoded["regular_str"] == "valid_string"
    assert decoded["regular_int"] == 42
    assert decoded["embedding"] == [0.123] * 384


def test_decode_preserves_nested_properties():
    """Verify node and relationship properties restore temporal types when nested."""
    node_props = {
        "key": "Event:login",
        "created_at": neo4j.time.DateTime(2026, 1, 1, 0, 0, 0),
        "embedding": [0.5] * 384,
    }
    encoded = encode_neo4j_values(node_props)
    decoded = decode_neo4j_values(json.loads(json.dumps(encoded)))

    assert isinstance(decoded["created_at"], neo4j.time.DateTime)
    assert decoded["key"] == "Event:login"
    assert len(decoded["embedding"]) == 384


def test_atomic_backup_creation_success(tmp_path: Path):
    """Verify create_backup writes atomically to output_path and leaves no temp files."""
    mock_store = MagicMock()
    mock_store.run.side_effect = [
        # 1. nodes
        [
            {
                "element_id": "4:node:1",
                "labels": ["Entity", "Test"],
                "properties": {
                    "key": "Test:node",
                    "created": neo4j.time.DateTime(2026, 11, 10, 10, 0, 0),
                    "embedding": [0.1] * 384,
                },
            }
        ],
        # 2. relationships
        [
            {
                "element_id": "5:rel:1",
                "type": "TEST_REL",
                "properties": {"chunks": ["C-1"]},
                "source_labels": ["Test"],
                "source_key": "Test:node",
                "source_chunk_id": None,
                "target_labels": ["Test"],
                "target_key": "Test:node",
                "target_chunk_id": None,
            }
        ],
        # 3. constraints
        [{"name": "test_constraint"}],
        # 4. indexes
        [{"name": "test_idx", "lastRead": neo4j.time.DateTime(2026, 11, 10, 12, 0, 0)}],
    ]

    out_file = tmp_path / "backup.json"
    result = create_backup(mock_store, out_file)

    assert out_file.exists()
    assert result["metadata"]["node_count"] == 1
    assert result["metadata"]["relationship_count"] == 1
    # Check that no temporary files remain in tmp_path
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0


def test_atomic_backup_preserves_existing_on_failure(tmp_path: Path):
    """Verify that if read-back verification fails, existing backup is not destroyed."""
    existing_file = tmp_path / "backup.json"
    existing_file.write_text('{"existing": "valid_content"}', encoding="utf-8")

    # Mock store returning invalid embedding (len != 384) to trigger validation failure
    mock_store = MagicMock()
    mock_store.run.side_effect = [
        # nodes with corrupt embedding length
        [
            {
                "element_id": "4:node:1",
                "labels": ["Entity"],
                "properties": {"key": "Node:1", "embedding": [0.1] * 100},  # Invalid dim
            }
        ],
        [],  # rels
        [],  # constraints
        [],  # indexes
    ]

    with pytest.raises(ValueError, match="Read-back embedding invalid"):
        create_backup(mock_store, existing_file)

    # Existing backup must be completely untouched
    assert existing_file.exists()
    assert existing_file.read_text(encoding="utf-8") == '{"existing": "valid_content"}'

    # Temp file must be cleaned up
    temp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(temp_files) == 0
