"""Backup and restore utilities for Neo4j GraphRAG database.

Provides:
- Full graph snapshot serialization including properties, embeddings, constraints, and indexes.
- Read-back verification of backup archives before any database writes.
- Targeted logical rollback distinguishing targeted Cypher inverse ops from snapshot restore.
- Full snapshot restoration with explicit confirmation guard (not for ordinary rollback).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import neo4j.time

from app.graph_store import GraphStore

logger = logging.getLogger(__name__)


def encode_neo4j_values(obj: Any) -> Any:
    """Recursively encode Neo4j temporal types into JSON-serializable tagged dictionaries."""
    if isinstance(obj, neo4j.time.DateTime):
        return {"__neo4j_type__": "DateTime", "value": obj.iso_format()}
    elif isinstance(obj, neo4j.time.Date):
        return {"__neo4j_type__": "Date", "value": obj.iso_format()}
    elif isinstance(obj, neo4j.time.Time):
        return {"__neo4j_type__": "Time", "value": obj.iso_format()}
    elif isinstance(obj, neo4j.time.Duration):
        return {"__neo4j_type__": "Duration", "value": obj.iso_format()}
    elif isinstance(obj, datetime.datetime):
        return {"__neo4j_type__": "DateTime", "value": obj.isoformat()}
    elif isinstance(obj, datetime.date):
        return {"__neo4j_type__": "Date", "value": obj.isoformat()}
    elif isinstance(obj, datetime.time):
        return {"__neo4j_type__": "Time", "value": obj.isoformat()}
    elif isinstance(obj, dict):
        return {k: encode_neo4j_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_neo4j_values(v) for v in obj]
    elif isinstance(obj, tuple):
        return [encode_neo4j_values(v) for v in obj]
    return obj


def decode_neo4j_values(obj: Any) -> Any:
    """Recursively decode tagged dictionaries back into original Neo4j temporal types."""
    if isinstance(obj, dict):
        if "__neo4j_type__" in obj and "value" in obj and len(obj) == 2:
            t = obj["__neo4j_type__"]
            val = obj["value"]
            if t == "DateTime":
                return neo4j.time.DateTime.from_iso_format(val)
            elif t == "Date":
                return neo4j.time.Date.from_iso_format(val)
            elif t == "Time":
                return neo4j.time.Time.from_iso_format(val)
            elif t == "Duration":
                return neo4j.time.Duration.from_iso_format(val)
        return {k: decode_neo4j_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_neo4j_values(v) for v in obj]
    return obj


def create_backup(store: GraphStore, output_path: Path) -> dict[str, Any]:
    """Capture full database snapshot and verify read-back before any writes.
    
    Captures:
    - All nodes (labels, properties, 384-dimensional embedding vectors)
    - All relationships (type, properties, source and target node keys/IDs)
    - Database constraints and vector index configurations
    
    Verifies that the serialized JSON can be read back and matches database counts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Capture all nodes with labels and full properties (including embedding)
    nodes_records = store.run(
        """
        MATCH (n)
        RETURN elementId(n) AS element_id,
               labels(n) AS labels,
               properties(n) AS properties
        ORDER BY elementId(n)
        """
    )
    nodes_data = []
    for r in nodes_records:
        props = dict(r["properties"])
        # Ensure embedding list of floats is preserved
        if "embedding" in props and props["embedding"] is not None:
            props["embedding"] = [float(x) for x in props["embedding"]]
        nodes_data.append({
            "element_id": r["element_id"],
            "labels": list(r["labels"]),
            "properties": props,
        })

    # 2. Capture all relationships with type, properties, and endpoint keys
    rel_records = store.run(
        """
        MATCH (s)-[r]->(t)
        RETURN elementId(r) AS element_id,
               type(r) AS type,
               properties(r) AS properties,
               labels(s) AS source_labels,
               s.key AS source_key,
               s.id AS source_chunk_id,
               labels(t) AS target_labels,
               t.key AS target_key,
               t.id AS target_chunk_id
        ORDER BY elementId(r)
        """
    )
    rels_data = []
    for r in rel_records:
        rels_data.append({
            "element_id": r["element_id"],
            "type": r["type"],
            "properties": dict(r["properties"]),
            "source": {
                "labels": list(r["source_labels"]),
                "key": r["source_key"],
                "id": r["source_chunk_id"],
            },
            "target": {
                "labels": list(r["target_labels"]),
                "key": r["target_key"],
                "id": r["target_chunk_id"],
            },
        })

    # 3. Capture constraints
    try:
        constraints = store.run("SHOW CONSTRAINTS")
    except Exception as exc:
        logger.warning(f"Failed to fetch constraints: {exc}")
        constraints = []

    # 4. Capture indexes
    try:
        indexes = store.run("SHOW INDEXES")
    except Exception as exc:
        logger.warning(f"Failed to fetch indexes: {exc}")
        indexes = []

    snapshot = {
        "metadata": {
            "node_count": len(nodes_data),
            "relationship_count": len(rels_data),
            "exported_at": "pre-repair",
        },
        "nodes": nodes_data,
        "relationships": rels_data,
        "constraints": [dict(c) for c in constraints],
        "indexes": [dict(i) for i in indexes],
    }

    # Encode temporal types before JSON serialization
    encoded_snapshot = encode_neo4j_values(snapshot)

    # 5. Atomic write and read-back verification via temporary file
    temp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(encoded_snapshot, f, indent=2)

        # Read-back verification from temp file
        with open(temp_path, "r", encoding="utf-8") as f:
            raw_verified = json.load(f)
        verified_data = decode_neo4j_values(raw_verified)

        if len(verified_data["nodes"]) != len(nodes_data):
            raise ValueError(
                f"Read-back verification failed: expected {len(nodes_data)} nodes, "
                f"got {len(verified_data['nodes'])}"
            )
        if len(verified_data["relationships"]) != len(rels_data):
            raise ValueError(
                f"Read-back verification failed: expected {len(rels_data)} relationships, "
                f"got {len(verified_data['relationships'])}"
            )

        # Verify embeddings in read-back data
        for n in verified_data["nodes"]:
            emb = n["properties"].get("embedding")
            if emb is not None:
                if not isinstance(emb, list) or len(emb) != 384:
                    raise ValueError(
                        f"Read-back embedding invalid for node {n['properties'].get('key', n['element_id'])}"
                    )

        # Atomic replacement: only replace destination after successful verification
        os.replace(temp_path, output_path)

    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    logger.info(
        f"Backup successfully created and verified at {output_path} "
        f"({len(nodes_data)} nodes, {len(rels_data)} relationships)."
    )
    return verified_data


def targeted_logical_rollback(store: GraphStore) -> None:
    """Execute targeted logical rollback of repair mutations.
    
    NOTE: Clearly distinguished from snapshot restore. Targeted logical rollback
    operates on specific repair entities and relationships using inverse Cypher:
    - Removes new relationships (AFFECTS, AGAINST, MENTIONS) created during repair.
    - Removes new Bank C entity.
    - Reverts Failure:callback timeout back to Release:callback timeout.
    - Recreates deleted ghost entities and relationships.
    
    This preserves database uptime, node element IDs of untouched nodes,
    and does NOT perform destructive graph clearing.
    """
    logger.info("Starting targeted logical rollback...")

    # Step 1: Remove repair-created relationships
    store.run(
        """
        MATCH (c:Entity {key: 'Complaint:c-004'})-[r:AGAINST]->(b:Entity {key: 'Bank:bank c'})
        DELETE r
        """
    )
    store.run(
        """
        MATCH (ch:Chunk {id: 'C-004-0'})-[r:MENTIONS]->(b:Entity {key: 'Bank:bank c'})
        DELETE r
        """
    )
    store.run(
        """
        MATCH (f:Entity {key: 'Failure:declined transaction'})-[r:AFFECTS]->(gy:Entity {key: 'Gateway:gateway y'})
        DELETE r
        """
    )
    store.run(
        """
        MATCH (ch:Chunk {id: 'ENG-3-0'})-[r:MENTIONS]->(f:Entity {key: 'Failure:declined transaction'})
        DELETE r
        """
    )
    store.run(
        """
        MATCH (f:Entity {key: 'Failure:callback timeout'})-[r:AFFECTS]->(gx:Entity {key: 'Gateway:gateway x'})
        DELETE r
        """
    )

    # Step 2: Remove Bank:bank c
    store.run(
        """
        MATCH (b:Entity {key: 'Bank:bank c'})
        DELETE b
        """
    )

    # Step 3: Revert callback timeout key and type back to Release
    store.run(
        """
        MATCH (ct:Entity {key: 'Failure:callback timeout'})
        SET ct.key = 'Release:callback timeout', ct.type = 'Release'
        """
    )

    # Step 4: Re-create DEPLOYED_ON from Release:callback timeout to Gateway:gateway x
    store.run(
        """
        MATCH (ct:Entity {key: 'Release:callback timeout'}), (gx:Entity {key: 'Gateway:gateway x'})
        MERGE (ct)-[r:DEPLOYED_ON]->(gx)
        ON CREATE SET r.chunks = ['ENG-1-0']
        """
    )

    # Step 5: Restore ghost entities and original relationships
    # Restore Product:gateway x
    store.run(
        """
        MERGE (p:Entity {key: 'Product:gateway x'})
        ON CREATE SET p.name = 'gateway x', p.type = 'Product'
        """
    )
    store.run(
        """
        MATCH (ct:Entity {key: 'Release:callback timeout'}), (p:Entity {key: 'Product:gateway x'})
        MERGE (ct)-[r:DEPLOYED_ON]->(p)
        ON CREATE SET r.chunks = ['ENG-1-0']
        """
    )
    store.run(
        """
        MATCH (c:Chunk {id: 'ENG-1-0'}), (p:Entity {key: 'Product:gateway x'})
        MERGE (c)-[:MENTIONS]->(p)
        """
    )

    # Restore Failure:gateway y
    store.run(
        """
        MERGE (f:Entity {key: 'Failure:gateway y'})
        ON CREATE SET f.name = 'gateway y', f.type = 'Failure'
        """
    )
    store.run(
        """
        MATCH (f:Entity {key: 'Failure:gateway y'}), (gy:Entity {key: 'Gateway:gateway y'})
        MERGE (f)-[r:AFFECTS]->(gy)
        ON CREATE SET r.chunks = ['ENG-3-0']
        """
    )
    store.run(
        """
        MATCH (c:Chunk {id: 'ENG-3-0'}), (f:Entity {key: 'Failure:gateway y'})
        MERGE (c)-[:MENTIONS]->(f)
        """
    )

    # Restore Release:release
    store.run(
        """
        MERGE (rel:Entity {key: 'Release:release'})
        ON CREATE SET rel.name = 'release', rel.type = 'Release'
        """
    )
    store.run(
        """
        MATCH (gy:Entity {key: 'Gateway:gateway y'}), (rel:Entity {key: 'Release:release'})
        MERGE (gy)-[r:CHANGED_IN]->(rel)
        ON CREATE SET r.chunks = ['ENG-3-0']
        """
    )
    store.run(
        """
        MATCH (c:Chunk {id: 'ENG-3-0'}), (rel:Entity {key: 'Release:release'})
        MERGE (c)-[:MENTIONS]->(rel)
        """
    )

    logger.info("Targeted logical rollback completed successfully.")


def restore_from_snapshot(store: GraphStore, backup_path: Path, confirm: bool = False) -> None:
    """Exact snapshot restoration from backup file.
    
    SAFETY RESTRICTION:
    Snapshot restoration clears and rebuilds the database graph. It requires
    confirm=True and must NOT be used as an ordinary rollback operation.
    Ordinary rollback must use targeted_logical_rollback().
    """
    if not confirm:
        raise ValueError(
            "CRITICAL SAFETY GUARD: restore_from_snapshot drops all nodes and edges. "
            "It must not be used as an ordinary rollback. Explicit confirm=True is required."
        )

    backup_path = Path(backup_path)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found at {backup_path}")

    with open(backup_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = decode_neo4j_values(raw_data)

    logger.warning("Initiating full snapshot restore from backup file...")

    # Step 1: Detach delete all existing nodes
    store.run("MATCH (n) DETACH DELETE n")

    # Step 2: Recreate nodes with exact properties and embeddings
    for node in data["nodes"]:
        labels_str = ":".join(node["labels"])
        props = node["properties"]
        cypher = f"CREATE (n:{labels_str} $props)"
        store.run(cypher, props=props)

    # Step 3: Recreate relationships
    for rel in data["relationships"]:
        rel_type = rel["type"]
        props = rel["properties"]
        src = rel["source"]
        tgt = rel["target"]

        if src["key"]:
            src_match = "s.key = $src_key"
            src_param = {"src_key": src["key"]}
        else:
            src_match = "s.id = $src_id"
            src_param = {"src_id": src["id"]}

        if tgt["key"]:
            tgt_match = "t.key = $tgt_key"
            tgt_param = {"tgt_key": tgt["key"]}
        else:
            tgt_match = "t.id = $tgt_id"
            tgt_param = {"tgt_id": tgt["id"]}

        src_label = src["labels"][0] if src["labels"] else "Node"
        tgt_label = tgt["labels"][0] if tgt["labels"] else "Node"

        cypher = f"""
        MATCH (s:{src_label}), (t:{tgt_label})
        WHERE {src_match} AND {tgt_match}
        CREATE (s)-[r:{rel_type} $props]->(t)
        """
        params = {**src_param, **tgt_param, "props": props}
        store.run(cypher, **params)

    logger.info("Snapshot restore completed. Verifying restored counts...")
    curr_nodes = store.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    curr_rels = store.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

    if curr_nodes != data["metadata"]["node_count"]:
        raise ValueError(
            f"Restoration node count mismatch: expected {data['metadata']['node_count']}, "
            f"got {curr_nodes}"
        )
    if curr_rels != data["metadata"]["relationship_count"]:
        raise ValueError(
            f"Restoration rel count mismatch: expected {data['metadata']['relationship_count']}, "
            f"got {curr_rels}"
        )
    logger.info("Snapshot restoration verified successfully.")
