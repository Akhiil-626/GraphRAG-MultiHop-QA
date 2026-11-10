"""Phase 4B2 Graph Repair Script.

Performs atomic repair of the existing GraphRAG knowledge graph in Neo4j:
- Preflight validation of current graph state and deletion targets.
- Pre-write snapshot backup with read-back verification.
- Offline embedding generation for Bank C before opening write transaction.
- Single atomic transaction executing all repair mutations with counter checks.
- Comprehensive read-only post-repair validation.
- Rollback and recovery reporting on any failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.graph_store import GraphStore
from app.llm import embed
from scripts.backup_restore import create_backup, targeted_logical_rollback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_graph")

BACKUP_PATH = Path("artifacts/backup_pre_repair.json")

# Full 64-character SHA-256 baseline digests of original node embeddings
BASELINE_EMBEDDING_HASHES: dict[str, str] = {
    # 8 Chunk Nodes
    "Chunk:C-001-0": "607bb7dbb5d123d0901a82a211ebd194d92de332b39ca881492009d202e32920",
    "Chunk:C-002-0": "f4383d6c38c7993622735d9c65edde2694e6ff780750ae430ef0f5574c58a5e9",
    "Chunk:C-003-0": "8b598f65cb3e04e7047880f4ac79eed13e5a9188ab180c45c2d05457fc15d187",
    "Chunk:C-004-0": "439d4563e1f77c783bd3a76675cfd79ca8a0132b960110e10ec1ec34be5a0c76",
    "Chunk:C-005-0": "2ee4ca416333dbde1ae955172e294e1e3d560fcf6b86b84f48c82c3100c28545",
    "Chunk:ENG-1-0": "0f03e9c3d71331774ad03f43b52c0cca0d275854324e7a3b565aec4fe56462c1",
    "Chunk:ENG-2-0": "6aafd09bd79c077f1fb86bd6b03c5019d40ba75c54341930f6fde497a69fb0be",
    "Chunk:ENG-3-0": "83c98955368c1c9170a778c646165aa0730309f3cd7886adb3f71186d7f94f49",
    # 16 Retained Original Entity Nodes
    "Entity:Bank:bank a": "1ffd75fb7b0f19ea366ef1e6f3603afb094426800645d254e0fea0caf1800703",
    "Entity:Bank:bank b": "aefdc6eb6c7f9bb3ce7039cb58336f2537999fc4778e90783039a22cf70a2bf7",
    "Entity:Complaint:c-001": "40d90cad4e5a345c541c80fadced1b483b517db4ee7469dd3bc2c73542f8b4c5",
    "Entity:Complaint:c-002": "c3014d3c77124cdd934b0e0d754fe33099c9f7aecde1b959d737f3a4c1893bb7",
    "Entity:Complaint:c-003": "367e0b72cd1deee8482f0cbc5002a3e419cf58222a919a9d7b7de17c14a2479b",
    "Entity:Complaint:c-004": "663be5cae4c553ceb82d0fa078b750ba76b56f5fde5b1accce05aa1e843745b6",
    "Entity:Complaint:c-005": "375b1505e0b6e6517d623e0ad07c972874489832622356bf10f1ba238bc08417",
    "Entity:Failure:declined transaction": "38dc5288af38de23a776bf2aea5c2be306ff5ffbd5f59468ba8c41f896e82b97",
    "Entity:Failure:duplicate debit": "d77c996a69ac14883a1e6c486acb973278c1f607c13d2ebcbbb1e200d5f7396d",
    "Entity:Gateway:gateway x": "b6e913f9647ff213aeaea590610be173f2888ed540bc81cd30d3162d25536116",
    "Entity:Gateway:gateway y": "baa7e5f3d802cb648e8e9b0e030d52bf4f3e327596e1d767284e150cc7b02272",
    "Entity:Product:card": "7dca3e18c43776389fa9f76a073211b29679d7ba85e21b582202adb6ee632f39",
    "Entity:Product:credit card": "ff99f10d63c7d9dfc957d150138acda55f617c4685a0b413f2860b9d520b1fb1",
    "Entity:Product:upi": "e57dd4f98dbc13a088fca8f6df072a62bb027235d785bd70b8ed38564a56b585",
    "Entity:Release:callback timeout": "8daae95d86d364a6a598e396e5cb334b06d3ea723799f4f771f747f7ffb78e0d",
    "Entity:Release:release 4.2": "9b68386126fd9c8fa652f801226b9057abf005cc77786d9db95074bdf14c9492",
    # 3 Deleted Original Entity Nodes (recorded for completeness)
    "Entity:Product:gateway x": "b6e913f9647ff213aeaea590610be173f2888ed540bc81cd30d3162d25536116",
    "Entity:Failure:gateway y": "baa7e5f3d802cb648e8e9b0e030d52bf4f3e327596e1d767284e150cc7b02272",
    "Entity:Release:release": "1611ef7b8c04e1540134b8fb42db6e173ac194d04694471f0e1b6c7b24dcfdb1",
}

# The 24 original nodes retained in the post-repair graph
RETAINED_NODE_KEYS = [
    "Chunk:C-001-0",
    "Chunk:C-002-0",
    "Chunk:C-003-0",
    "Chunk:C-004-0",
    "Chunk:C-005-0",
    "Chunk:ENG-1-0",
    "Chunk:ENG-2-0",
    "Chunk:ENG-3-0",
    "Entity:Bank:bank a",
    "Entity:Bank:bank b",
    "Entity:Complaint:c-001",
    "Entity:Complaint:c-002",
    "Entity:Complaint:c-003",
    "Entity:Complaint:c-004",
    "Entity:Complaint:c-005",
    "Entity:Failure:declined transaction",
    "Entity:Failure:duplicate debit",
    "Entity:Gateway:gateway x",
    "Entity:Gateway:gateway y",
    "Entity:Product:card",
    "Entity:Product:credit card",
    "Entity:Product:upi",
    "Entity:Release:callback timeout",  # Key updated in-place to Failure:callback timeout
    "Entity:Release:release 4.2",
]


def compute_vector_sha256(vector: list[float]) -> str:
    """Compute 64-character SHA-256 hex digest of a float vector normalized to 8 decimal places."""
    normalized = [round(float(x), 8) for x in vector]
    serialized = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PreflightError(Exception):
    """Raised when preflight state fails invariant checks."""


class ValidationError(Exception):
    """Raised when post-repair validation fails."""


def run_preflight_checks(store: GraphStore) -> None:
    """Confirm graph matches expected starting state before any writes."""
    logger.info("Executing preflight checks...")

    # 1. Total counts
    n_nodes = store.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    n_chunks = store.run("MATCH (n:Chunk) RETURN count(n) AS c")[0]["c"]
    n_entities = store.run("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    n_rels = store.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    n_mentions = store.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")[0]["c"]

    if n_nodes != 27 or n_chunks != 8 or n_entities != 19:
        raise PreflightError(
            f"Preflight node counts mismatch: expected (27, 8, 19), got ({n_nodes}, {n_chunks}, {n_entities})"
        )
    if n_rels != 59 or n_mentions != 33:
        raise PreflightError(
            f"Preflight rel counts mismatch: expected 59 rels (33 mentions), got {n_rels} ({n_mentions})"
        )

    # 2. Collision checks for re-keyed and new entities
    if store.run("MATCH (n:Entity {key: 'Failure:callback timeout'}) RETURN count(n) AS c")[0]["c"] > 0:
        raise PreflightError("Preflight key collision: Failure:callback timeout already exists")
    if store.run("MATCH (n:Entity {key: 'Bank:bank c'}) RETURN count(n) AS c")[0]["c"] > 0:
        raise PreflightError("Preflight key collision: Bank:bank c already exists")

    # 3. Check replacement relationships do not already exist
    rep_checks = [
        ("Failure:callback timeout", "AFFECTS", "Gateway:gateway x"),
        ("Failure:declined transaction", "AFFECTS", "Gateway:gateway y"),
        ("Complaint:c-004", "AGAINST", "Bank:bank c"),
    ]
    for src_key, rel_type, tgt_key in rep_checks:
        q = f"MATCH (s:Entity {{key: $src}})-[r:{rel_type}]->(t:Entity {{key: $tgt}}) RETURN count(r) AS c"
        if store.run(q, src=src_key, tgt=tgt_key)[0]["c"] > 0:
            raise PreflightError(f"Replacement relationship already exists: ({src_key})-[:{rel_type}]->({tgt_key})")

    # 4. Check deletion targets have exactly the expected connected relationships
    deletion_targets = [
        ("Product:gateway x", 2),
        ("Failure:gateway y", 2),
        ("Release:release", 2),
    ]
    for key, expected_rels in deletion_targets:
        res = store.run(
            "MATCH (n:Entity {key: $key})-[r]-() RETURN count(r) AS c",
            key=key,
        )
        if not res or res[0]["c"] != expected_rels:
            actual = res[0]["c"] if res else 0
            raise PreflightError(
                f"Deletion target {key} has unexpected relationship count: expected {expected_rels}, got {actual}"
            )

    logger.info("Preflight checks PASSED.")


def generate_bank_c_embedding() -> list[float]:
    """Generate 384-dimensional embedding for Bank C before opening database write transaction."""
    logger.info("Generating embedding for Bank C via all-minilm...")
    vectors = embed(["bank c"])
    if not vectors or len(vectors) != 1:
        raise RuntimeError("Failed to generate embedding for 'bank c'")

    embedding = [float(x) for x in vectors[0]]
    if len(embedding) != 384:
        raise ValueError(f"Bank C embedding dimension mismatch: expected 384, got {len(embedding)}")
    if not all(isinstance(x, float) for x in embedding):
        raise TypeError("Bank C embedding contains non-float values")

    logger.info(f"Bank C embedding generated successfully (dim={len(embedding)}).")
    return embedding


def execute_repair_transaction(store: GraphStore, bank_c_embedding: list[float]) -> None:
    """Execute all repair writes in one explicit atomic transaction with counter verification."""
    logger.info("Opening single explicit Neo4j write transaction for atomic repair...")

    with store.driver.session() as session:
        tx = session.begin_transaction()
        try:
            # Step 1: Re-key Release:callback timeout to Failure:callback timeout
            logger.info("Tx Step 1: Re-keying Release:callback timeout -> Failure:callback timeout")
            res1 = tx.run(
                """
                MATCH (ct:Entity {key: 'Release:callback timeout'})
                SET ct.key = 'Failure:callback timeout',
                    ct.type = 'Failure'
                """
            )
            c1 = res1.consume().counters
            if c1.properties_set != 2 or c1.labels_added != 0 or c1.labels_removed != 0:
                raise RuntimeError(f"Step 1 failed: unexpected counters {c1}")

            # Step 2: Delete DEPLOYED_ON from callback timeout to Gateway:gateway x
            logger.info("Tx Step 2: Deleting DEPLOYED_ON between callback timeout and Gateway:gateway x")
            res2 = tx.run(
                """
                MATCH (ct:Entity {key: 'Failure:callback timeout'})-[r:DEPLOYED_ON]->(gx:Entity {key: 'Gateway:gateway x'})
                DELETE r
                """
            )
            c2 = res2.consume().counters
            if c2.relationships_deleted != 1:
                raise RuntimeError(f"Step 2 failed: expected 1 rel deleted, got {c2.relationships_deleted}")

            # Step 3: Delete Product:gateway x and its 2 relationships
            logger.info("Tx Step 3: Deleting ghost entity Product:gateway x")
            res3 = tx.run(
                """
                MATCH (p:Entity {key: 'Product:gateway x'})
                DETACH DELETE p
                """
            )
            c3 = res3.consume().counters
            if c3.nodes_deleted != 1 or c3.relationships_deleted != 2:
                raise RuntimeError(
                    f"Step 3 failed: expected 1 node, 2 rels deleted, got {c3.nodes_deleted} nodes, {c3.relationships_deleted} rels"
                )

            # Step 4: Create (Failure:callback timeout)-[:AFFECTS {chunks: ['ENG-1-0']}]->(Gateway:gateway x)
            logger.info("Tx Step 4: Creating AFFECTS from callback timeout to Gateway:gateway x")
            res4 = tx.run(
                """
                MATCH (ct:Entity {key: 'Failure:callback timeout'}), (gx:Entity {key: 'Gateway:gateway x'})
                CREATE (ct)-[r:AFFECTS {chunks: ['ENG-1-0']}]->(gx)
                """
            )
            c4 = res4.consume().counters
            if c4.relationships_created != 1:
                raise RuntimeError(f"Step 4 failed: expected 1 rel created, got {c4.relationships_created}")

            # Step 5: Delete ghost entity Failure:gateway y and its 2 relationships
            logger.info("Tx Step 5: Deleting ghost entity Failure:gateway y")
            res5 = tx.run(
                """
                MATCH (f:Entity {key: 'Failure:gateway y'})
                DETACH DELETE f
                """
            )
            c5 = res5.consume().counters
            if c5.nodes_deleted != 1 or c5.relationships_deleted != 2:
                raise RuntimeError(
                    f"Step 5 failed: expected 1 node, 2 rels deleted, got {c5.nodes_deleted} nodes, {c5.relationships_deleted} rels"
                )

            # Step 6: Delete hallucinated entity Release:release and its 2 relationships
            logger.info("Tx Step 6: Deleting entity Release:release")
            res6 = tx.run(
                """
                MATCH (rel:Entity {key: 'Release:release'})
                DETACH DELETE rel
                """
            )
            c6 = res6.consume().counters
            if c6.nodes_deleted != 1 or c6.relationships_deleted != 2:
                raise RuntimeError(
                    f"Step 6 failed: expected 1 node, 2 rels deleted, got {c6.nodes_deleted} nodes, {c6.relationships_deleted} rels"
                )

            # Step 7: Create (Failure:declined transaction)-[:AFFECTS {chunks: ['ENG-3-0']}]->(Gateway:gateway y)
            logger.info("Tx Step 7: Creating AFFECTS from declined transaction to Gateway:gateway y")
            res7 = tx.run(
                """
                MATCH (dt:Entity {key: 'Failure:declined transaction'}), (gy:Entity {key: 'Gateway:gateway y'})
                CREATE (dt)-[r:AFFECTS {chunks: ['ENG-3-0']}]->(gy)
                """
            )
            c7 = res7.consume().counters
            if c7.relationships_created != 1:
                raise RuntimeError(f"Step 7 failed: expected 1 rel created, got {c7.relationships_created}")

            # Step 8: Create MENTIONS relationship from ENG-3-0 to Failure:declined transaction
            logger.info("Tx Step 8: Creating MENTIONS from ENG-3-0 to declined transaction")
            res8 = tx.run(
                """
                MATCH (c:Chunk {id: 'ENG-3-0'}), (dt:Entity {key: 'Failure:declined transaction'})
                CREATE (c)-[:MENTIONS]->(dt)
                """
            )
            c8 = res8.consume().counters
            if c8.relationships_created != 1:
                raise RuntimeError(f"Step 8 failed: expected 1 rel created, got {c8.relationships_created}")

            # Step 9: Create Bank:bank c with 384d embedding
            logger.info("Tx Step 9: Creating node Bank:bank c")
            res9 = tx.run(
                """
                CREATE (b:Entity {
                    name: 'bank c',
                    type: 'Bank',
                    key: 'Bank:bank c',
                    embedding: $embedding
                })
                """,
                {"embedding": bank_c_embedding},
            )
            c9 = res9.consume().counters
            if c9.nodes_created != 1 or c9.properties_set != 4 or c9.labels_added != 1:
                raise RuntimeError(
                    f"Step 9 failed: expected 1 node created, 4 props set, 1 label added, got {c9.nodes_created} nodes, {c9.properties_set} props, {c9.labels_added} labels"
                )

            # Step 10: Create (Complaint:c-004)-[:AGAINST {chunks: ['C-004-0']}]->(Bank:bank c)
            logger.info("Tx Step 10: Creating AGAINST from Complaint:c-004 to Bank:bank c")
            res10 = tx.run(
                """
                MATCH (comp:Entity {key: 'Complaint:c-004'}), (b:Entity {key: 'Bank:bank c'})
                CREATE (comp)-[r:AGAINST {chunks: ['C-004-0']}]->(b)
                """
            )
            c10 = res10.consume().counters
            if c10.relationships_created != 1:
                raise RuntimeError(f"Step 10 failed: expected 1 rel created, got {c10.relationships_created}")

            # Step 11: Create MENTIONS relationship from C-004-0 to Bank:bank c
            logger.info("Tx Step 11: Creating MENTIONS from C-004-0 to Bank:bank c")
            res11 = tx.run(
                """
                MATCH (c:Chunk {id: 'C-004-0'}), (b:Entity {key: 'Bank:bank c'})
                CREATE (c)-[:MENTIONS]->(b)
                """
            )
            c11 = res11.consume().counters
            if c11.relationships_created != 1:
                raise RuntimeError(f"Step 11 failed: expected 1 rel created, got {c11.relationships_created}")

            # Commit transaction atomically
            tx.commit()
            logger.info("All 11 repair steps committed atomically in single transaction.")

        except Exception as exc:
            logger.error(f"Repair transaction failed; initiating automatic rollback: {exc}")
            try:
                tx.rollback()
                logger.info("Transaction successfully rolled back. Graph unchanged.")
            except Exception as rb_exc:
                logger.critical(f"Transaction rollback error: {rb_exc}")
            raise RuntimeError(f"Atomic repair transaction aborted: {exc}") from exc


def run_post_repair_validation(store: GraphStore) -> None:
    """Execute strictly read-only post-repair validation.
    
    If any check fails, logs the error and recovery procedures and raises ValidationError.
    """
    logger.info("Executing post-repair read-only validation...")

    try:
        # 1. Actual self-loops
        self_loops = store.run("MATCH (n)-[r]->(n) RETURN count(r) AS c")[0]["c"]
        if self_loops != 0:
            raise ValidationError(f"Found {self_loops} actual self-loop relationships (expected 0)")

        # 2. ENG-3 pseudo-self-loop anomaly
        eng3_pseudo = store.run(
            "MATCH (f:Entity {name: 'gateway y', type: 'Failure'}) RETURN count(f) AS c"
        )[0]["c"]
        if eng3_pseudo != 0:
            raise ValidationError(f"Found {eng3_pseudo} ghost Failure:gateway y entities (expected 0)")

        # 3. Overall counts
        n_nodes = store.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
        n_chunks = store.run("MATCH (n:Chunk) RETURN count(n) AS c")[0]["c"]
        n_entities = store.run("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
        n_rels = store.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        n_mentions = store.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")[0]["c"]
        n_semantic = n_rels - n_mentions

        if n_nodes != 25 or n_chunks != 8 or n_entities != 17:
            raise ValidationError(
                f"Node counts mismatch: expected 25 total (8 chunks, 17 entities), got {n_nodes} ({n_chunks}, {n_entities})"
            )
        if n_rels != 57 or n_mentions != 32 or n_semantic != 25:
            raise ValidationError(
                f"Rel counts mismatch: expected 57 total (32 mentions, 25 semantic), got {n_rels} ({n_mentions}, {n_semantic})"
            )

        # 4. Ghost entities absent
        ghost_count = store.run(
            """
            MATCH (n:Entity)
            WHERE n.key IN ['Product:gateway x', 'Failure:gateway y', 'Release:release']
            RETURN count(n) AS c
            """
        )[0]["c"]
        if ghost_count != 0:
            raise ValidationError(f"Found {ghost_count} ghost entities still in graph")

        # 5. Original chunk preservation
        chunks_res = store.run("MATCH (c:Chunk) RETURN c.id AS id ORDER BY c.id")
        chunk_ids = [r["id"] for r in chunks_res]
        expected_chunk_ids = ["C-001-0", "C-002-0", "C-003-0", "C-004-0", "C-005-0", "ENG-1-0", "ENG-2-0", "ENG-3-0"]
        if chunk_ids != expected_chunk_ids:
            raise ValidationError(f"Chunk IDs mismatch: expected {expected_chunk_ids}, got {chunk_ids}")

        # 6. Repaired relationships and provenance
        # 6a. callback timeout AFFECTS gateway x with chunks: ['ENG-1-0']
        ct_aff = store.run(
            """
            MATCH (ct:Entity {key: 'Failure:callback timeout'})-[r:AFFECTS]->(gx:Entity {key: 'Gateway:gateway x'})
            RETURN r.chunks AS chunks
            """
        )
        if not ct_aff or ct_aff[0]["chunks"] != ["ENG-1-0"]:
            raise ValidationError("Repaired relationship callback timeout AFFECTS gateway x missing or invalid provenance")

        # 6b. declined transaction AFFECTS gateway y with chunks: ['ENG-3-0']
        dt_aff = store.run(
            """
            MATCH (dt:Entity {key: 'Failure:declined transaction'})-[r:AFFECTS]->(gy:Entity {key: 'Gateway:gateway y'})
            RETURN r.chunks AS chunks
            """
        )
        if not dt_aff or dt_aff[0]["chunks"] != ["ENG-3-0"]:
            raise ValidationError("Repaired relationship declined transaction AFFECTS gateway y missing or invalid provenance")

        # 6c. c-004 AGAINST bank c with chunks: ['C-004-0']
        c4_ag = store.run(
            """
            MATCH (comp:Entity {key: 'Complaint:c-004'})-[r:AGAINST]->(b:Entity {key: 'Bank:bank c'})
            RETURN r.chunks AS chunks
            """
        )
        if not c4_ag or c4_ag[0]["chunks"] != ["C-004-0"]:
            raise ValidationError("Repaired relationship Complaint:c-004 AGAINST Bank:bank c missing or invalid provenance")

        # 6d. ENG-3-0 MENTIONS declined transaction
        m_dt = store.run(
            """
            MATCH (c:Chunk {id: 'ENG-3-0'})-[:MENTIONS]->(dt:Entity {key: 'Failure:declined transaction'})
            RETURN count(dt) AS count
            """
        )[0]["count"]
        if m_dt != 1:
            raise ValidationError("Missing MENTIONS from ENG-3-0 to Failure:declined transaction")

        # 6e. C-004-0 MENTIONS bank c
        m_bc = store.run(
            """
            MATCH (c:Chunk {id: 'C-004-0'})-[:MENTIONS]->(b:Entity {key: 'Bank:bank c'})
            RETURN count(b) AS count
            """
        )[0]["count"]
        if m_bc != 1:
            raise ValidationError("Missing MENTIONS from C-004-0 to Bank:bank c")

        # 7. Embedding verification for all 24 retained nodes
        logger.info("Verifying SHA-256 digests for all 24 retained original node embeddings...")
        nodes_res = store.run(
            """
            MATCH (n)
            RETURN labels(n) AS labels, n.id AS id, n.key AS key, n.embedding AS embedding
            """
        )
        checked_nodes = 0
        for r in nodes_res:
            labels = r["labels"]
            emb = r["embedding"]
            if "Chunk" in labels:
                node_id = f"Chunk:{r['id']}"
            elif "Entity" in labels:
                # Handle in-place key update for callback timeout
                key = r["key"]
                if key == "Failure:callback timeout":
                    node_id = "Entity:Release:callback timeout"
                else:
                    node_id = f"Entity:{key}"
            else:
                continue

            if node_id in BASELINE_EMBEDDING_HASHES and node_id in RETAINED_NODE_KEYS:
                expected_hash = BASELINE_EMBEDDING_HASHES[node_id]
                actual_hash = compute_vector_sha256(emb)
                if actual_hash != expected_hash:
                    raise ValidationError(
                        f"Embedding SHA-256 mismatch for {node_id}: expected {expected_hash}, got {actual_hash}"
                    )
                checked_nodes += 1

        if checked_nodes != 24:
            raise ValidationError(
                f"Expected to verify 24 retained original node embeddings, but checked {checked_nodes}"
            )
        logger.info(f"All 24 retained original node embeddings verified against 64-char SHA-256 digests.")

        # 8. New Bank C embedding check
        bank_c_emb = store.run(
            "MATCH (b:Entity {key: 'Bank:bank c'}) RETURN b.embedding AS emb"
        )[0]["emb"]
        if not isinstance(bank_c_emb, list) or len(bank_c_emb) != 384:
            raise ValidationError(
                f"Bank C embedding dimension invalid: expected 384, got {len(bank_c_emb) if isinstance(bank_c_emb, list) else type(bank_c_emb)}"
            )

        # 9. Vector indexes health
        indexes = store.run("SHOW INDEXES")
        chunk_idx = next((i for i in indexes if i.get("name") == "chunk_idx"), None)
        entity_idx = next((i for i in indexes if i.get("name") == "entity_idx"), None)
        if not chunk_idx or chunk_idx.get("state") != "ONLINE":
            raise ValidationError(f"Vector index chunk_idx not ONLINE: {chunk_idx}")
        if not entity_idx or entity_idx.get("state") != "ONLINE":
            raise ValidationError(f"Vector index entity_idx not ONLINE: {entity_idx}")

        logger.info("Post-repair validation PASSED completely.")

    except Exception as exc:
        logger.critical(f"Post-repair validation FAILED: {exc}")
        logger.critical(
            "RECOVERY PROCEDURE: To revert changes, run targeted logical rollback via: "
            "python -c 'from app.graph_store import GraphStore; from scripts.backup_restore import targeted_logical_rollback; store=GraphStore(); targeted_logical_rollback(store); store.close()'\n"
            f"Or restore full snapshot from backup archive at {BACKUP_PATH}."
        )
        raise


def repair_graph() -> None:
    """Main workflow coordinating backup, preflight, repair, and validation."""
    logger.info("Initializing GraphStore...")
    store = GraphStore()
    try:
        # Step 1: Preflight checks (read-only)
        run_preflight_checks(store)

        # Step 2: Backup and read-back verification (read-only)
        create_backup(store, BACKUP_PATH)

        # Step 3: Generate Bank C embedding offline before opening write transaction
        bank_c_emb = generate_bank_c_embedding()

        # Step 4: Execute single atomic write transaction
        execute_repair_transaction(store, bank_c_emb)

        # Step 5: Post-repair read-only validation
        run_post_repair_validation(store)

        logger.info("Phase 4B2 graph repair workflow completed successfully.")
    finally:
        store.close()


if __name__ == "__main__":
    repair_graph()
