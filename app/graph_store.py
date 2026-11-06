from neo4j import GraphDatabase
from app import config, llm
from app.schema import RELATIONS


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def run(self, query: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, **params)]

    def setup(self):
        self.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
        self.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        for label in ("Entity", "Chunk"):
            self.run(
                f"CREATE VECTOR INDEX {label.lower()}_idx IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.embedding) OPTIONS {{indexConfig: "
                f"{{`vector.dimensions`: {config.EMBED_DIM}, "
                f"`vector.similarity_function`: 'cosine'}}}}")

    def clear(self):
        self.run("MATCH (n) DETACH DELETE n")

    def add_chunk(self, cid: str, text: str):
        self.run("MERGE (c:Chunk {id:$id}) SET c.text=$t "
                 "WITH c CALL db.create.setNodeVectorProperty(c,'embedding',$v)",
                 id=cid, t=text, v=llm.embed([text])[0])

    def add_triple(self, t: dict, chunk_id: str):
        rel = t["r"]
        if rel not in RELATIONS:  # relation is interpolated into Cypher: whitelist only
            raise ValueError(f"invalid relation {rel}")
        self.run(
            f"""MERGE (a:Entity {{key:$ak}}) ON CREATE SET a.name=$an, a.type=$at
                MERGE (b:Entity {{key:$bk}}) ON CREATE SET b.name=$bn, b.type=$bt
                MERGE (a)-[r:{rel}]->(b)
                  ON CREATE SET r.chunks=[$cid]
                  ON MATCH SET r.chunks = CASE WHEN $cid IN r.chunks
                                          THEN r.chunks ELSE r.chunks + $cid END
                WITH a, b MATCH (c:Chunk {{id:$cid}})
                MERGE (c)-[:MENTIONS]->(a) MERGE (c)-[:MENTIONS]->(b)""",
            ak=f"{t['s_type']}:{t['s']}", an=t["s"], at=t["s_type"],
            bk=f"{t['o_type']}:{t['o']}", bn=t["o"], bt=t["o_type"], cid=chunk_id)

    def embed_entities(self):
        rows = self.run("MATCH (e:Entity) WHERE e.embedding IS NULL "
                        "RETURN e.key AS key, e.name AS name")
        if not rows:
            return
        for r, v in zip(rows, llm.embed([r["name"] for r in rows])):
            self.run("MATCH (e:Entity {key:$k}) "
                     "CALL db.create.setNodeVectorProperty(e,'embedding',$v)",
                     k=r["key"], v=v)

    def seed_entities(self, qvec, k: int = 3) -> list[str]:
        rows = self.run("CALL db.index.vector.queryNodes('entity_idx',$k,$v) "
                        "YIELD node RETURN node.key AS key", k=k, v=qvec)
        return [r["key"] for r in rows]

    def similar_chunks(self, qvec, k: int = 3) -> list[dict]:
        return self.run("CALL db.index.vector.queryNodes('chunk_idx',$k,$v) "
                        "YIELD node, score RETURN node.id AS id, node.text AS text, score",
                        k=k, v=qvec)

    def expand(self, keys: list[str], hops: int = 2, limit: int = 60) -> list[dict]:
        hops = int(hops)
        return self.run(
            f"""MATCH (s:Entity) WHERE s.key IN $keys
                MATCH p=(s)-[*1..{hops}]-(:Entity)
                WHERE all(n IN nodes(p) WHERE n:Entity)
                UNWIND relationships(p) AS r WITH DISTINCT r
                RETURN startNode(r).name AS s, type(r) AS rel,
                       endNode(r).name AS o, r.chunks AS chunks LIMIT $limit""",
            keys=keys, limit=limit)

    def chunk_texts(self, ids: list[str]) -> list[dict]:
        return self.run("MATCH (c:Chunk) WHERE c.id IN $ids "
                        "RETURN c.id AS id, c.text AS text", ids=ids)

    def shared_gateway_report(self) -> list[dict]:
        return self.run("""
            MATCH (c:Entity {type:'Complaint'})-[:VIA]->(g:Entity {type:'Gateway'})
            MATCH (c)-[:AGAINST]->(b:Entity {type:'Bank'})
            MATCH (c)-[:REPORTS]->(f:Entity {type:'Failure'})
            RETURN g.name AS gateway, f.name AS failure,
                   count(DISTINCT b) AS banks, count(DISTINCT c) AS complaints
            ORDER BY complaints DESC""")
