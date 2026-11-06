import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm:latest")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
MIN_GROUP_SIZE = int(os.getenv("MIN_GROUP_SIZE", "3"))
PII_SALT = os.getenv("PII_SALT", "change-me")
