import json
import ollama
from app import config


def chat(prompt: str, as_json: bool = False) -> str:
    kw = {"format": "json"} if as_json else {}
    r = ollama.chat(model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0}, **kw)
    return r["message"]["content"]


def chat_json(prompt: str, retries: int = 2) -> dict:
    for _ in range(retries + 1):
        try:
            data = json.loads(chat(prompt, as_json=True))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=config.EMBED_MODEL, input=texts)["embeddings"]
