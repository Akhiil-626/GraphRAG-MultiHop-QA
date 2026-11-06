from app import llm
from app.schema import describe, is_valid

PROMPT = """Extract knowledge-graph triples from the text.
Allowed relations (SOURCE_TYPE -> TARGET_TYPE):
{rels}
Rules: use only these relations and types. If a complaint id is given, name the Complaint entity exactly that id. Keep entity names short and singular.
Return ONLY JSON: {{"triples":[{{"s":"","s_type":"","r":"","o":"","o_type":""}}]}}
Complaint id: {cid}
Text: {text}"""


def extract_triples(text: str, cid: str = "none") -> list[dict]:
    data = llm.chat_json(PROMPT.format(rels=describe(), cid=cid, text=text))
    triples = data.get("triples", [])
    return [t for t in triples if isinstance(t, dict) and is_valid(t)]
