from app import llm
from app.schema import describe, is_valid

PROMPT = """Extract knowledge-graph triples from the text.
Allowed relations (SOURCE_TYPE -> TARGET_TYPE):
{rels}

Field definitions:
- "s": The subject entity name from the text.
  * If Complaint id is a real id (e.g. "c-001"), use it as the Complaint subject.
  * If Complaint id is "none", this is an engineering document: extract the subject from entities in the text (such as a Failure, Gateway, or Release). NEVER use "none" as an entity name or subject!
- "s_type": The ontology type of the subject. MUST match the SOURCE_TYPE allowed for relation "r". NEVER leave empty!
- "r": The relation name from the allowed relations list.
- "o": The specific object entity name from the text (e.g. "Bank A", "UPI", "Gateway X", "callback timeout"). Never use an ontology type name as an entity name.
- "o_type": The ontology type of the object. MUST match the TARGET_TYPE allowed for relation "r".

Source and Target Type Constraints:
- When Complaint id is given (e.g. "c-001"): use ONLY AGAINST (to Bank), ABOUT (to Product), VIA (to Gateway), REPORTS (to Failure). Reported failures MUST use REPORTS (target: Failure); never use ABOUT for a failure.
- When Complaint id is "none": there is NO complaint entity. Extract ONLY supported engineering relations between Failure, Gateway, Release, Product (e.g. Failure CAUSED_BY Failure/Release, Failure AFFECTS Gateway, Gateway CHANGED_IN Release, Release DEPLOYED_ON Gateway/Product). NEVER invent complaint relations (AGAINST, ABOUT, VIA, REPORTS) in engineering documents.

Rules:
1. NEVER output "none" as an entity name or subject. "none" is only metadata indicating no complaint id exists.
2. NEVER swap "o" and "o_type". Example: {{"o": "Gateway X", "o_type": "Gateway"}}, NEVER {{"o": "Gateway", "o_type": "Gateway X"}}.
3. Extract ONLY facts explicitly stated in the text. Do NOT generate relations for entities not mentioned in the text.
4. If no facts match the allowed relations, return {{"triples": []}}.
5. Return strictly valid JSON: {{"triples": [{{"s": "", "s_type": "", "r": "", "o": "", "o_type": ""}}]}}

Example:
Complaint id: c-999
Text: Customer of DemoBank paid through Gateway Alpha using the Debit Card product and experienced card error.
Output:
{{"triples": [
  {{"s": "c-999", "s_type": "Complaint", "r": "AGAINST", "o": "DemoBank", "o_type": "Bank"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "ABOUT", "o": "Debit Card", "o_type": "Product"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "VIA", "o": "Gateway Alpha", "o_type": "Gateway"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "REPORTS", "o": "card error", "o_type": "Failure"}}
]}}

Complaint id: {cid}
Text: {text}"""


def extract_triples(text: str, cid: str = "none") -> list[dict]:
    data = llm.chat_json(PROMPT.format(rels=describe(), cid=cid, text=text))
    triples = data.get("triples", [])
    return [t for t in triples if isinstance(t, dict) and is_valid(t)]
