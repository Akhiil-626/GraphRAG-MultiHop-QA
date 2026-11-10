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

Entity Typing Guidelines:
- Technical failure conditions, timeouts, errors, declines, or exceptions MUST be typed as "Failure" (e.g. "callback timeout", "duplicate debit", "declined transaction" are Failure entities; NEVER type a timeout or failure as a Release).
- Only specific software versions or release identifiers (e.g. "Release 4.2") are "Release" entities. NEVER use generic category words like "release" or "failure" as entity names.

Rules:
1. NEVER output "none", "null", or ontology type names as entity names or subjects.
2. NEVER swap "o" and "o_type". Example: {{"o": "Gateway X", "o_type": "Gateway"}}, NEVER {{"o": "Gateway", "o_type": "Gateway X"}}.
3. NEGATIVE ASSERTION SUPPRESSION: NEVER extract relationships from negative statements or statements of non-relation (e.g. "unrelated to any recent release", "is not caused by", "does not affect"). If text says an entity is unrelated to something, do NOT generate any triple for that statement.
4. Extract ONLY facts explicitly stated in the text. Do NOT generate relations for entities not mentioned in the text.
5. Preserve subject and object identities exactly as stated in the text; never substitute a gateway or other entity name as the failure subject.
6. If no facts match the allowed relations, return {{"triples": []}}.
7. Return strictly valid JSON: {{"triples": [{{"s": "", "s_type": "", "r": "", "o": "", "o_type": ""}}]}}

Examples:

Example 1 (Complaint document):
Complaint id: c-999
Text: Customer of DemoBank paid through Gateway Alpha using the Debit Card product and experienced card error.
Output:
{{"triples": [
  {{"s": "c-999", "s_type": "Complaint", "r": "AGAINST", "o": "DemoBank", "o_type": "Bank"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "ABOUT", "o": "Debit Card", "o_type": "Product"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "VIA", "o": "Gateway Alpha", "o_type": "Gateway"}},
  {{"s": "c-999", "s_type": "Complaint", "r": "REPORTS", "o": "card error", "o_type": "Failure"}}
]}}

Example 2 (Engineering document - failure causation and gateway impact):
Complaint id: none
Text: Duplicate debit is caused by callback timeout. Callback timeout affects Gateway X.
Output:
{{"triples": [
  {{"s": "duplicate debit", "s_type": "Failure", "r": "CAUSED_BY", "o": "callback timeout", "o_type": "Failure"}},
  {{"s": "callback timeout", "s_type": "Failure", "r": "AFFECTS", "o": "Gateway X", "o_type": "Gateway"}}
]}}

Example 3 (Engineering document - negative assertion suppression):
Complaint id: none
Text: Declined transaction affects Gateway Y. It is unrelated to any recent release.
Output:
{{"triples": [
  {{"s": "declined transaction", "s_type": "Failure", "r": "AFFECTS", "o": "Gateway Y", "o_type": "Gateway"}}
]}}

Complaint id: {cid}
Text: {text}
Output:"""


def extract_triples(text: str, cid: str = "none") -> list[dict]:
    data = llm.chat_json(PROMPT.format(rels=describe(), cid=cid, text=text))
    triples = data.get("triples", [])
    return [t for t in triples if isinstance(t, dict) and is_valid(t)]
