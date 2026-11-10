ENTITY_TYPES = {"Complaint", "Bank", "Product", "Gateway", "Failure", "Release"}

# Reserved tokens and ontology type names that must never be used as entity names.
RESERVED_NAMES = {t.lower() for t in ENTITY_TYPES} | {"none", "null", "n/a", "nil", "undefined"}

# relation -> (allowed source types, allowed target types)
# Note on DEPLOYED_ON: Product is preserved alongside Gateway as an allowed deployment target
# to support broader fintech ontology semantics (e.g. releases updating customer-facing products),
# while extraction prompting and entity resolution ensure physical infrastructure (Gateway X)
# is not erroneously typed as a Product.
RELATIONS = {
    "AGAINST": ({"Complaint"}, {"Bank"}),
    "ABOUT": ({"Complaint"}, {"Product"}),
    "VIA": ({"Complaint"}, {"Gateway"}),
    "REPORTS": ({"Complaint"}, {"Failure"}),
    "AFFECTS": ({"Failure"}, {"Gateway"}),
    "CAUSED_BY": ({"Failure"}, {"Failure", "Release"}),
    "CHANGED_IN": ({"Gateway"}, {"Release"}),
    "DEPLOYED_ON": ({"Release"}, {"Gateway", "Product"}),
}


def is_valid(t: dict) -> bool:
    """Validate that a triple dict conforms strictly to ontology relations, types, and entity names.

    Rejects:
    - Unknown relations
    - Disallowed source or target types
    - Empty or whitespace-only entity names
    - Reserved ontology type names (e.g. 'release', 'failure') or tokens ('none', 'null')
    - Self-loops (where subject and object names are identical)
    """
    if not isinstance(t, dict):
        return False
    try:
        r = t.get("r")
        if r not in RELATIONS:
            return False
        src, tgt = RELATIONS[r]
        s_type, o_type = t.get("s_type"), t.get("o_type")
        if s_type not in src or o_type not in tgt:
            return False
        if s_type not in ENTITY_TYPES or o_type not in ENTITY_TYPES:
            return False
        s = str(t.get("s", "")).strip()
        o = str(t.get("o", "")).strip()
        if not s or not o:
            return False
        s_norm, o_norm = s.lower(), o.lower()
        if s_norm in RESERVED_NAMES or o_norm in RESERVED_NAMES:
            return False
        if s_norm == o_norm:
            return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def describe() -> str:
    return "\n".join(f"{r}: {'|'.join(sorted(s))} -> {'|'.join(sorted(o))}"
                     for r, (s, o) in RELATIONS.items())
