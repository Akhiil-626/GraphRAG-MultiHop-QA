ENTITY_TYPES = {"Complaint", "Bank", "Product", "Gateway", "Failure", "Release"}

# relation -> (allowed source types, allowed target types)
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
    try:
        src, tgt = RELATIONS[t["r"]]
        return bool(t["s_type"] in src and t["o_type"] in tgt
                    and str(t["s"]).strip() and str(t["o"]).strip())
    except (KeyError, TypeError):
        return False


def describe() -> str:
    return "\n".join(f"{r}: {'|'.join(sorted(s))} -> {'|'.join(sorted(o))}"
                     for r, (s, o) in RELATIONS.items())
