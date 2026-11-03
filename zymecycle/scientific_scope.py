"""Bounded scientific intent and relevance contracts; no model, I/O or activity inference."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

Material = Literal["PET", "OTHER_POLYMER", "UNSPECIFIED", "AMBIGUOUS"]
Quantity = Literal["RECORDED_RESPONSE", "KINETIC_QUANTITY", "UNSPECIFIED", "AMBIGUOUS"]
Purpose = Literal["DIRECT", "APPLICABILITY", "METHOD_REQUIREMENTS", "EXPERIMENT_PLANNING"]
PLANNING_CONSTRAINTS_ISSUE = (
    "PLANNING_REQUIRES_RESEARCHER_CONSTRAINTS: candidate conditions need clarification; "
    "no established optimum or completed experimental plan is claimed."
)


class ScientificScopeError(ValueError):
    """A bounded scientific-scope explanation, never private researcher input."""


class RequestScope(TypedDict):
    material: Material
    materials: list[str]
    quantity: Quantity
    purpose: Purpose


class EvidenceScope(TypedDict):
    materials: list[str]
    quantities: list[str]
    source_names: list[str]


class ScopeAssessment(TypedDict):
    request: RequestScope
    evidence: EvidenceScope
    status: Literal["MATCH", "MISMATCH", "UNRESOLVED", "BACKGROUND_ONLY"]
    relevance_supported: bool
    background_only: bool
    issues: list[str]
    explanations: list[str]


# A bounded vocabulary, not a polymer classifier. Unknown/mixed targets remain unresolved.
_MATERIALS = {
    "PET": r"PET|poly\s*\(\s*ethylene[\s-]*terephthalate\s*\)|polyethylene[\s-]*terephthalate",
    "polypropylene": r"polypropylene|polypropene|PP",
    "polyethylene": r"polyethylene(?![\s-]*(?:terephthalate|furanoate))|HDPE|LDPE",
    "polystyrene": r"polystyrene|PS",
    "polyurethane": r"polyurethane|PUR",
    "polyamide": r"polyamide|nylon",
    "PLA": r"PLA|polylactic[\s-]+acid|poly\s*\(\s*lactic[\s-]+acid\s*\)",
    "PEF": r"PEF|polyethylene[\s-]+furanoate",
    "PCL": r"PCL|polycaprolactone",
}
_KINETICS = re.compile(
    r"\bk\s*[_-]?\s*cat\s*(?:/|over|per|÷)\s*k\s*[_-]?\s*m\b"
    r"|\bk\s*[_-]?\s*(?:cat|m)\b|\bkinetic\s+(?:parameters?|constants?|quantit\w*)\b"
    r"|\bcatalytic\s+efficiency\b|\bturnover\s+(?:number|constant)\b|\bv\s*[_-]?\s*max\b"
    r"|\bk[_\s-]*cat\s*(?:[·*]\s*)?k[_\s-]*m\s*(?:\^?\s*[-−]1|⁻¹)",
    re.I,
)
_ENDPOINT = re.compile(
    r"\b(?:recorded|endpoint|end-point)\s+(?:(?:PET|activity|product)\s+)?(?:responses?|values?|measurements?|yields?)\b"
    r"|\bproduct\s+yield\b|\bnormalized\s+activity\b",
    re.I,
)


def _negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 55) : start]
    return bool(
        re.search(
            r"\b(?:not|excluding|rather\s+than|instead\s+of|do\s+not\s+(?:infer|estimate)|not\s+asking\s+for|without\s+inferring)\s+(?:(?:on|using|any|the)\s+)?$",
            prefix,
            re.I,
        )
    )


def is_enzyme_discovery_question(question: str) -> bool:
    """Flow: Receives external question str from interpret_request(); returns bool for bounded enzyme-discovery intent, never scientific support or permission to substitute a polymer/kinetic quantity."""
    return bool(re.search(r'\b(?:enzymes?|hydrolases?|proteins?|experiments?|conditions?|responses?|activity)\b', question, re.I)
                and re.search(r'\b(?:best|top|high(?:est)?|leading|proficient|efficient|first|optim\w*)\b', question, re.I)
                and re.search(r'\bPET\b', question, re.I)
                and not _KINETICS.search(question))


def interpret_scientific_scope(question: str) -> RequestScope:
    """Flow: Receives original external question str from interpret_request(); returns RequestScope for assess_scientific_scope(), preserving unresolved intent instead of assuming a substrate or quantity.

    This is bounded clause-aware interpretation, not general semantic understanding.
    Multiple active targets remain ambiguous; negated/background mentions are not targets.
    """
    text = question.replace("ₘ", "m").replace("ₖ", "k")
    found: list[tuple[str, int]] = []
    targeted: list[str] = []
    for name, pattern in _MATERIALS.items():
        for match in re.finditer(r"(?<![\w])(?:" + pattern + r")(?![\w])", text, re.I):
            if _negated(text, match.start()):
                continue
            clause = re.split(r"[;.?!]", text[: match.start()])[-1]
            if re.match(r"\s*(?:discuss|mention|for\s+context|as\s+background)\b", clause, re.I):
                continue
            found.append((name, match.start()))
            prefix = text[max(0, match.start() - 45) : match.start()]
            if re.search(
                r"(?:\b(?:on|using|towards?|against|degrades?|degradation\s+of|hydroly[sz]es?|and|or|versus|vs)|/)\s+(?:(?:crystalline|amorphous)\s+)?$",
                prefix,
                re.I,
            ):
                targeted.append(name)
    materials = list(dict.fromkeys(targeted or [n for n, _ in sorted(found, key=lambda v: v[1])]))
    material: Material = (
        "UNSPECIFIED"
        if not materials
        else "AMBIGUOUS"
        if len(materials) > 1
        else "PET"
        if materials == ["PET"]
        else "OTHER_POLYMER"
    )
    kinetic = any(not _negated(text, m.start()) for m in _KINETICS.finditer(text))
    endpoint = bool(_ENDPOINT.search(text))
    purpose: Purpose = "DIRECT"
    if kinetic and re.search(
        r"\b(?:what|which|how)\b.*\b(?:additional\s+experiments?|measurements?\s+(?:are|would|should)|"
        r"experiments?\s+(?:are|would|should)|needed|required|estimate|estimating|design)\b",
        text,
        re.I,
    ):
        purpose = "METHOD_REQUIREMENTS"
    elif re.search(
        r"\b(?:can|could|does|do)\b.*\b(?:data|evidence|endpoint|yield)\b.*"
        r"\b(?:establish|support|show|tell|determine|infer|prove)\b"
        r"|\b(?:why|whether)\b.*\b(?:not|cannot|can't)\b"
        r"|\bdifference\s+between\b",
        text,
        re.I,
    ):
        purpose = "APPLICABILITY"
    if not kinetic and re.search(
        r"\b(?:experiment\w*|condition\w*)\b.*\b(?:recommend\w*|optim\w*|candidate\w*|plan\w*|propos\w*)\b"
        r"|\b(?:recommend\w*|optim\w*|candidate\w*|plan\w*|propos\w*)\b.*\b(?:experiment\w*|condition\w*)\b"
        r"|\bwhat\s+conditions?\s+should\s+i\s+test\b"
        r"|\b(?:replicat\w*|repeat)\b.*\bexperiment\w*\b"
        r"|\bexperiment\w*\b.*\b(?:replicat\w*|repeat)\b"
        r"|\bwhat\s+experiments?\s+should\s+(?:i|we)\s+(?:conduct|run|perform)\b"
        # Harmless equivalents of a planning request; not enzyme- or template-specific.
        r"|\bwhat\s+should\s+(?:i|we)\s+(?:test|try|measure)(?:\s+\w+){0,3}\s+next\b"
        r"|\bwhat\s+to\s+(?:test|try|measure)(?:\s+\w+){0,2}\s+next\b"
        r"|\bsuggest(?:\s+\w+){0,2}\s+(?:next|follow[\s-]?up)\s+experiments?\b",
        text,
        re.I,
    ):
        purpose = "EXPERIMENT_PLANNING"
    quantity: Quantity = (
        "AMBIGUOUS"
        if kinetic and endpoint and purpose == "DIRECT"
        else "KINETIC_QUANTITY"
        if kinetic
        else "RECORDED_RESPONSE"
        if endpoint or is_enzyme_discovery_question(question)
        else "UNSPECIFIED"
    )
    return {"material": material, "materials": materials, "quantity": quantity, "purpose": purpose}


def assess_scientific_scope(request: RequestScope, evidence: EvidenceScope) -> ScopeAssessment:
    """Flow: Receives RequestScope from interpret_scientific_scope() and executed EvidenceScope from assess_agent_scope(); returns ScopeAssessment for validate_investigation() and labelled researcher reporting.

    Relevance is distinct from measurement comparability, provenance and execution success.
    Requirements/applicability answers never supply the requested unsupported measurement.
    """
    explanations: list[str] = []
    issues: list[str] = []
    material = request["material"]
    quantity = request["quantity"]
    planning = request["purpose"] == "EXPERIMENT_PLANNING"
    material_match = bool(evidence["materials"]) and set(request["materials"]) == set(
        evidence["materials"]
    )
    if material == "OTHER_POLYMER" and not material_match:
        target = ", ".join(request["materials"])
        explanations.append(
            f"PET evidence alone does not establish {target} degradation. The requested material is {target}; "
            "selected PET observations are background only, not direct evidence for that polymer."
        )
        if request["purpose"] in ("DIRECT", "EXPERIMENT_PLANNING"):
            issues.append("SCIENTIFIC_SCOPE_MATERIAL_MISMATCH")
    elif material == "PET" and evidence["materials"] and not material_match:
        explanations.append(
            "The selected evidence does not match the requested PET material. "
            "Other-polymer measurements are background only and cannot establish PET degradation."
        )
        if request["purpose"] in ("DIRECT", "EXPERIMENT_PLANNING"):
            issues.append("SCIENTIFIC_SCOPE_MATERIAL_MISMATCH")
    if quantity in ("KINETIC_QUANTITY", "AMBIGUOUS"):
        explanations.append(
            "Recorded endpoint responses, product yields or normalized activity do not establish kcat/KM "
            "or other kinetic parameters. No kinetic constant, conversion or kinetic superiority is inferred."
        )
        if request["purpose"] == "DIRECT":
            issues.append("SCIENTIFIC_SCOPE_QUANTITY_MISMATCH")
        elif request["purpose"] == "METHOD_REQUIREMENTS" and material in ("PET", "UNSPECIFIED"):
            explanations.append(
                "HYPOTHETICAL measurement requirements: define the kinetic quantity and an appropriate "
                "interfacial model for insoluble PET; obtain calibrated time-resolved product measurements "
                "over justified substrate/enzyme levels with controlled conditions, controls, replication "
                "and uncertainty. These are conditional requirements, not fitted parameters or a validated protocol."
            )
    if material == "AMBIGUOUS":
        issues.append("SCIENTIFIC_SCOPE_AMBIGUOUS_MATERIAL")
        explanations.append(
            "Multiple requested materials remain unresolved. Clarify which polymer each comparison concerns; "
            "PET observations cannot be substituted for another polymer."
        )
    if planning:
        explanations.append(
            "Exploratory experiment planning is not a quantitative comparison. Confirm the intended "
            "substrate/PET form, available temperatures, pH bounds and measurement objective before "
            "choosing candidate conditions. A highest recorded response in a documented condition "
            "group is distinct from a HYPOTHETICAL next experiment and an experimentally established "
            "optimum; no optimum or activity superiority is established here."
        )
    elif material == "UNSPECIFIED" or quantity == "UNSPECIFIED":
        explanations.append(
            "Requested material or measurement quantity is not fully explicit. Any PET results are "
            "bounded source-background responses only; clarify the intended material and quantity "
            "before treating them as a direct comparison."
        )
    if issues:
        status: Literal["MATCH", "MISMATCH", "UNRESOLVED", "BACKGROUND_ONLY"] = (
            "UNRESOLVED" if material == "AMBIGUOUS" else "MISMATCH"
        )
    elif planning:
        status = "UNRESOLVED"
    elif explanations:
        status = "BACKGROUND_ONLY"
    elif material_match and "RECORDED_RESPONSE" in evidence["quantities"]:
        status = "MATCH"
    else:
        status = "UNRESOLVED"
        explanations.append(
            "No selected executed evidence establishes the requested material and quantity."
        )
        issues.append("SCIENTIFIC_SCOPE_EVIDENCE_UNRESOLVED")
    return {
        "request": request,
        "evidence": evidence,
        "status": status,
        "relevance_supported": status == "MATCH",
        "background_only": status != "MATCH" and not (planning and not issues),
        "issues": issues,
        "explanations": explanations,
    }


def validate_direct_dialogue_scope(question: str) -> None:
    """Flow: Receives external question str from prepare_dialogue_turn() or general_research_question(); returns None for their existing live pathway or raises ScientificScopeError before a direct unsupported comparison can launch inference."""
    scope = assess_scientific_scope(
        interpret_scientific_scope(question),
        {"materials": [], "quantities": [], "source_names": []},
    )
    if scope["request"]["purpose"] == "DIRECT" and any(
        code in scope["issues"]
        for code in (
            "SCIENTIFIC_SCOPE_MATERIAL_MISMATCH",
            "SCIENTIFIC_SCOPE_QUANTITY_MISMATCH",
            "SCIENTIFIC_SCOPE_AMBIGUOUS_MATERIAL",
        )
    ):
        raise ScientificScopeError(" ".join(scope["explanations"]))
