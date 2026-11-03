"""Fail-closed, extractive researcher communication over selected canonical evidence."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

from .advisor_types import FeasibilityAssessment
from .agent_types import Fact, ResearchReport
from .contracts import DISCOVERY_SCOPE_LIMIT, Condition
from .privacy import tool_privacy
from .scientific_scope import PLANNING_CONSTRAINTS_ISSUE

SECTIONS = ("evidence", "uncertainty", "next_action", "rationale", "controls", "publications")
FIELDS = {
    "evidence": {"statement", "measurement_summary"},
    "uncertainty": {"statement"},
    "next_action": {"proposed_action"},
    "rationale": {"expected_information"},
    "controls": {"controlled_variables", "required_measurements"},
    "publications": {"publication"},
}
# These contain bounded record JSON or aggregate dictionaries, not readable explanation sentences.
STRUCTURED_PREFIXES = (
    "PAZy publication-derived assay:",
    "Multi-study aggregate coverage:",
    "Cross-source comparability:",
)


class ProposalContext(TypedDict):
    priority: int
    prerequisites: list[str]
    feasibility: FeasibilityAssessment
    target_conditions: list[Condition]


class CitedStatement(TypedDict):
    fact_id: str
    field: str
    text: str
    evidence_category: str | None
    sources: list[dict[str, str | None]]
    proposal_context: ProposalContext | None


class Communication(TypedDict):
    status: Literal["VERIFIED_EXTRACTIVE", "VERIFIED_BACKGROUND_EXPLANATION", "DETERMINISTIC_FALLBACK"]
    reason_code: str | None
    sections: dict[str, list[CitedStatement]]
    persistence: Literal["TRANSIENT"]


def _proposal(fact: Fact) -> dict | None:
    if fact["kind"] != "PROPOSAL" or fact["evidence_category"] != "HYPOTHETICAL":
        return None
    prefix = "PROPOSED (not observed): "
    if not fact["text"].startswith(prefix):
        return None
    value = json.loads(fact["text"][len(prefix) :])
    return value if isinstance(value, dict) else None


def _sources(fact: Fact) -> list[dict[str, str | None]]:
    # No internal locator, live API identifier or retrieved text is added to citation metadata.
    return [{k: s[k] for k in ("name", "url", "doi")} for s in fact["provenance"]]


def _permitted(fact: Fact, section: str, field: str) -> list[str]:
    if field not in FIELDS[section]:
        return []
    if field == "publication":
        return list(dict.fromkeys(s["doi"] for s in fact["provenance"] if s["doi"]))
    if field == 'measurement_summary':
        prefix = 'PAZy publication-derived assay: '
        if fact['kind'] != 'EVIDENCE' or fact['evidence_category'] != 'MEASURED' or not fact['text'].startswith(prefix):
            return []
        value = json.loads(fact['text'][len(prefix):]).get('measurement_summary')
        return [value] if isinstance(value, str) else []
    if field == "statement":
        kind = ("EVIDENCE",) if section == "evidence" else ("LIMITATION", "UNANSWERED")
        return (
            [fact["text"]]
            if fact["kind"] in kind and not fact["text"].startswith(STRUCTURED_PREFIXES)
            else []
        )
    proposal = _proposal(fact)
    if proposal is None:
        return []
    value = proposal[field]
    return ["; ".join(value)] if isinstance(value, list) else [value]


def canonical_explanation(facts: list[Fact]) -> dict[str, list[dict[str, str]]]:
    """Flow: Receives list[Fact] from validate_explanation(); returns six exact cited-reference arrays as dict[str, list[dict[str, str]]] for validate_explanation(), without generating claims or selecting evidence IDs."""
    sections: dict[str, list[dict[str, str]]] = {key: [] for key in SECTIONS}

    def add(section: str, fact: Fact, field: str, text: str) -> None:
        sections[section].append({"fact_id": fact["id"], "field": field, "text": text})

    publications: set[str] = set()
    proposals = []
    for fact in facts:
        for field in ("statement", "measurement_summary"):
            for text in _permitted(fact, "evidence", field):
                add("evidence", fact, field, text)
        for text in _permitted(fact, "uncertainty", "statement"):
            add("uncertainty", fact, "statement", text)
        proposal = _proposal(fact)
        if proposal is not None:
            proposals.append((fact, proposal))
        for doi in _permitted(fact, "publications", "publication"):
            if doi not in publications:
                add("publications", fact, "publication", doi)
                publications.add(doi)
    if proposals:
        first, proposal = min(proposals, key=lambda item: item[1]["priority"])
        add("next_action", first, "proposed_action", proposal["proposed_action"])
        add("rationale", first, "expected_information", proposal["expected_information"])
        for fact, proposal in proposals:
            if proposal["category"] == "PROPOSED_EXPERIMENT":
                for field in ("controlled_variables", "required_measurements"):
                    add("controls", fact, field, "; ".join(proposal[field]))
    return sections


def _recovered_request_errors(report: ResearchReport) -> bool:
    """Flow: Receives native ResearchReport from validate_explanation(); returns bool for validate_explanation() only when failed request syntax/lineage attempts have matching valid executed tools, never accepting failed scientific results."""
    trace = report['trace']
    for index, event in enumerate(trace):
        if event['status'] != 'ERROR':
            continue
        code = event.get('error_code')
        if code not in ('ARGUMENT_SCHEMA', 'REFERENCE_LINEAGE', 'DUPLICATE_SCIENTIFIC_CALL'):
            return False
        candidates = trace[:index] if code == 'DUPLICATE_SCIENTIFIC_CALL' else trace[index + 1:]
        if not any(e['name'] == event['name'] and e['status'] != 'ERROR' for e in candidates):
            return False
    return True


def validate_explanation(candidate: object, report: ResearchReport) -> Communication:
    """Validate an extractive model explanation against canonical facts (fail-closed).

    Purpose: allows researcher-facing prose only as verbatim copies of
        validated fact text/proposal fields, cited by fact ID across
        exactly six sections (evidence, uncertainty, next_action,
        rationale, controls, publications). Any paraphrase, uncited
        claim, oversized payload, incomplete grounding, or
        live-protein-metadata citation yields a DETERMINISTIC_FALLBACK
        with a reason code — the unverified draft is never retained.
    Args:
        candidate: model explanation object (or None when absent).
        report: validated ResearchReport from run_investigation().
    Returns:
        Communication with status VERIFIED_EXTRACTIVE (cited sections)
            or DETERMINISTIC_FALLBACK (reason_code set, sections empty);
            always TRANSIENT persistence.
    Limitations:
        Citations verify copying, not truth; the underlying facts carry
            the scientific weight.
    Flow: Receives model explanation object and validated ResearchReport from run_investigation(); returns transient Communication for present_investigation() and render_communication()."""

    def fallback(code: str) -> Communication:
        # Never retain the unverified draft, exception detail or provider content.
        return {
            "status": "DETERMINISTIC_FALLBACK",
            "reason_code": code,
            "sections": {},
            "persistence": "TRANSIENT",
        }

    if candidate is None:
        return fallback("NOT_REQUESTED_OR_ABSENT")
    scope = report['validation'].get('scientific_scope')
    # A source-cited refusal may explain a rejected question. It cannot make the
    # comparison grounded, change its scope status or turn failure into support.
    background = bool(candidate == {'render':'CANONICAL_SELECTED'} and scope and scope['background_only']
        and report.get('synthesis',{}).get('reason_code') in (
            'SCIENTIFIC_SCOPE_MATERIAL_MISMATCH','SCIENTIFIC_SCOPE_QUANTITY_MISMATCH',
            'SCIENTIFIC_SCOPE_AMBIGUOUS_MATERIAL','SCIENTIFIC_SCOPE_EVIDENCE_UNRESOLVED')
        and _recovered_request_errors(report)
        and all(issue in scope['issues'] or issue == ' '.join(scope['explanations'])
                or issue.startswith(('Integrated advice is partial:', 'Multi-study investigation is partial:'))
                or issue == DISCOVERY_SCOPE_LIMIT
                or (issue == 'One or more tool calls failed or were blocked; no substitute evidence was generated.' and _recovered_request_errors(report))
                for issue in report['validation']['issues']))
    if (
        (report.get("synthesis", {}).get("status") != "VALIDATED_FACT_SELECTION" and not background)
        or not report["validation"]["grounded"]
        or report["validation"].get("selection_counts", {}).get("omitted", 0)
    ):
        return fallback("SCIENTIFIC_GROUNDING_INCOMPLETE")
    planning_clarification = bool(
        scope and not scope["issues"] and (
            scope["request"]["purpose"] == "EXPERIMENT_PLANNING"
            or "exploratory_planning" in report.get("routing", {}).get("objectives", [])
        )
    )
    if not background and any(
        not issue.startswith(
            ("Integrated advice is partial:", "Multi-study investigation is partial:")
        )
        and not (planning_clarification and issue == PLANNING_CONSTRAINTS_ISSUE)
        and issue != DISCOVERY_SCOPE_LIMIT
        and not (issue == 'One or more tool calls failed or were blocked; no substitute evidence was generated.'
                 and _recovered_request_errors(report))
        for issue in report["validation"]["issues"]
    ):
        return fallback("SCIENTIFIC_SCOPE_INCOMPLETE")
    chosen = set(report["validation"]["selected_fact_ids"])
    # Live protein metadata has no new communication projection; retain its existing display boundary.
    excluded_calls = {
        e["call_id"] for e in report["trace"] if not tool_privacy(e["name"]).communication_allowed
    }
    recovered_calls = {e['call_id'] for e in report['trace'] if e['status'] == 'ERROR'} if _recovered_request_errors(report) else set()
    facts = {
        f["id"]: f
        for f in report["facts"]
        if f["id"] in chosen and f["call_id"] not in excluded_calls
        and not (f['call_id'] in recovered_calls and f['kind'] == 'UNANSWERED'
                 and f['evidence_category'] is None and not f['provenance'])
    }
    # Uncitable recovered request failures remain in report facts, execution
    # diagnostics and PARTIAL status; never attach a fabricated paper citation.
    # This explicit model instruction expands references, never scientific claims.
    # fact_ids remains mandatory and all original grounding checks run below.
    if candidate == {"render": "CANONICAL_SELECTED"}:
        candidate = canonical_explanation(list(facts.values()))
    if not isinstance(candidate, dict) or set(candidate) != set(SECTIONS):
        return fallback("INVALID_CONTRACT")
    sections: dict[str, list[CitedStatement]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    try:
        if len(json.dumps(candidate, allow_nan=False).encode()) > 65536:
            return fallback("EXPLANATION_TOO_LARGE")
        for section in SECTIONS:
            entries = candidate[section]
            if not isinstance(entries, list) or len(entries) > 100:
                return fallback("INVALID_CONTRACT")
            sections[section] = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"fact_id", "field", "text"}:
                    return fallback("INVALID_CONTRACT")
                if any(not isinstance(entry[k], str) for k in entry):
                    return fallback("INVALID_CONTRACT")
                identifier, field, text = entry["fact_id"], entry["field"], entry["text"]
                fact = facts.get(identifier)
                if fact is None or not fact["provenance"]:
                    return fallback("MISSING_OR_INVALID_CITATION")
                if text not in _permitted(fact, section, field):
                    return fallback("UNVERIFIED_WORDING_OR_ROLE")
                key = (section, identifier, field)
                if (*key, text) in seen:
                    return fallback("DUPLICATE_STATEMENT")
                seen.add((*key, text))
                proposal = _proposal(fact)
                context = (
                    {
                        k: proposal[k]
                        for k in ("priority", "prerequisites", "feasibility", "target_conditions")
                    }
                    if proposal
                    else None
                )
                sections[section].append(
                    {
                        **entry,
                        "evidence_category": None
                        if field == "publication"
                        else fact["evidence_category"],
                        "sources": _sources(fact),
                        "proposal_context": context,
                    }
                )
        # The concise explanation cannot hide a retrieved source observation or swap the top action.
        evidence_ids = {i["fact_id"] for i in sections["evidence"]}
        required_evidence = {
            f["id"] for f in facts.values() if _permitted(f, "evidence", "statement") or _permitted(f, "evidence", "measurement_summary")
        }
        if not required_evidence <= evidence_ids:
            return fallback("EVIDENCE_COVERAGE_INCOMPLETE")
        if (
            any(_permitted(f, "uncertainty", "statement") for f in facts.values())
            and not sections["uncertainty"]
        ):
            return fallback("UNCERTAINTY_OMITTED")
        required_uncertainty = {f["id"] for f in facts.values() if f["kind"] == "UNANSWERED"}
        if not required_uncertainty <= {i["fact_id"] for i in sections["uncertainty"]}:
            return fallback("UNCERTAINTY_COVERAGE_INCOMPLETE")
        proposals = [(f, _proposal(f)) for f in facts.values() if _proposal(f) is not None]
        if proposals:
            first = min(proposals, key=lambda item: item[1]["priority"])[0]["id"]
            if len(sections["next_action"]) != 1 or sections["next_action"][0]["fact_id"] != first:
                return fallback("PRIORITY_NOT_PRESERVED")
            if not any(i["fact_id"] == first for i in sections["rationale"]):
                return fallback("RATIONALE_OMITTED")
            for fact, proposal in proposals:
                if proposal["category"] == "PROPOSED_EXPERIMENT":
                    if not all(
                        any(
                            i["fact_id"] == fact["id"] and i["field"] == field
                            for i in sections["controls"]
                        )
                        for field in ("controlled_variables", "required_measurements")
                    ):
                        return fallback("CONTROLS_OMITTED")
        dois = {s["doi"] for f in facts.values() for s in f["provenance"] if s["doi"]}
        if {i["text"] for i in sections["publications"]} != dois:
            return fallback("PUBLICATIONS_OMITTED")
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return fallback("INVALID_CONTRACT")
    return {
        "status": "VERIFIED_BACKGROUND_EXPLANATION" if background else "VERIFIED_EXTRACTIVE",
        "reason_code": None,
        "sections": sections,
        "persistence": "TRANSIENT",
    }
