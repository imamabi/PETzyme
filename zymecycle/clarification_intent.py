"""Claim-screened natural clarification over independently validated scientific intent.

This bounded English request screen is not an arbitrary scientific prose verifier.
Assertions belong in canonical facts, displayed separately. Unknown or contradictory
claims retain a labelled fallback; ordinary request vocabulary is not allowlisted.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection

from .dialogue_types import ClarificationDecision, ClarificationSelection, DialogueContext
from .privacy import outbound_safe, question_contains_private_input

MAX_QUESTION_CHARS = 600  # Bounds transient model text, not a scientific threshold.
MAX_QUESTION_WORDS = 100
# Speech acts and missing-information topics classify intent, not complete sentences.
REQUEST = re.compile(
    r"\b(?:could|would|can|will|may)\s+(?:you|we|i)\b|\b(?:what|which|who|where)\b"
    r"|\b(?:please|kindly)\b|\b(?:provide|share|send|supply|clarify|confirm|specify)\b"
    r"|\b(?:do|did|are|is)\s+you\b|^\s*(?:is|does|are)\b",
    re.I,
)
TOPIC = {
    "IDENTIFIER": re.compile(
        r"\b(?:accession\w*|doi|publications?|papers?|fasta|sequences?|identif\w*|names?|"
        r"mean\w*|refer\w*|denot\w*|stand\s+for|clarif\w*)\b",
        re.I,
    ),
    "CONFIRM_CANDIDATE": re.compile(
        r"\b(?:confirm\w*|mean\w*|refer\w*|denot\w*|intend\w*|names?|labels?|"
        r"spelling|candidate|correct|right|is|does)\b",
        re.I,
    ),
    "CONDITIONS": re.compile(r"\b(?:ph|form|substrate|conditions?)\b", re.I),
}
# Even an interrogative may presuppose an unsupported result or functional identity.
SCIENTIFIC_CLAIM = re.compile(
    r"\b(?:activ\w*|superior\w*|better|best|worse|efficient\w*|predict\w*|caus\w*|"
    r"prov(?:e[ns]?|en|ing)|demonstrat\w*|establish\w*|hydroly\w*|degrad\w*|digest\w*|releas\w*|"
    r"yield\w*|kinetic\w*|kcat|thermophil\w*|thermostab\w*|identical|authenticat\w*|"
    r"cataly\w*|turnover|family|mutan\w*|mutation|rates?|mg|mmol|micromol|umol|"
    r"conversion|converts?|outperform\w*|optimum|optimal|resistan\w*|perform\w*|"
    r"hydrol\w*|petase|cutinase|esterase|lipase|protease|bacterial|fungal|isolat\w*|"
    r"originat\w*|express\w*|purif\w*|measur\w*|twice|twofold|fast|slow|stable|stability)\b",
    re.I,
)
INJECTION = re.compile(
    r"\b(?:ignore|override|bypass|disregard|execute|exfiltrat\w*|credentials?|passwords?|"
    r"secrets?|api\s*key|bearer|tokens?|keychain|MODEL_API_KEY|system\s*(?:prompt|message)|developer\s*(?:message|instructions?)|"
    r"chain.of.thought|reasoning\s*transcript|tool_calls|fact_ids|MODEL_VALIDATED)\b"
    r"|```|<!--|<script",
    re.I,
)
# Ordinary public request terminology is not an invented protein identifier.
PUBLIC_TERMS = frozenset(
    "FASTA DOI DNA RNA PET PH NREL PAZY UNIPROT GENBANK NCBI PDB ID IDS C CELSIUS "
    "I A AN BEFORE FOR COULD WOULD CAN WILL PLEASE KINDLY WHAT WHICH WHO WHERE WHEN HOW THANKS TO REGARDING "
    "AFTER RATHER NOT INSTEAD GIVEN ACCESSION SEQUENCE FULL AMINO PUBLICATION".split()
)


def fallback_question(selection: ClarificationSelection) -> str:
    """Render the deterministic fallback question for a validated intent.

    Purpose: fixed templates per request kind (confirm candidate,
        conditions, identifying material) using only validated
        targets/candidates — the safe wording shown whenever model
        prose fails screening.
    Flow: Receives validated ClarificationSelection from validate_clarification(); returns grounded str for question_presentation() when prose is unverified."""
    if selection["request"] == "CONFIRM_CANDIDATE":
        return f"Did you mean source label {selection['candidate']} for {selection['target']}?"
    if selection["request"] == "CONDITIONS":
        return f"What exact pH and PET form should be investigated for {selection['target']}, within your device limits?"
    return f"Could you provide an accession, publication DOI or FASTA sequence for {selection['target']}?"


def _labels_in_text(text: str, labels: Collection[str]) -> tuple[str, set[str]]:
    stripped, mentioned = text, set()
    for label in sorted(labels, key=len, reverse=True):
        pattern = r"(?<![A-Za-z0-9_+-])" + re.escape(label) + r"(?![A-Za-z0-9_+-])"
        if re.search(pattern, stripped, re.I):
            mentioned.add(label)
        stripped = re.sub(pattern, " TARGET ", stripped, flags=re.I)
    return stripped, mentioned


def _constraint_failure(
    text: str, context: DialogueContext, numbers: Collection[float]
) -> str | None:
    allowed = {*context["temperatures_c"], context["maximum_ph"], *numbers}
    if context["assay_ph"] is not None:
        allowed.add(context["assay_ph"])
    if any(float(n) not in allowed for n in re.findall(r"(?<![\w])-?\d+(?:\.\d+)?", text)):
        return "UNVERIFIED_NUMBERS"
    for n in re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|Celsius|C\b)", text, re.I):
        if float(n) not in context["temperatures_c"]:
            return "UNAVAILABLE_TEMPERATURE"
    for match in re.finditer(
        r"\bpH\s*(?:<=|≤|<|=|is|of|at most|up to|cannot exceed|must be|should be)?\s*(\d+(?:\.\d+)?)",
        text,
        re.I,
    ):
        value = float(match[1])
        fragment = text[max(0, match.start() - 24) : match.end() + 24].casefold()
        maximum = bool(
            re.search(r"maximum|limit|ceiling|at most|up to|cannot exceed|<=|≤", fragment)
        )
        if (
            re.search(r"\bpH\s+(?:must|should)\s+be", match[0], re.I)
            and value != context["assay_ph"]
        ):
            return "UNREQUESTED_ASSAY_PH"
        if re.search(r"\bpH\s*<(?![=])", match[0], re.I):
            return "CONSTRAINT_CONTRADICTION"
        if maximum and value != context["maximum_ph"]:
            return "CONSTRAINT_CONTRADICTION"
        if (
            re.search(r"\b(?:at|using|assay)\s+pH\s*\d", fragment, re.I)
            and not re.search(r"or\s+(?:below|lower)|at most|<=|≤", fragment, re.I)
            and value != context["assay_ph"]
        ):
            return "UNREQUESTED_ASSAY_PH"
        if not maximum and value != context["assay_ph"]:
            return "UNREQUESTED_ASSAY_PH"
    if re.search(
        r"\b(?:above|over|exceed|higher than)\s+(?:pH\s*)?\d", text, re.I
    ) and not re.search(r"\b(?:not|never|cannot|can't|no)\b", text, re.I):
        return "CONSTRAINT_CONTRADICTION"
    return None


def _claim_failure(text: str, selection: ClarificationSelection) -> str | None:
    def neutral_purpose(match: re.Match[str]) -> str:
        purpose = match[0]
        # A comparison topic is not a claim that either protein has that function.
        # Outcome qualifiers and assertion clauses remain subject to the ordinary checks.
        if re.search(
            r"\b(?:high\w*|low\w*|increased|improved|observed|measured|recorded|excellent|strong|known)\b",
            purpose,
            re.I,
        ):
            return purpose
        return re.sub(
            r"\b(?:activity|activities|degradation|hydrolysis)\b",
            "research topic",
            purpose,
            flags=re.I,
        )

    claim_text = re.sub(
        r"\b(?:before|to|so (?:we|i) can)\s+(?:compare|comparing|investigate|investigating|evaluate|evaluating)\s+[^?;.!]*",
        neutral_purpose,
        text,
        flags=re.I,
    )
    if SCIENTIFIC_CLAIM.search(claim_text):
        return "UNSUPPORTED_SCIENTIFIC_CLAIM"
    # A second sentence is allowed only if it too requests information, not states a fact.
    sentences = [s.strip() for s in re.split(r"(?<!\d)[.!;](?!\d)\s*", text) if s.strip()]
    if any(not REQUEST.search(s) for s in sentences):
        return "UNVERIFIED_DECLARATIVE_CLAUSE"
    # Embedded wh-questions ask what a name means; they do not assert an identity.
    screened = re.sub(
        r"\b(?:what|which)\s+TARGET\s+(?:is|means|denotes|represents)\b",
        " identifying request ",
        text,
        flags=re.I,
    )
    if re.search(
        r"\b(?:TARGET|it|this\s+(?:enzyme|protein|sequence))\s+"
        r"(?:is|are|was|were|has|have|equals?|means|denotes|represents?|corresponds?|matches?|encodes?|comes|contains|belongs|works|worked)\b"
        r"|\b(?:known\s+as|also\s+called|aka|because|since|therefore|in\s+fact)\b"
        r"|\b(?:which|that)\s+(?:is|are|has|was|can)\b"
        r"|,\s*(?:which|that)\s+(?!you\b|we\b|i\b)"
        r"|\bTARGET\s*,?\s*(?:which|that)\s+(?!you\b|we\b|i\b)"
        r"|\bTARGET\s+(?:is\s+)?(?:not\s+)?(?:in|listed|found|recorded|absent|missing|known)\b",
        screened,
        re.I,
    ):
        return "UNVERIFIED_FACTUAL_CLAUSE"
    if (
        re.search(r"\b(?:same|equivalent)\b", text, re.I)
        and selection["request"] != "CONFIRM_CANDIDATE"
    ):
        return "UNVERIFIED_IDENTITY_RELATION"
    # Unknown alphanumeric IDs and unrecognized proper names are not introduced by prose.
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text):
        if word == "TARGET" or word.upper() in PUBLIC_TERMS:
            continue
        initial_request_word = bool(REQUEST.match(text) and re.match(re.escape(word) + r"\b", text))
        if re.search(r"\d", word) or word[0].isupper() and not initial_request_word:
            return "UNVERIFIED_REFERENT"
    return None


def question_presentation(
    selection: ClarificationSelection,
    context: DialogueContext,
    *,
    verified_numbers: Collection[float] = (),
    verified_symbols: Collection[str] = (),
) -> ClarificationDecision:
    """Screen model clarification wording; fall back when it over-claims.

    Purpose: conservative request/claim screen — unsupported numbers,
        symbols, labels, or scientific assertions in the prose trigger
        the labelled deterministic fallback. Not a general semantic
        prover: unusual vocabulary or extra sentences can cause
        false-positive fallbacks.
    Flow: Receives validated intent/context and local metadata from validate_clarification(); returns ClarificationDecision with screened natural wording or explicit deterministic fallback."""
    draft = selection["question"]
    text = unicodedata.normalize("NFKC", draft).replace("’", "'")
    labels = {
        selection["target"],
        *context["requested_labels"],
        *[x for x in context["confirmed_labels"] if x],
    }
    rejected = {x for group in context["rejected_labels"] for x in group}
    labels.update(rejected)
    if selection["candidate"] is not None:
        labels.add(selection["candidate"])
    stripped, mentioned = _labels_in_text(text, labels)
    # Actual local validation symbols are metadata, never sequence bytes or identity proof.
    for symbol in verified_symbols:
        stripped = re.sub(r"\b" + re.escape(symbol) + r"\b", " residue symbol ", stripped)
    failure = None
    if (
        len(draft) > MAX_QUESTION_CHARS
        or len(text.split()) > MAX_QUESTION_WORDS
        or any(unicodedata.category(c) == "Cf" or c.isalpha() and not c.isascii() for c in text)
    ):
        failure = "QUESTION_BOUNDS_OR_SCRIPT"
    elif (
        not outbound_safe(draft)
        or question_contains_private_input(draft)
        or re.search(r"[0-9a-fA-F]{64}|/Users/|/private/|/etc/|[A-Za-z][A-Za-z0-9+.-]*://", draft)
    ):
        failure = "QUESTION_PRIVACY"
    elif INJECTION.search(text):
        failure = "PROMPT_INJECTION_OR_SECRET_REQUEST"
    elif not REQUEST.search(text) or text.count("?") > 1:
        failure = "NOT_FOCUSED_REQUEST"
    elif selection["target"] not in mentioned:
        failure = "MISSING_TARGET"
    elif selection["request"] == "CONFIRM_CANDIDATE" and selection["candidate"] not in mentioned:
        failure = "MISSING_CONFIRMATION_CANDIDATE"
    elif selection["request"] != "CONFIRM_CANDIDATE" and any(
        re.search(
            r"\b(?:mean|referring to|refer to|same as)\s+"
            + re.escape(other)
            + r"(?![A-Za-z0-9_+-])"
            r"|" + re.escape(other) + r"(?:'s)?\s+(?:sequence|accession|doi)\b"
            r"|\b(?:sequence|accession|doi)\s+(?:of|for|from)\s+"
            + re.escape(other)
            + r"(?![A-Za-z0-9_+-])",
            text,
            re.I,
        )
        for other in mentioned - {selection["target"]} - rejected
    ):
        failure = "UNVERIFIED_IDENTITY_RELATION"
    elif any(
        not re.search(
            r"\b(?:not|rather than|instead of|excluding|ruled out)\s+"
            r"(?:the\s+(?:enzyme|label)\s+)?" + re.escape(label) + r"(?![A-Za-z0-9_+-])",
            text,
            re.I,
        )
        for label in mentioned & rejected
    ):
        failure = "REJECTED_CANDIDATE_REINTRODUCED"
    elif not TOPIC[selection["request"]].search(stripped):
        failure = "INTENT_WORDING_MISMATCH"
    else:
        failure = _constraint_failure(stripped, context, verified_numbers) or _claim_failure(
            stripped, selection
        )
    return {
        "action": "ASK_CLARIFICATION",
        "target": selection["target"],
        "request": selection["request"],
        "candidate": selection["candidate"],
        "question": fallback_question(selection) if failure else draft,
        "presentation": "DETERMINISTIC_FALLBACK" if failure else "MODEL_VALIDATED",
        "fallback_reason": failure,
    }
