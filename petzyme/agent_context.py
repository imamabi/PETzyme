"""Lossless source/fact references for bounded multi-study model context."""
from __future__ import annotations

from copy import deepcopy
import json

from .agent_evidence import model_evidence
from .agent_types import Fact, ToolEvent


def _signature(fact: dict) -> str:
    return json.dumps({k:fact[k] for k in ('kind','text','evidence_category','sources')},sort_keys=True)


def compact_model_evidence(event: ToolEvent, facts: list[Fact], messages: list[dict]) -> dict:
    """Compact multi-study tool replies via source catalogs and same_as aliases.

    Purpose: keeps long multi-study contexts bounded without losing
        information: repeated sources move into a per-reply
        source_catalog with source_refs, identical facts become
        same_as aliases, and preview fields already present verbatim in
        canonical facts are dropped with an evidence_location pointer.
        Other tools pass through model_evidence unchanged.
    Args:
        event: the executed ToolEvent.
        facts: the call's projected facts.
        messages: prior backend messages (scanned for known sources/facts).
    Returns:
        The model-visible reply dict for the next backend request.
    Limitations:
        Compaction is lossless by construction (pointers, not summaries),
            but selection obligations still cover every required ID.
    Flow: Receives ToolEvent/list[Fact] from execute_tool_call() and prior messages from run_investigation(); returns source-referenced dict for its next backend request."""
    reply = model_evidence(event,facts)
    if event['name'] not in ('investigate_pet_evidence', 'advise_next_experiment') or event['status'] == 'ERROR': return reply
    known_sources, known_facts, signatures = {}, {}, {}
    for message in messages:
        if message['role'] != 'tool': continue
        prior = json.loads(message['content'])
        known_sources.update(prior.get('source_catalog',{}))
        for fact in prior.get('facts',[]):
            if 'same_as' in fact:
                full = {**known_facts[fact['same_as']], 'id':fact['id']}
            else:
                full = {**fact, 'sources':fact.get('sources', [known_sources[r] for r in fact.get('source_refs',[])])}
            known_facts[fact['id']] = full
            signatures.setdefault(_signature(full),fact['id'])
    source_ids = {json.dumps(v,sort_keys=True):k for k,v in known_sources.items()}
    catalog, compact = {}, []
    for fact in reply['facts']:
        signature = _signature(fact)
        if signature in signatures:
            compact.append({'id':fact['id'],'same_as':signatures[signature],
                            'kind':fact['kind'],'evidence_category':fact['evidence_category']})
            continue
        refs = []
        for source in fact['sources']:
            key = json.dumps(source,sort_keys=True)
            if key not in source_ids:
                ref = f"{event['call_id']}.s{len(catalog)}"
                source_ids[key] = ref; catalog[ref] = source
            refs.append(source_ids[key])
        compact.append({**{k:v for k,v in fact.items() if k!='sources'},'source_refs':refs})
        signatures[signature] = fact['id']
    preview = deepcopy(reply['preview'])
    # Every removed value is already present verbatim in canonical facts, not a summary approximation.
    previews = ([c['preview'] for c in preview['components'] if c['name'] == 'investigate_pet_evidence']
                if event['name'] == 'advise_next_experiment' else [preview])
    for part in previews:
        for key in ('answer','coverage','comparisons','gaps','overlap_interpretation'): part.pop(key,None)
        observations = part.get('nrel',{}).pop('observations',[])
        if observations:
            part['nrel']['csv_sha256s'] = list(dict.fromkeys(o['csv_sha256'] for o in observations))
        part.get('pazy',{}).pop('records',None)
    preview['evidence_location'] = 'Original observations, full bounded assay projections, comparisons and gaps are in canonical facts; aliases refer to earlier identical facts.'
    reply.update(preview=preview,facts=compact,source_catalog=catalog)
    return reply
