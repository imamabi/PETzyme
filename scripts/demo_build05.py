"""Actual local scientific tools with explicitly scripted MOCK Meta decisions; no paid requests."""
from __future__ import annotations
import json
import time
import httpx
from zymecycle.agent import run_investigation
from zymecycle.advisor_types import ResearcherConstraints
from zymecycle.meta import MetaModelClient
from zymecycle.observations import load_nrel

QUESTIONS = {
    'A': 'What experiment should I conduct to determine whether ESM050 performs better at 60°C than at 40°C, at pH 6.5 using crystalline PET powder?',
    'B': 'What should I do to evaluate ESM050 at 50°C and pH 8?',
    'C': 'How could I experimentally evaluate whether an ESM-2-similar protein has PET-degradation activity comparable to LCC-ICCG?',
}


def demonstrate_scenario(scenario: str, *, csv_path: str = 'data/nrel/label_data.csv',
                         researcher_constraints: ResearcherConstraints | None = None) -> dict:
    """Flow: Receives external scenario str and local CSV configuration; returns final JSON-safe demonstration dict from run_investigation() with scripted MockTransport decisions."""
    question = QUESTIONS[scenario]
    table = load_nrel(csv_path)['result']
    if table is None: raise RuntimeError('Local NREL unavailable; no substitute evidence')
    sequence_column = table['columns'].index('sequence')
    sequences = [''.join(row[sequence_column].split()).upper() for row in table['rows'] if row[sequence_column]]
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        text = request.content.decode()
        assert not any(s in ''.join(text.split()).upper() for s in sequences)
        assert all('"' + k + '"' not in text.replace('\\"', '"') for k in ('vector', 'rows', 'normalized_sequence', 'amino_acid_sequence'))
        messages = json.loads(text)['messages']
        replies = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        if not replies:
            name, args = 'load_nrel', {}
        elif len(replies) == 1:
            name = 'find_similar_nrel_proteins' if scenario == 'C' else 'parse_nrel'
            args = {'table_ref': replies[0]['refs']['table']}
            if scenario == 'C': args.update(enzyme_identifier='LCC-ICCG', top_k=5, max_candidates=8)
        elif len(replies) == 2 and scenario != 'C':
            name = 'investigate_conditions'
            a = {'temperature_c': 40 if scenario == 'A' else 50, 'ph': 6.5 if scenario == 'A' else 8,
                 'pet_form': 'CRYSTALLINE_POWDER'}
            b = {**a, **({'temperature_c': 60} if scenario == 'A' else {'pet_form': 'AMORPHOUS_FILM'})}
            args = {'observations_ref': replies[-1]['refs']['observations'], 'enzyme_identifier': 'ESM050',
                    'condition_a': a, 'condition_b': b}
        elif (len(replies) == 2 and scenario == 'C') or (len(replies) == 3 and scenario != 'C'):
            name = 'recommend_next_experiments'
            args = {'investigation_ref': replies[-1]['refs']['similarity' if scenario == 'C' else 'investigation'],
                    'max_recommendations': 10}
        else:
            ids = [f['id'] for reply in replies for f in reply['facts'] if f['kind'] in ('EVIDENCE', 'PROPOSAL')]
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message':
                {'role': 'assistant', 'content': json.dumps({'fact_ids': ids})}}]})
        message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'a{len(replies) + 1}',
                   'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'tool_calls', 'message': message}]})

    began = time.perf_counter()
    result = run_investigation(question, MetaModelClient(transport=httpx.MockTransport(handler)),
                               csv_path=csv_path, researcher_constraints=researcher_constraints)
    report = result['result']
    proposals = [json.loads(f['text'].split('PROPOSED (not observed): ', 1)[1]) for f in report['facts'] if f['kind'] == 'PROPOSAL']
    return {'scenario': scenario, 'question': question, 'status': result['status'], 'mode': report['provider_mode'],
        'mock_meta_http_requests': requests, 'model_requests': report['model_requests'],
        'scientific_tool_attempts': report['tool_attempts'], 'elapsed_seconds': time.perf_counter() - began,
        'validation': report['validation'], 'privacy': 'PASS: no full sequences, vectors or tables in mock reasoning context',
        'tool_sequence': [{'name': e['name'], 'status': e['status'], 'preview': e['preview']} for e in report['trace']],
        'observed_facts': [f['text'] for f in report['facts'] if f['evidence_category'] == 'MEASURED'],
        'proposals': proposals,
        'limitations': list(dict.fromkeys(f['text'] for f in report['facts'] if f['kind'] in ('LIMITATION', 'UNANSWERED')))}


def main() -> None:
    """Flow: Receives external CLI invocation; prints final demonstrate_scenario() results without sequences, vectors or provider history."""
    import torch
    torch.set_num_threads(2)
    print(json.dumps([demonstrate_scenario(s) for s in QUESTIONS], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
