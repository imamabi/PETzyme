"""Real local ESM-2 demonstrations with explicitly mocked Meta decisions; zero paid calls."""
from __future__ import annotations

import json
import math
import time
import httpx

from petzyme.agent import run_investigation
from petzyme.agent_reporting import format_research_report
from petzyme.esm import compare_protein_embeddings, embed_protein, load_esm2_runtime
from petzyme.meta import MetaModelClient
from petzyme.observations import load_nrel, select_nrel_protein
from petzyme.protein_sequences import prepare_protein_sequence
from petzyme.protein_similarity import find_similar_nrel_proteins

QUESTION = 'Find proteins with ESM-2 representations similar to LCC-ICCG.'


def demonstrate_agent() -> dict:
    """Flow: Receives external demo invocation; returns final privacy-audited dict from run_investigation() using real ESM-2 tools and mocked Meta HTTP."""
    requests = 0
    with_sequences = load_nrel()['result']
    index = with_sequences['columns'].index('sequence')
    sequences = [''.join(row[index].split()).upper() for row in with_sequences['rows']]

    def handler(request: httpx.Request) -> httpx.Response:
        """Flow: Receives mocked httpx.Request from MetaModelClient.complete(); returns synthetic httpx.Response for its next demonstration decision."""
        nonlocal requests
        requests += 1
        # Inspect in memory; never return or save raw requests, sequences or vectors.
        text = request.content.decode()
        assert not any(sequence in ''.join(text.split()).upper() for sequence in sequences if sequence)
        assert all('"'+key+'"' not in text.replace('\\"','"') for key in ('vector','normalized_sequence','amino_acid_sequence','rows'))
        messages = json.loads(text)['messages']
        replies = [json.loads(m['content']) for m in messages if m['role']=='tool']
        if not replies:
            name,arguments,call_id='load_nrel',{},'seq1'
        elif len(replies)==1:
            name,arguments,call_id='find_similar_nrel_proteins',{
                'table_ref':replies[0]['refs']['table'],'enzyme_identifier':'LCC-ICCG','top_k':5,'max_candidates':8},'seq2'
        else:
            ids = [f['id'] for f in replies[-1]['facts'] if f['kind']=='EVIDENCE']
            message={'role':'assistant','content':json.dumps({'fact_ids':ids})}
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':message}]})
        message={'role':'assistant','content':None,'tool_calls':[{'id':call_id,'type':'function',
                 'function':{'name':name,'arguments':json.dumps(arguments)}}]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls','message':message}]})

    result = run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(handler)))
    report = result['result']
    return {'status':result['status'],'mode':report['provider_mode'],'mock_meta_requests':requests,
            'real_esm_tools':True,'tool_sequence':[{'name':e['name'],'status':e['status']} for e in report['trace']],
            'grounded':report['validation']['grounded'],'complete':report['validation']['complete'],
            'privacy':'PASS: no full sequence/vector/table in mock reasoning context',
            'public_report':format_research_report(report)}


def real_demonstrations() -> dict:
    """Flow: Receives external local-smoke invocation; returns final JSON-safe dict of real embed_protein(), compare_protein_embeddings() and agent demonstration results."""
    import torch
    torch.set_num_threads(2)
    began=time.perf_counter()
    table=load_nrel()
    if table['status']!='SUCCESS': raise RuntimeError('Local NREL unavailable')
    protein=select_nrel_protein(table['result'],'LCC-ICCG')
    if protein['status']!='SUCCESS': raise RuntimeError('LCC-ICCG unavailable or ambiguous')
    prepared=prepare_protein_sequence(protein['result'])
    if prepared['status']!='SUCCESS': raise RuntimeError('LCC-ICCG sequence invalid')
    first=embed_protein(prepared['result'])
    repeat=embed_protein(prepared['result'])
    if first['status']!='SUCCESS' or repeat['status']!='SUCCESS':
        raise RuntimeError('Real ESM-2 inference failed; no synthetic replacement')
    a,b=first['result'],repeat['result']
    identical=compare_protein_embeddings(a,b)
    original=prepared['result']['normalized_sequence']
    replacement='A' if original[0]!='A' else 'G'
    changed=prepare_protein_sequence(replacement+original[1:])
    modified=embed_protein(changed['result'])
    if modified['status']!='SUCCESS': raise RuntimeError('Controlled-input real inference failed')
    comparison=compare_protein_embeddings(a,modified['result'])
    engine=load_esm2_runtime()
    previous_device,previous_fallback=engine.device,engine.fallback
    try:
        engine.model.to('cpu')
        engine.device,engine.fallback='cpu','Explicit CPU validation run'
        cpu=embed_protein(prepared['result'],runtime=engine)
    finally:
        engine.model.to(previous_device)
        engine.device,engine.fallback=previous_device,previous_fallback
    if cpu['status']!='SUCCESS': raise RuntimeError('Real CPU inference failed')
    ranking=find_similar_nrel_proteins(table['result'],'LCC-ICCG',top_k=5,max_candidates=8)
    if ranking['status']!='SUCCESS': raise RuntimeError('Real bounded ranking failed')
    demonstration={
        'execution':'REAL_PRETRAINED_ESM2_LOCAL','query':'LCC-ICCG','source_row':a['source_row'],
        'sequence_length':a['sequence_length'],'embedding_shape':[len(a['vector'])],
        'finite':all(math.isfinite(v) for v in a['vector']),'model':a['model'],
        'device':a['device'],'fallback':a['fallback'],
        'repeat_max_absolute_difference':max(abs(x-y) for x,y in zip(a['vector'],b['vector'])),
        'reproducibility_absolute_tolerance':1e-5,
        'identical_sequence_cosine':identical['result']['cosine_similarity'],
        'cpu_validation':{'status':cpu['status'],'device':cpu['result']['device'],
            'finite':all(math.isfinite(v) for v in cpu['result']['vector']),
            'cosine_to_default_device':compare_protein_embeddings(a,cpu['result'])['result']['cosine_similarity'],
            'max_absolute_difference_to_default':max(abs(x-y) for x,y in zip(a['vector'],cpu['result']['vector']))},
        'controlled_modification':{'category':'COMPUTED; diagnostic input, not a biological recommendation',
            'source':'LOCAL_SEQUENCE derived diagnostically from NREL LCC-ICCG; not an original NREL record',
            'position_1_based':1,'from_residue':original[0],'to_residue':replacement,
            'cosine_similarity':comparison['result']['cosine_similarity'],
            'max_absolute_vector_difference':max(abs(x-y) for x,y in zip(a['vector'],modified['result']['vector']))},
        'ranking':ranking,'agent_integration':demonstrate_agent()}
    demonstration['elapsed_seconds']=time.perf_counter()-began
    return demonstration


def main() -> None:
    """Flow: Receives external CLI invocation; prints final real_demonstrations() metadata/scores without raw sequences or vectors."""
    print(json.dumps(real_demonstrations(),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
