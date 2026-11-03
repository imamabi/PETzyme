import json

from zymecycle.agent_context import compact_model_evidence
from zymecycle.agent_evidence import model_evidence, project_evidence
from zymecycle.pet_evidence import investigate_pet_evidence


def test_compact_context_retains_every_fact_id_value_warning_and_source():
    messages=[]; expanded={};sources={}; before=after=0
    for i,t in enumerate((40,60)):
        result=investigate_pet_evidence({'protein_name':'ESM050','temperature_c':t,'ph':6.5,'form':'powder'})
        preview,facts=project_evidence(f'c{i}','investigate_pet_evidence',result)
        event={'call_id':f'c{i}','name':'investigate_pet_evidence','status':result['status'],'refs':{},'preview':preview}
        old=model_evidence(event,facts)
        new=compact_model_evidence(event,facts,messages)
        before+=len(json.dumps(old).encode());after+=len(json.dumps(new).encode())
        sources.update(new['source_catalog'])
        for old_fact,fact in zip(old['facts'],new['facts']):
            assert old_fact['id']==fact['id']
            if 'same_as' in fact:
                full={**expanded[fact['same_as']],'id':fact['id']}
            else:
                full={k:v for k,v in fact.items() if k!='source_refs'}
                full['sources']=[sources[r] for r in fact['source_refs']]
            assert full==old_fact
            expanded[fact['id']]=full
        assert new['preview']['units']==old['preview']['units']
        assert new['preview']['partial_reasons']==old['preview']['partial_reasons']
        assert new['preview']['nrel']['csv_sha256s']==[old['preview']['nrel']['observations'][0]['csv_sha256']]
        assert len(new['facts'])==len(old['facts'])
        if i: assert any('same_as' in f for f in new['facts']) and not new['source_catalog']
        messages.append({'role':'tool','content':json.dumps(new)})
    assert after < before * .75
