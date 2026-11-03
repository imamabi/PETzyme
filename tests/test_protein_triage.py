"""Synthetic fixtures plus bounded real-local triage; no external records are written."""
import hashlib
import json
from pathlib import Path
import pytest
from petzyme import protein_triage as triage
from petzyme.observations import load_nrel, select_nrel_protein
from petzyme.results import outcome

SEQ = 'ACDEFGHIKLMNPQRSTVWY' * 3
OTHER = 'YWVTSRQPNMLKIHGFEDCA' * 3


@pytest.fixture
def local(tmp_path, monkeypatch):
    path = tmp_path/'data/nrel/label_data.csv'
    path.parent.mkdir(parents=True)
    path.write_text(',sequence\nE1,'+SEQ+'\nE2,'+OTHER+'\nAMB,ACDX\n')
    manifest = {'dataset_doi':'10.5281/zenodo.15417757','extracted_file':'data/nrel/label_data.csv',
                'unit_reconciliation':{'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path/'NREL_SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(triage,'ROOT',tmp_path)
    source = {'name':'PAZy activity snapshot','url':'https://example.invalid/pin','doi':None,
              'locator':'#synthetic','attribution':'Synthetic fixture','retrieved_at_utc':None}
    records = [
        {'protein_id':11,'protein_name':'SyntheticAlpha','literature_doi':'10.0000/synthetic',
         'synonyms':[{'name':'shared-label'},{'name':'documented-alpha'}]},
        {'protein_id':12,'protein_name':'SyntheticBeta','literature_doi':'10.0000/synthetic',
         'synonyms':[{'name':'shared-label'}]},
    ]
    monkeypatch.setattr(triage,'load_activity_source',lambda:outcome('SUCCESS',{'records':records,'source':source},[source],[],None))
    return path


def test_name_resolution_is_candidate_not_sequence_proof(local):
    result = triage.triage_protein(name='E1')
    assert result['status']=='SUCCESS' and result['result']['identity_status']=='CANDIDATE_MATCH'
    assert result['result']['candidates'][0]['basis']==['EXACT_NAME']
    assert result['result']['persistence']=='TRANSIENT' and result['evidence_category']=='COMPUTED'


def test_documented_alias_ambiguity_and_omissions(local):
    value = triage.triage_protein(name='shared-label',limit=1)['result']
    assert value['identity_status']=='CANDIDATE_MATCH' and value['multiple_source_records']
    assert value['total_candidates']==2 and value['omitted_candidates']==1
    assert value['candidates'][0]['basis']==['DOCUMENTED_ALIAS']
    assert value['candidates'][0]['identifier']=='11'
    assert value['candidates'][0]['source']['doi']=='10.0000/synthetic'
    exact = triage.triage_protein(name='SyntheticAlpha')['result']
    assert exact['candidates'][0]['basis']==['EXACT_NAME']


def test_normalized_suggestions_not_silent_equivalence(local):
    value = triage.triage_protein(name='shared label')['result']
    assert value['identity_status']=='CANDIDATE_MATCH' and value['multiple_source_records']
    assert all(c['basis']==['NORMALIZED_NAME_CANDIDATE'] for c in value['candidates'])
    assert triage.triage_protein(name='UNLISTED')['result']['identity_status']=='UNRESOLVED'


@pytest.mark.parametrize('value',[SEQ,SEQ.lower(),'>not-an-identity-proof\n'+SEQ[:20]+'\n'+SEQ[20:]])
def test_exact_sequence_and_fasta_match_without_echoing_input(local,value):
    result = triage.triage_protein(sequence_input=value)
    assert result['result']['identity_status']=='IDENTIFIED_SEQUENCE'
    assert result['result']['candidates'][0]['basis']==['EXACT_SEQUENCE']
    assert result['result']['sequence_length']==len(SEQ)
    assert SEQ not in json.dumps(result) and 'not-an-identity-proof' not in json.dumps(result)
    assert result['result']['alias_source_checked'] is False


def test_fragment_short_sequence_and_nonmatching_sequence(local):
    fragment = triage.triage_protein(sequence_input=SEQ[3:34])['result']
    assert fragment['identity_status']=='CANDIDATE_MATCH'
    assert fragment['candidates'][0]['basis']==['EXACT_FRAGMENT']
    short = triage.triage_protein(sequence_input=SEQ[:10])['result']
    assert short['identity_status']=='UNRESOLVED' and not short['candidates']
    assert 'SHORT_SEQUENCE_FRAGMENT_SEARCH_DISABLED' in short['issues']
    assert triage.triage_protein(sequence_input='W'*40)['result']['identity_status']=='UNRESOLVED'


def test_name_and_sequence_cannot_be_silently_combined(local):
    good=triage.triage_protein(name='E1',sequence_input=SEQ)['result']
    assert good['identity_status']=='IDENTIFIED_SEQUENCE' and good['name_sequence_agreement']=='MATCH'
    wrong=triage.triage_protein(name='E2',sequence_input=SEQ)['result']
    assert wrong['identity_status']=='CANDIDATE_MATCH' and wrong['name_sequence_agreement']=='UNVERIFIED'
    assert 'NAME_SEQUENCE_NOT_RECONCILED' in wrong['issues']


@pytest.mark.parametrize('text,reason',[
    ('','MISSING'),('ACDX','AMBIGUOUS_RESIDUES'),('BJZ','AMBIGUOUS_RESIDUES'),
    ('ACDUO','UNSUPPORTED_RESIDUES'),('ACD-*1','UNSUPPORTED_RESIDUES'),('ß','UNSUPPORTED_RESIDUES'),
    ('>\nACDEF','INVALID_FASTA'),('>one\nACDEF\n>two\nACDEG','INVALID_FASTA'),
    ('>no-body','INVALID_FASTA'),('A'*1023,'TOO_LONG'),('A'*16385,'INPUT_TOO_LARGE')])
def test_invalid_input_is_unresolved_and_not_repaired(local,text,reason):
    result=triage.triage_protein(name='E1',sequence_input=text)
    assert result['status']=='ERROR' and result['result']['identity_status']=='UNRESOLVED'
    assert result['result']['sequence_validation']==reason and not result['result']['candidates']


@pytest.mark.parametrize('args',[{}, {'name':''},{'name':7},{'sequence_input':[]},
    {'name':'E1','limit':True},{'name':'E1','limit':6}])
def test_argument_contract(local,args):
    assert triage.triage_protein(**args)['status']=='ERROR'


def test_authentication_failure_cannot_identify_sequence(local):
    local.write_text(',sequence\nE1,'+OTHER+'\n')
    value=triage.triage_protein(sequence_input=OTHER)['result']
    assert value['identity_status']=='UNRESOLVED' and value['issues']==['NREL_SOURCE_UNVERIFIED']


def test_alias_source_failure_is_explicit(local,monkeypatch):
    monkeypatch.setattr(triage,'load_activity_source',lambda:outcome('ERROR',None,[],[],None))
    result=triage.triage_protein(name='E1')
    assert result['status']=='PARTIAL' and result['result']['identity_status']=='CANDIDATE_MATCH'
    assert result['result']['issues']==['ALIAS_SOURCE_UNAVAILABLE']


def test_real_local_sequence_authenticated_and_fragment_only_candidate():
    table=load_nrel()['result'];protein=select_nrel_protein(table,'ESM050')['result']
    sequence=protein['amino_acid_sequence']
    exact=triage.triage_protein(sequence_input=sequence)['result']
    assert exact['identity_status']=='IDENTIFIED_SEQUENCE'
    assert any(c['identifier']=='ESM050' for c in exact['candidates'])
    assert triage.triage_protein(sequence_input=sequence[30:70])['result']['identity_status']=='CANDIDATE_MATCH'


def test_no_inference_network_or_writes(local,monkeypatch,capsys,caplog):
    import builtins,io,os
    from petzyme import esm,meta,pazy
    def blocked(*a,**k):raise AssertionError('Inference/storage forbidden')
    original=builtins.open;original_io=io.open
    def readonly(opener):
        def wrapped(path,mode='r',*a,**k):
            assert not set(mode)&set('wax+')
            return opener(path,mode,*a,**k)
        return wrapped
    monkeypatch.setattr(builtins,'open',readonly(original));monkeypatch.setattr(io,'open',readonly(original_io))
    monkeypatch.setattr(os,'open',blocked)
    monkeypatch.setattr(esm,'load_esm2_runtime',blocked);monkeypatch.setattr(meta.MetaModelClient,'complete',blocked)
    monkeypatch.setattr(pazy.PazyClient,'_get',blocked)
    assert triage.triage_protein(name='E1',sequence_input=SEQ)['result']['identity_status']=='IDENTIFIED_SEQUENCE'
    captured=capsys.readouterr();assert not captured.out and not captured.err and not caplog.text


def test_discovery_dispatch_and_no_model_sequence_input(local):
    from petzyme.tools import describe_tools,invoke_tool
    from petzyme.agent_request import interpret_request,select_tool_catalog
    from petzyme.ui_adapter import EXAMPLES
    descriptor=next(t for t in describe_tools()['result'] if t['name']=='triage_protein')
    assert descriptor['output_persistence']=='FORBIDDEN_PAZY_DERIVED'
    assert descriptor['input_schema']['additionalProperties'] is False
    result=invoke_tool('triage_protein',{'sequence_input':'>private-header\n'+SEQ})
    assert result['result']['identity_status']=='IDENTIFIED_SEQUENCE'
    assert SEQ not in json.dumps(result) and 'private-header' not in json.dumps(result)
    assert invoke_tool('triage_protein',{'sequence_input':SEQ,'limit':True})['status']=='ERROR'
    for question in EXAMPLES.values():
        assert 'triage_protein' not in {t['function']['name'] for t in select_tool_catalog(interpret_request(question))}
