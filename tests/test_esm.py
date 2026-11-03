"""Tensor and synthetic-runtime unit tests, distinct from the real checkpoint smoke test."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from zymecycle import esm
from zymecycle.protein_sequences import prepare_protein_sequence


class SyntheticRuntime:
    spec = {'checkpoint':'synthetic-test-only', 'revision':'test', 'license':'test',
            'dimension':2, 'layer':1, 'pooling':'synthetic-test-mean', 'dtype':'float32'}
    device, fallback = 'cpu', None
    def encode(self, sequence):
        return [1.0, 2.0]


@pytest.fixture
def embedding():
    return esm.embed_protein(prepare_protein_sequence('ACDEF')['result'], runtime=SyntheticRuntime())['result']


def test_pooling_excludes_special_and_padding():
    hidden = torch.tensor([[[900.,900.], [1.,3.], [3.,5.], [700.,700.], [float('nan'),0.]]])
    assert esm.pool_residue_embeddings(hidden, torch.tensor([[1,1,1,1,0]]),
                                      torch.tensor([[1,0,0,1,1]]), 2) == [2.,4.]


@pytest.mark.parametrize('kind', ['count','shape','nonfinite','empty'])
def test_invalid_pooling(kind):
    hidden = torch.ones(1,4,2)
    attention = torch.ones(1,4)
    special = torch.tensor([[1,0,0,1]])
    count = 2
    if kind == 'count': count = 3
    if kind == 'shape': attention = attention[:,:3]
    if kind == 'nonfinite': hidden[0,1,0] = float('nan')
    if kind == 'empty': special = torch.ones(1,4); count = 0
    with pytest.raises(ValueError): esm.pool_residue_embeddings(hidden, attention, special, count)


def test_embedding_contract_and_no_grad_validation(embedding):
    assert embedding['vector'] == [1.,2.] and embedding['model']['checkpoint'] == 'synthetic-test-only'
    assert embedding['sequence_length'] == 5
    assert embedding['provenance'][0]['name'] == 'LOCAL_SEQUENCE'
    assert embedding['provenance'][-1]['name'] == 'ESM-2'
    json.dumps(embedding, allow_nan=False)


@pytest.mark.parametrize('vector', [[],[0.,0.],[1.],[float('nan'),1.],[float('inf'),1.],[True,1.]])
def test_invalid_runtime_results(vector):
    class Bad(SyntheticRuntime):
        def encode(self, sequence): return vector
    result = esm.embed_protein(prepare_protein_sequence('ACDEF')['result'], runtime=Bad())
    assert result['status'] == 'ERROR' and result['result'] is None


def test_model_load_failure_and_no_secret_or_sequence_in_error(monkeypatch):
    def failure(): raise OSError('ACDEF PRIVATE RESPONSE')
    monkeypatch.setattr(esm, 'load_esm2_runtime', failure)
    result = esm.embed_protein(prepare_protein_sequence('ACDEF')['result'])
    assert result['status'] == 'ERROR' and result['result'] is None
    assert 'ACDEF' not in json.dumps(result) and 'PRIVATE' not in json.dumps(result)


def test_unchanged_validated_sequence_required():
    prepared = prepare_protein_sequence('ACDEF')['result']
    prepared['normalized_sequence'] = 'ACDEG'
    assert esm.embed_protein(prepared, runtime=SyntheticRuntime())['status'] == 'ERROR'
    assert esm.embed_protein(prepare_protein_sequence('AX')['result'], runtime=SyntheticRuntime())['status'] == 'ERROR'


def test_cosine_known_values_and_scaling(embedding):
    other = copy.deepcopy(embedding)
    for left,right,expected in [([1.,0.],[1.,0.],1.), ([1.,0.],[0.,1.],0.),
                                ([1.,0.],[-1.,0.],-1.), ([1e300,1e300],[1e-300,1e-300],1.)]:
        embedding['vector'], other['vector'] = left,right
        result = esm.compare_protein_embeddings(embedding,other)
        assert result['status'] == 'SUCCESS' and result['evidence_category'] == 'COMPUTED'
        assert result['result']['cosine_similarity'] == pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize('field,value', [('dimension',3),('checkpoint','other'),('revision','other'),
    ('pooling','different'),('layer',2),('dtype','float16')])
def test_incompatible_models(embedding, field,value):
    other = copy.deepcopy(embedding)
    other['model'][field] = value
    assert esm.compare_protein_embeddings(embedding,other)['status'] == 'ERROR'


def test_zero_nonfinite_or_wrong_dimension_comparison(embedding):
    for vector in ([0.,0.],[float('nan'),1.],[1.]):
        other = copy.deepcopy(embedding); other['vector'] = vector
        assert esm.compare_protein_embeddings(embedding,other)['status'] == 'ERROR'


def test_missing_model_metadata_rejected(embedding):
    embedding['model']={'dimension':2}
    assert esm.compare_protein_embeddings(embedding,embedding)['status']=='ERROR'


def test_bad_local_checkpoint_integrity_is_an_error(tmp_path,monkeypatch):
    manifest=json.loads((esm.ROOT/'MODEL_PROVENANCE.json').read_text())
    directory=tmp_path/manifest['local_directory']
    directory.mkdir(parents=True)
    for name in manifest['files']:
        (directory/name).write_bytes(b'corrupted-test-only')
    (tmp_path/'MODEL_PROVENANCE.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(esm,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='integrity'):
        esm.ESM2Runtime()


def synthetic_engine():
    engine = object.__new__(esm.ESM2Runtime)
    engine.device, engine.fallback = 'cpu',None
    def tokenizer(sequence, **kwargs):
        assert kwargs['truncation'] is False
        size=len(sequence)+2
        return {'input_ids':torch.ones(1,size,dtype=torch.long),
                'attention_mask':torch.ones(1,size,dtype=torch.long),
                'special_tokens_mask':torch.tensor([[1]+[0]*len(sequence)+[1]])}
    class Backbone(torch.nn.Module):
        def __init__(self): super().__init__(); self.dropout=torch.nn.Dropout(.9)
        def forward(self, input_ids, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=self.dropout(torch.ones(1,input_ids.shape[1],2)))
    engine.tokenizer, engine.model = tokenizer,Backbone()
    return engine


def test_deterministic_eval_and_inference_mode():
    engine=synthetic_engine()
    assert engine.model.training
    assert engine.encode('ACDEF') == engine.encode('ACDEF') == [1.,1.]
    assert not engine.model.training


def test_token_count_mismatch_not_silently_truncated():
    engine=synthetic_engine()
    original=engine.tokenizer
    engine.tokenizer=lambda sequence,**kwargs: original(sequence[:-1],**kwargs)
    with pytest.raises(ValueError): engine.encode('ACDEF')


def test_mps_runtime_failure_recomputes_on_cpu(monkeypatch):
    engine=synthetic_engine(); engine.device='mps'
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,device,*a,**kw: self if device=='mps' else original_to(self,device,*a,**kw))
    original=engine.model.forward
    attempts=[]
    def forward(*args,**kwargs):
        attempts.append(engine.device)
        if engine.device=='mps': raise NotImplementedError('test-only unsupported operation')
        return original(*args,**kwargs)
    engine.model.forward=forward
    assert engine.encode('ACDEF') == [1.,1.]
    assert attempts == ['mps','cpu'] and engine.device=='cpu'
    assert 'recomputed on CPU' in engine.fallback
