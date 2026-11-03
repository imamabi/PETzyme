"""Real local triage with mocked researcher turns; no model/backend calls."""
from copy import deepcopy
import json
import pytest
from petzyme.clarification import start_clarification, advance_clarification
from petzyme.observations import load_nrel, select_nrel_protein

QUESTION = 'Should I use ESM050 or LCCMC? My device supports 40C and 60C, but I cannot operate above pH 7.'


def assert_context(state):
    assert state['objective']=='CHOOSE_BETWEEN_ENZYMES'
    assert state['requested_labels'][0]=='ESM050'
    assert state['confirmed_labels'][0]=='ESM050'
    assert state['temperatures_c']==[40,60] and state['maximum_ph']==7


def test_unknown_lccmc_never_becomes_lcc_iccg():
    turn=start_clarification(QUESTION);s=turn['state']
    assert_context(s)
    assert s['requested_labels'][1]=='LCCMC' and s['confirmed_labels'][1] is None
    assert s['pending_candidate'] is None and 'FASTA' in s['pending_question']
    assert 'LCC-ICCG' not in json.dumps(turn)


def test_source_candidate_confirmation_rejection_and_no_repeat():
    s=start_clarification(QUESTION.replace('LCCMC','LCC ICCG'))['state']
    assert s['pending_candidate']=='LCC-ICCG'
    assert s['pending_question'].count('?')==1
    before=deepcopy(s)
    rejected=advance_clarification(s,'No')['state']
    assert s==before
    assert_context(rejected)
    assert rejected['rejected_labels'][1]==['LCC-ICCG']
    assert rejected['pending_candidate'] is None and 'FASTA' in rejected['pending_question']
    again=advance_clarification(rejected,'LCC ICCG')['state']
    assert again['pending_candidate'] is None
    assert again['rejected_labels']==rejected['rejected_labels']


def test_yes_only_confirms_source_label_and_resumes_original_question():
    s=start_clarification(QUESTION.replace('LCCMC','LCC ICCG'))['state']
    turn=advance_clarification(s,'Yes')
    assert_context(turn['state'])
    assert turn['state']['confirmed_labels']==['ESM050','LCC-ICCG']
    assert turn['state']['pending_index'] is None
    assert 'No winning enzyme' in turn['state']['pending_question']
    assert 'unverified' in turn['message']


def test_real_fasta_and_raw_sequence_triage_resume_without_retaining_input():
    sequence=select_nrel_protein(load_nrel()['result'],'DP038')['result']['amino_acid_sequence']
    for payload,kind in [('>PRIVATE_HEADER\n'+sequence,'ANSWER'),(sequence,'SEQUENCE')]:
        s=advance_clarification(start_clarification(QUESTION)['state'],'No')['state']
        turn=advance_clarification(s,payload,kind)
        assert turn['triage_outcome']=='IDENTIFIED_SEQUENCE' and turn['sequence_validation']=='VALID'
        assert_context(turn['state'])
        assert turn['state']['confirmed_labels'][1]=='DP038'
        assert turn['state']['requested_labels'][1]=='LCCMC'
        assert sequence not in json.dumps(turn) and 'PRIVATE_HEADER' not in json.dumps(turn)
        assert 'construct identity and function remain unverified' in turn['message']
        assert 'No winning enzyme' in turn['state']['pending_question']


def test_j_residue_fragment_and_reference_do_not_overwrite_context():
    s=start_clarification(QUESTION)['state']
    bad=advance_clarification(s,'ACDJ','SEQUENCE')
    assert bad['triage_outcome']=='UNRESOLVED' and bad['sequence_validation']=='AMBIGUOUS_RESIDUES'
    assert_context(bad['state'])
    assert bad['state']['confirmed_labels'][1] is None
    ref=advance_clarification(bad['state'],'10.1021/acscatal.5c03460','REFERENCE')
    assert 'no accession/DOI lookup' in ref['message']
    assert '10.1021' not in json.dumps(ref['state'])
    sequence=select_nrel_protein(load_nrel()['result'],'DP038')['result']['amino_acid_sequence']
    fragment=advance_clarification(ref['state'],sequence[25:65],'SEQUENCE')
    assert fragment['triage_outcome']=='CANDIDATE_MATCH'
    assert fragment['state']['confirmed_labels'][1] is None
    assert_context(fragment['state'])


@pytest.mark.parametrize('reply',['No, actually pH 8 and 80C','replace ESM050 with DP038','Yes, use pH 8'])
def test_ambiguous_turns_cannot_override_constraints_or_confirmed_label(reply):
    s=start_clarification(QUESTION)['state']
    turn=advance_clarification(s,reply)
    assert_context(turn['state'])
    assert turn['state']['confirmed_labels'][1] is None


@pytest.mark.parametrize('a,b',[('DP038','unknown-X'),('TEP021','unknown-X'),('LCC-ICCG','unknown-X')])
def test_generic_identifiers_not_hardcoded(a,b):
    q=f'Should I use {a} or {b}? My device supports 35C and 55C, pH <=6.8.'
    s=start_clarification(q)['state']
    assert s['requested_labels']==[a,b]
    assert s['temperatures_c']==[35,55] and s['maximum_ph']==6.8
    assert s['confirmed_labels'][0]==a
    assert s['pending_index']==1


def test_no_conditions_inferred_or_source_records_retained():
    with pytest.raises(ValueError):start_clarification('Should I use ESM050 or DP038?')
    with pytest.raises(ValueError):start_clarification(QUESTION+' maximum pH 8')
    with pytest.raises(ValueError):start_clarification(QUESTION.replace('LCCMC','ACDEFGHIKLMNPQRSTVWY'*2))
    s=start_clarification(QUESTION)['state']
    assert not {'candidates','sequence','sequence_input','triage_result','records','provenance','response'} & s.keys()


def test_rejected_label_is_not_suggested_again_even_after_exact_sequence_match():
    s=start_clarification(QUESTION.replace('LCCMC','LCC ICCG'))['state']
    s=advance_clarification(s,'No')['state']
    sequence=select_nrel_protein(load_nrel()['result'],'LCC-ICCG')['result']['amino_acid_sequence']
    turn=advance_clarification(s,sequence,'SEQUENCE')
    assert turn['triage_outcome']=='IDENTIFIED_SEQUENCE'
    assert turn['state']['confirmed_labels'][1] is None
    assert turn['state']['pending_candidate'] is None
    assert turn['state']['rejected_labels'][1]==['LCC-ICCG']
    assert_context(turn['state'])


def test_two_unresolved_names_are_clarified_one_question_at_a_time():
    q='Should I use DP 038 or LCC ICCG? My device supports 40C and 60C, pH <=7.'
    s=start_clarification(q)['state']
    assert s['pending_candidate']=='DP038' and s['pending_index']==0
    s=advance_clarification(s,'Yes')['state']
    assert s['confirmed_labels']==['DP038',None]
    assert s['pending_candidate']=='LCC-ICCG' and s['pending_index']==1
    assert s['pending_question'].count('?')==1
    s=advance_clarification(s,'No')['state']
    assert s['confirmed_labels']==['DP038',None] and s['pending_candidate'] is None
    assert s['temperatures_c']==[40,60] and s['maximum_ph']==7


def test_temperature_mentions_and_positive_above_phrase_are_not_inferred_as_limits():
    q=QUESTION.replace('but I cannot','but not 80C; I cannot')
    assert start_clarification(q)['state']['temperatures_c']==[40,60]
    with pytest.raises(ValueError):
        start_clarification(QUESTION.replace('cannot operate above','can operate above'))
    with pytest.raises(ValueError):
        start_clarification(QUESTION.replace('40C and 60C','40C as well as 60C'))
