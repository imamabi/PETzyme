"""Independent acceptance inputs against real local triage and Streamlit state."""
import json
import pytest
from zymecycle.clarification import start_clarification, advance_clarification
from zymecycle.observations import load_nrel, select_nrel_protein
from test_ui_clarification import app, begin, reply

EXACT_QUESTION = ('Should I use ESM050 or LCCMC?\n'
                  'I have 40C and 60C available.\nMy maximum pH is 7.')
AMBIGUOUS = 'ALINPRGRSTTPIATRVQRQRPGLIKJKKM'


def test_exact_acceptance_question_retains_upper_bound_not_exact_ph():
    state = start_clarification(EXACT_QUESTION)['state']
    assert state['confirmed_labels'] == ['ESM050', None]
    assert state['requested_labels'] == ['ESM050', 'LCCMC']
    assert state['temperatures_c'] == [40, 60]
    assert state['maximum_ph'] == 7 and 'ph' not in state
    assert state['pending_candidate'] is None
    assert 'FASTA' in state['pending_question']
    assert 'LCC-ICCG' not in json.dumps(state)


def test_complete_rejection_sentence_records_candidate_and_prevents_repeat():
    state = start_clarification(EXACT_QUESTION.replace('LCCMC', 'LCC ICCG'))['state']
    assert state['pending_candidate'] == 'LCC-ICCG'
    rejected = advance_clarification(state, 'No, that is not my enzyme.')['state']
    assert rejected['rejected_labels'] == [[], ['LCC-ICCG']]
    assert state['rejected_labels'] == [[], []]
    repeated = advance_clarification(rejected, 'LCC ICCG')['state']
    assert repeated['pending_candidate'] is None
    assert repeated['temperatures_c'] == [40, 60] and repeated['maximum_ph'] == 7
    assert repeated['requested_labels'] == ['ESM050', 'LCC ICCG']
    assert 'FASTA' in repeated['pending_question']


def test_actual_ambiguous_sequence_length_and_context(monkeypatch):
    import zymecycle.clarification as module
    actual = module.triage_protein
    calls = []
    def observed(*args, **kwargs):
        result = actual(*args, **kwargs)
        if kwargs.get('sequence_input') is not None:
            calls.append(result['result'])
        return result
    state = start_clarification(EXACT_QUESTION)['state']
    monkeypatch.setattr(module, 'triage_protein', observed)
    turn = advance_clarification(state, AMBIGUOUS, 'SEQUENCE')
    assert len(calls) == 1
    assert calls[0]['sequence_length'] == 30
    assert calls[0]['alignment_performed'] is False
    assert calls[0]['issues'] == ['AMBIGUOUS_RESIDUES']
    assert turn['triage_outcome'] == 'UNRESOLVED'
    assert turn['sequence_validation'] == 'AMBIGUOUS_RESIDUES'
    assert '30 residues' in turn['message']
    assert turn['state'] == state
    assert AMBIGUOUS not in json.dumps(turn)


@pytest.mark.parametrize('label', ['DP038', 'TEP021'])
def test_other_real_identifiers_remain_distinct(label):
    state = start_clarification(EXACT_QUESTION.replace('ESM050', label))['state']
    assert state['confirmed_labels'] == [label, None]
    assert state['requested_labels'][0] == label
    assert 'No identity substitution' in state['pending_question']


def test_exact_reference_provenance_is_transient_and_not_construct_identity():
    sequence = select_nrel_protein(load_nrel()['result'], 'DP038')['result']['amino_acid_sequence']
    turn = advance_clarification(start_clarification(EXACT_QUESTION)['state'], sequence, 'SEQUENCE')
    assert turn['triage_outcome'] == 'IDENTIFIED_SEQUENCE'
    assert turn['state']['confirmed_labels'] == ['ESM050', 'DP038']
    assert '10.5281/zenodo.15417757' in turn['message']
    assert '#file=label_data.csv&record=' in turn['message']
    assert 'CSV SHA-256' in turn['message']
    assert 'construct identity and function remain unverified' in turn['message']
    assert 'No winning enzyme' in turn['state']['pending_question']
    assert 'zenodo' not in json.dumps(turn['state'])
    assert sequence not in json.dumps(turn)


def test_real_ui_exact_turns_rejection_sequence_reset_and_independent_context(app):
    at = begin(app, EXACT_QUESTION.replace('LCCMC', 'LCC ICCG'))
    reply(at, 'No, that is not my enzyme.')
    assert at.session_state['clarification_state']['rejected_labels'] == [[], ['LCC-ICCG']]
    reply(at, AMBIGUOUS, 'SEQUENCE')
    assert any('30 residues' in m.value for m in at.markdown)
    assert any('AMBIGUOUS_RESIDUES' in t.value for t in at.text)
    assert AMBIGUOUS not in str(at.session_state)
    sequence = select_nrel_protein(load_nrel()['result'], 'DP038')['result']['amino_acid_sequence']
    reply(at, '>PRIVATE_ACCEPTANCE_HEADER\n' + sequence, 'SEQUENCE')
    assert any('IDENTIFIED_SEQUENCE' in t.value for t in at.text)
    assert any('CSV SHA-256' in m.value and '10.5281/zenodo.15417757' in m.value for m in at.markdown)
    assert sequence not in str(at.session_state)
    assert 'PRIVATE_ACCEPTANCE_HEADER' not in str(at.session_state)
    assert at.session_state['clarification_state']['temperatures_c'] == [40, 60]
    assert at.session_state['clarification_state']['maximum_ph'] == 7
    assert any('No winning enzyme' in i.value for i in at.info)
    at.run()
    assert not any('CSV SHA-256' in m.value for m in at.markdown)
    at.button(key='clarify_reset').click().run()
    assert 'clarification_state' not in at.session_state
    begin(at, 'Should I use DP038 or TEP021? I have 35C and 55C available. My maximum pH is 6.8.')
    state = at.session_state['clarification_state']
    assert state['confirmed_labels'] == ['DP038', 'TEP021']
    assert state['temperatures_c'] == [35, 55] and state['maximum_ph'] == 6.8
    assert state['rejected_labels'] == [[], []]
    assert 'ESM050' not in json.dumps(state)

FINAL_QUESTION = 'ESM050 or LCCMC? My device can maintain 40C and 60C, but pH cannot exceed 7.'


def test_final_acceptance_wording_and_ambiguous_sequence_continuity():
    state = start_clarification(FINAL_QUESTION)['state']
    assert state['requested_labels'] == ['ESM050', 'LCCMC']
    assert state['confirmed_labels'] == ['ESM050', None]
    assert state['pending_candidate'] is None and 'FASTA' in state['pending_question']
    assert state['temperatures_c'] == [40, 60] and state['maximum_ph'] == 7
    assert 'ph' not in state
    turn = advance_clarification(state, AMBIGUOUS, 'SEQUENCE')
    assert turn['state'] == state
    assert turn['triage_outcome'] == 'UNRESOLVED'
    assert turn['sequence_validation'] == 'AMBIGUOUS_RESIDUES'
    assert '30 residues' in turn['message']


@pytest.mark.parametrize('replacement', [
    'pH can exceed 7', 'pH cannot exceed 7; maximum pH 8',
])
def test_new_wording_rejects_missing_or_conflicting_upper_bound(replacement):
    with pytest.raises(ValueError):
        start_clarification(FINAL_QUESTION.replace('pH cannot exceed 7', replacement))


@pytest.mark.parametrize('spelling,label', [('DP 038', 'DP038'), ('TEP 021', 'TEP021')])
def test_source_backed_spelling_candidates_require_confirmation(spelling, label):
    state = start_clarification(FINAL_QUESTION.replace('LCCMC', spelling))['state']
    assert state['pending_candidate'] == label
    assert state['confirmed_labels'][1] is None
    state = advance_clarification(state, 'No, that is not my enzyme.')['state']
    assert state['rejected_labels'][1] == [label]
    state = advance_clarification(state, spelling)['state']
    assert state['pending_candidate'] is None and state['confirmed_labels'][1] is None
    assert state['temperatures_c'] == [40, 60] and state['maximum_ph'] == 7


def test_real_documented_alias_stays_transient_and_source_attributed():
    from zymecycle.pet_retrieval import load_activity_source
    from zymecycle.protein_triage import triage_protein
    snapshot = load_activity_source()['result']
    assert snapshot is not None
    for record in snapshot['records']:
        for alias in record.get('synonyms') or []:
            name = alias.get('name')
            if not isinstance(name, str) or not name or name == record['protein_name']:
                continue
            result = triage_protein(name=name, limit=5)
            candidates = result['result']['candidates']
            match = next((c for c in candidates if c['source_name'] == 'PAZy activity snapshot'
                          and c['identifier'] == str(record['protein_id'])
                          and 'DOCUMENTED_ALIAS' in c['basis']), None)
            if match is not None:
                assert match['source']['doi'] == record['literature_doi']
                assert match['source']['locator']
                assert result['result']['identity_status'] == 'CANDIDATE_MATCH'
                assert result['result']['persistence'] == 'TRANSIENT'
                return
    pytest.fail('No genuine documented alias verified in the pinned local archive')


def test_distinct_valid_identifiers_and_unsupported_typo_are_not_merged():
    from zymecycle.protein_triage import triage_protein
    state = start_clarification(FINAL_QUESTION.replace('ESM050', 'DP038').replace('LCCMC', 'DP039'))['state']
    assert state['confirmed_labels'] == ['DP038', 'DP039']
    assert state['pending_candidate'] is None
    assert triage_protein(name='DP038-misspelled')['result']['identity_status'] == 'UNRESOLVED'


def test_final_wording_real_ui_sequence_fragment_exact_match_and_reset(app):
    at = begin(app, FINAL_QUESTION)
    assert at.session_state['clarification_state']['confirmed_labels'] == ['ESM050', None]
    reply(at, AMBIGUOUS, 'SEQUENCE')
    assert any('AMBIGUOUS_RESIDUES' in t.value for t in at.text)
    assert any('30 residues' in m.value for m in at.markdown)
    sequence = select_nrel_protein(load_nrel()['result'], 'DP038')['result']['amino_acid_sequence']
    reply(at, sequence[25:65], 'SEQUENCE')
    assert any('CANDIDATE_MATCH' in t.value for t in at.text)
    assert at.session_state['clarification_state']['confirmed_labels'][1] is None
    reply(at, '>PRIVATE_FINAL_HEADER\n' + sequence, 'SEQUENCE')
    assert any('IDENTIFIED_SEQUENCE' in t.value for t in at.text)
    state = at.session_state['clarification_state']
    assert state['requested_labels'] == ['ESM050', 'LCCMC']
    assert state['confirmed_labels'] == ['ESM050', 'DP038']
    assert state['temperatures_c'] == [40, 60] and state['maximum_ph'] == 7
    assert any('No winning enzyme' in i.value for i in at.info)
    assert sequence not in str(at.session_state) and 'PRIVATE_FINAL_HEADER' not in str(at.session_state)
    at.button(key='clarify_reset').click().run()
    assert 'clarification_state' not in at.session_state
