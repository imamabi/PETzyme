import pytest
from petzyme.agent_request import interpret_request, select_tool_catalog


def test_multistudy_comparison_exposes_existing_record_comparison_tools():
    question = 'Compare NREL and PAZy evidence for ESM050 at 40C and 60C, pH 6.5, crystalline powder.'
    names = {tool['function']['name'] for tool in select_tool_catalog(interpret_request(question))}
    assert {'investigate_pet_evidence', 'load_nrel', 'parse_nrel', 'investigate_conditions', 'check_comparability'} <= names


def test_unnamed_pet_experiment_evidence_uses_bounded_multistudy_path():
    question = 'What evidence supports PET hydrolase experiments on amorphous PET films at pH 8, and which missing controls would limit interpretation?'
    request = interpret_request(question)
    names = {tool['function']['name'] for tool in select_tool_catalog(request)}
    assert request['sources'] == ['NREL', 'PAZy'] and 'multi_study' in request['objectives']
    assert 'investigate_pet_evidence' in names
    assert request['ph_values'] == [8.0] and request['pet_forms'] == ['AMORPHOUS_FILM']


def test_question_context_and_relevance():
    request = interpret_request('Does ESM050 perform better at 60°C than 40°C at pH 6.5 using crystalline PET powder?')
    assert request['temperatures'] == [60, 40] and request['ph_values'] == [6.5]
    assert request['enzyme_identifiers'] == ['ESM050']
    tools = select_tool_catalog(request)
    names = {tool['function']['name'] for tool in tools}
    assert 'investigate_conditions' in names and 'search_pazy_pet' not in names
    for tool in tools:
        properties = tool['function']['parameters']['properties']
        assert not {'metadata', 'csv_path', 'observations', 'table'} & properties.keys()


def test_pazy_least_privilege():
    tools = select_tool_catalog(interpret_request('Find PET-associated enzyme information and scientific references using PAZy.'))
    assert {t['function']['name'] for t in tools} == {'search_pazy_pet', 'get_pazy_metadata', 'get_associated_literature'}
    assert all('protein' not in t['function']['parameters']['properties'] for t in tools)


@pytest.mark.parametrize('question', ['', 'x' * 4001, 'Use PANDA', '>fasta\nACDEF', 'ACDEF' * 20, 'key LLM|private'])
def test_reject_unsafe_or_empty(question):
    with pytest.raises(ValueError): interpret_request(question)
