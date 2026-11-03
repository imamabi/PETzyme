import json
from test_evidence import A, B, observations, documented
from zymecycle.tools import describe_tools, invoke_tool


def test_all_four_tools_discoverable_and_composable(observations):
    names = {tool['name'] for tool in describe_tools()['result']}
    assert {'assess_evidence', 'check_comparability', 'investigate_conditions', 'identify_evidence_gaps'} <= names
    assessment = invoke_tool('assess_evidence', {'observations': observations, 'expected_conditions': [A, B]})
    assert assessment['status'] == 'SUCCESS'
    comparison = invoke_tool('check_comparability', {'observations': [observations[0], observations[3]]})
    assert comparison['result']['category'] == 'LIMITED_COMPARISON'
    investigation = invoke_tool('investigate_conditions', {
        'observations': observations, 'condition_a': A, 'condition_b': B, 'enzyme_identifier': 'E2'})
    assert investigation['status'] == 'PARTIAL'
    gaps = invoke_tool('identify_evidence_gaps', {'investigation': investigation['result']})
    assert gaps['result'][-1]['code'] == 'MEASURE_REQUESTED_CONDITION'
    json.dumps(gaps, allow_nan=False)


def test_optional_documented_metadata_schema(observations):
    result = invoke_tool('check_comparability', {'observations': observations[:2],
                                                'metadata': documented(observations[:2]),
                                                'varying_variable': 'temperature_c'})
    assert result['result']['category'] == 'COMPARABLE_WITHIN_DOCUMENTED_GROUP'
    empty_details = documented(observations[:2])
    for record in empty_details: record['details'] = {}
    result = invoke_tool('assess_evidence', {'observations': observations[:2], 'metadata': empty_details})
    assert result['status'] == 'SUCCESS'


def test_reject_invalid_new_tool_json(observations):
    assert invoke_tool('assess_evidence', {'observations': observations, 'metadata': [{'invented': True}]})['status'] == 'ERROR'
    assert invoke_tool('check_comparability', {'observations': observations, 'varying_variable': 'pet_form'})['status'] == 'ERROR'
