"""Explicitly scripted MOCK Meta decisions; real scientific dispatcher execution.

No live Meta option exists in this demonstration. PAZy is optionally live, never saved.
"""
from __future__ import annotations

import argparse
import json
import httpx
from petzyme.agent import run_investigation
from petzyme.agent_types import ResearchReport
from petzyme.contracts import ToolResult
from petzyme.agent_reporting import format_research_report
from petzyme.meta import MetaModelClient

QUESTIONS = {
    'A': 'For crystalline PET powder at 40°C and pH 6.5, which enzymes have the highest recorded NREL responses, and how reliable is the evidence?',
    'B': 'Does ESM050 perform better at 60°C than 40°C at pH 6.5 using crystalline PET powder?',
    'C': 'What evidence exists for ESM050 at 50°C and pH 8, and what information would we need to answer this question?',
    'D': 'Find PET-associated enzyme information and scientific references using PAZy.',
}
CONDITION = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}


class ScenarioMeta:
    """HTTP response fixture only; not a production reasoning implementation."""
    def __init__(self, scenario: str) -> None:
        """Flow: Receives scenario str from run_scenario(); returns None after configuring respond() test decisions."""
        self.scenario = scenario
        self.requests: list[dict] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Flow: Receives test httpx.Request from MetaModelClient.complete(); returns mocked httpx.Response for the demonstration's next decision."""
        body = json.loads(request.content)
        self.requests.append(body)
        number = len(self.requests)
        prefix = self.scenario.lower()
        tool_outputs = [json.loads(message['content']) for message in body['messages'] if message['role'] == 'tool']
        if self.scenario == 'A':
            plan = [('load_nrel', {}), ('parse_nrel', {'table_ref': 'a1:table'}),
                    ('filter_observations', {'observations_ref': 'a2:observations', **CONDITION}),
                    ('summarize_condition', {'observations_ref': 'a3:observations', **CONDITION}),
                    ('assess_evidence', {'observations_ref': 'a3:observations', 'expected_conditions': [CONDITION]})]
        elif self.scenario in ('B', 'C'):
            a = CONDITION if self.scenario == 'B' else {**CONDITION, 'temperature_c': 50, 'ph': 8}
            b = {**CONDITION, 'temperature_c': 60} if self.scenario == 'B' else {**a, 'pet_form': 'AMORPHOUS_FILM'}
            plan = [('load_nrel', {}), ('parse_nrel', {'table_ref': f'{prefix}1:table'}),
                    ('investigate_conditions', {'observations_ref': f'{prefix}2:observations', 'condition_a': a,
                                                'condition_b': b, 'enzyme_identifier': 'ESM050'})]
            if self.scenario == 'C': plan.append(('identify_evidence_gaps', {'investigation_ref': 'c3:investigation'}))
        else:
            plan = [('search_pazy_pet', {'limit': 1})]
            if tool_outputs and tool_outputs[0]['preview'].get('proteins'):
                plan.append(('get_pazy_metadata', {'protein_id': tool_outputs[0]['preview']['proteins'][0]['id']}))
            if len(tool_outputs) > 1 and tool_outputs[1]['refs'].get('protein'):
                plan.append(('get_associated_literature', {'protein_ref': tool_outputs[1]['refs']['protein']}))
        if number <= len(plan):
            name, arguments = plan[number - 1]
            message = {'role': 'assistant', 'content': None, 'refusal': None, 'tool_calls': [
                {'id': f'{prefix}{number}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
            reason = 'tool_calls'
        else:
            outputs = tool_outputs[3:] if self.scenario == 'A' else tool_outputs[1:] if self.scenario == 'D' else tool_outputs
            ids = [fact['id'] for output in outputs for fact in output['facts']]
            message = {'role': 'assistant', 'content': json.dumps({'fact_ids': ids}), 'refusal': None}
            reason = 'stop'
        return httpx.Response(200, json={'choices': [{'index': 0, 'message': message, 'finish_reason': reason}]})


def run_scenario(scenario: str, csv_path: str = 'data/nrel/label_data.csv') -> ToolResult[ResearchReport]:
    """Flow: Receives external scenario str and trusted CSV path str; returns ToolResult[ResearchReport] from run_investigation() for main() public rendering."""
    scripted = ScenarioMeta(scenario)
    backend = MetaModelClient(transport=httpx.MockTransport(scripted.respond))
    return run_investigation(QUESTIONS[scenario], backend, csv_path=csv_path)


def main(argv: list[str] | None = None) -> int:
    """Flow: Receives external CLI arguments list[str] | None; returns exit status int after privacy-safe format_research_report() output."""
    parser = argparse.ArgumentParser(description='Mock Meta decisions with real ZymeCycle scientific tools')
    parser.add_argument('--scenario', choices=QUESTIONS, default='A')
    parser.add_argument('--csv', default='data/nrel/label_data.csv')
    args = parser.parse_args(argv)
    result = run_scenario(args.scenario, args.csv)
    if result['result'] is None: return 1
    print(format_research_report(result['result']))
    return 0 if result['status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
