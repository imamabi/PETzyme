import io
import json
import os
import sys
import pytest
from zymecycle import ui_live
from zymecycle.ui_adapter import EXAMPLES, run_offline


class FakeProcess:
    def __init__(self, output=b'', code=0):
        read, write = os.pipe()
        os.write(write, output)
        os.close(write)
        self.stdin, self.stdout = io.BytesIO(), os.fdopen(read,'rb')
        self.code = code
    def poll(self): return self.code
    def wait(self, **kwargs): return self.code


def test_no_launch_or_keychain_without_explicit_authorization(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No authorized launch')
    monkeypatch.setattr(ui_live.subprocess, 'Popen', forbidden)
    assert ui_live.run_keychain_investigation(EXAMPLES['Temperature evidence'])['status'] == 'ERROR'
    assert ui_live.run_keychain_investigation('>PRIVATE_SEQUENCE', authorized=True)['status'] == 'ERROR'


def test_live_launch_fixed_command_environment_credential_and_public_protocol(monkeypatch):
    safe = run_offline(EXAMPLES['Temperature evidence'])
    update = safe['result']['progress'][0]
    output = (json.dumps({'update':update})+'\n'+json.dumps({'result':safe})+'\n').encode()
    calls = []
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs['env']['MODEL_API_KEY'] == 'synthetic-parent-credential'
        assert kwargs['stderr'] is ui_live.subprocess.DEVNULL
        return FakeProcess(output)
    monkeypatch.setenv('MODEL_API_KEY', 'synthetic-parent-credential')
    monkeypatch.setattr(ui_live.subprocess, 'Popen', fake)
    updates = []
    result = ui_live.run_keychain_investigation(EXAMPLES['Temperature evidence'], authorized=True, on_update=updates.append)
    assert result == safe and updates == [update]
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable,'-m','scripts.launch_meta','--authorize-paid-meta','--',sys.executable,'-m','zymecycle.ui_live','--live-worker']
    assert 'synthetic-parent-credential' not in json.dumps(result)


def test_windows_pipe_path_preserves_public_worker_protocol(monkeypatch):
    safe = run_offline(EXAMPLES['Temperature evidence'])
    output = (json.dumps({'result': safe}) + '\n').encode()
    monkeypatch.setattr(ui_live.subprocess, 'Popen', lambda *args, **kwargs: FakeProcess(output))
    monkeypatch.setattr(ui_live.sys, 'platform', 'win32')
    result = ui_live.run_keychain_investigation(EXAMPLES['Temperature evidence'], authorized=True)
    assert result == safe


def test_unified_app_routes_environment_credential_without_exposing_it(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    calls = []
    def fake_live(question, **kwargs):
        calls.append(question)
        return run_offline(question)
    monkeypatch.delenv('PETZYME_OFFLINE', raising=False)
    monkeypatch.setenv('MODEL_API_KEY', 'synthetic-app-only-key')
    monkeypatch.setattr('scripts.launch_meta.keychain_item_present', lambda: False)
    monkeypatch.setattr(ui_live, 'run_keychain_investigation', fake_live)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'), default_timeout=60).run()
    app.text_area(key='unified_question').input('Investigate ESM050 PET experimental evidence.')
    app.button(key='FormSubmitter:ask_zymecycle-Ask PETzyme').click().run(timeout=60)
    assert not app.exception and len(calls) == 1
    assert 'synthetic-app-only-key' not in str(app)


@pytest.mark.parametrize('output,code', [(b'',1),(b'not json\n',0),(b'{"result":"invalid"}\n',0),
    (b'{"reasoning":"private"}\n',0),(b'{}\n'*10,0)])
def test_child_failures_sanitized_without_retries(monkeypatch, output, code):
    calls=[]
    def fake(*args,**kwargs): calls.append(1); return FakeProcess(output,code)
    monkeypatch.setattr(ui_live.subprocess,'Popen',fake)
    result = ui_live.run_keychain_investigation(EXAMPLES['Temperature evidence'],authorized=True)
    assert result['status']=='ERROR' and len(calls)==1
    assert 'private' not in json.dumps(result) and 'No retry' in result['message']


def test_worker_never_constructs_live_client_without_key(monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    monkeypatch.setattr(sys,'argv',['ui_live','--live-worker'])
    assert ui_live.worker_main()==1


def test_cohort_worker_rejects_stale_code_before_constructing_provider(monkeypatch, capsys):
    monkeypatch.setenv('MODEL_API_KEY', 'synthetic-child-test-only')
    monkeypatch.setattr(sys, 'argv', ['ui_live', '--live-worker'])
    payload = {'question': 'Which of those were tested at 60C?',
               'cohort_origin': 'What are the five enzymes with the best recorded PET-degradation responses?',
               'cohort_fingerprint': 'stale-code'}
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    monkeypatch.setattr('zymecycle.meta.MetaModelClient', lambda **kwargs: pytest.fail('provider must not be constructed'))
    assert ui_live.worker_main() == 0
    output = capsys.readouterr().out
    assert 'code differ' in output and 'synthetic-child-test-only' not in output


def test_worker_deadline_terminates_the_child_group_without_retry(monkeypatch):
    process=FakeProcess()
    process.pid=999999
    process.poll=lambda:None
    kills=[]
    monkeypatch.setattr(ui_live.subprocess,'Popen',lambda *args,**kwargs:process)
    times=iter([0,1500])
    monkeypatch.setattr(ui_live.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(ui_live.os,'killpg',lambda pid,sig:kills.append((pid,sig)))
    result=ui_live.run_keychain_investigation(EXAMPLES['Temperature evidence'],authorized=True)
    assert result['status']=='ERROR' and kills==[(999999,ui_live.signal.SIGTERM)]


def test_worker_mocked_backend_only_and_output_is_redacted(monkeypatch, capsys):
    from scripts.demo_build03 import ScenarioMeta
    from zymecycle.meta import MetaModelClient
    import httpx
    monkeypatch.setenv('MODEL_API_KEY','synthetic-child-test-only')
    monkeypatch.setattr(sys,'argv',['ui_live','--live-worker'])
    monkeypatch.setattr(sys,'stdin',io.TextIOWrapper(io.BytesIO(json.dumps({'question':EXAMPLES['Temperature evidence']}).encode())))
    backend = MetaModelClient(transport=httpx.MockTransport(ScenarioMeta('B').respond))
    monkeypatch.setattr('zymecycle.meta.MetaModelClient',lambda **kwargs:backend)
    assert ui_live.worker_main()==0
    output = capsys.readouterr().out
    assert 'synthetic-child-test-only' not in output and '"rows"' not in output and '"vector"' not in output
    messages=[json.loads(line) for line in output.splitlines()]
    assert len(messages)==4 and messages[-1]['result']['status']=='SUCCESS'


def test_automatic_worker_uses_no_estimated_spending_ceiling(monkeypatch, capsys):
    from scripts.demo_build03 import ScenarioMeta
    from zymecycle.meta import MetaModelClient
    import httpx

    monkeypatch.setenv('MODEL_API_KEY', 'synthetic-child-test-only')
    monkeypatch.setattr(sys, 'argv', ['ui_live', '--live-worker'])
    payload = {'question': EXAMPLES['Temperature evidence'], 'automatic_runtime': True,
               'completion_tokens': 50000}
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    backend = MetaModelClient(transport=httpx.MockTransport(ScenarioMeta('B').respond), completion_tokens=50000,
                              automatic_runtime=True)
    def factory(**kwargs):
        assert kwargs['max_estimated_cost_usd'] is None and kwargs['completion_tokens'] == 50000
        assert kwargs['automatic_runtime'] is True
        return backend
    monkeypatch.setattr('zymecycle.meta.MetaModelClient', factory)
    assert ui_live.worker_main() == 0
    output = capsys.readouterr().out
    assert 'synthetic-child-test-only' not in output
    assert json.loads(output.splitlines()[-1])['result']['status'] == 'SUCCESS'
