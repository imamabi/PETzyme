import json
import sys

from test_activity_snapshot import records
from test_cross_overlap import inputs
from scripts import audit_build07a
from petzyme.results import outcome


def test_manual_entry_defaults_to_offline_aggregate_only(inputs,monkeypatch,capsys):
    snapshot,nrel=inputs
    monkeypatch.setattr(sys,'argv',['audit_build07a'])
    monkeypatch.setattr(audit_build07a,'load_pazy_activity_snapshot',lambda:outcome('SUCCESS',snapshot,[],[],None))
    monkeypatch.setattr(audit_build07a,'load_nrel',lambda:outcome('SUCCESS',nrel,[],[],None))
    def forbidden(*a,**k):raise AssertionError('Default CLI attempted live verification')
    monkeypatch.setattr(audit_build07a,'verify_snapshot_identities',forbidden)
    audit_build07a.main()
    output=capsys.readouterr().out
    result=json.loads(output)
    assert result['live_http']['requests']==0 and result['live_verification_status']=='NOT_REQUESTED'
    assert 'candidate_pairs' not in result['overlap'] and 'SYNTHETIC' not in output
    assert 'amino_acid_sequence' not in output and 'sequence_sha256' not in output


def test_missing_snapshot_fails_without_download(inputs,monkeypatch,capsys):
    _,nrel=inputs
    monkeypatch.setattr(sys,'argv',['audit_build07a'])
    monkeypatch.setattr(audit_build07a,'load_pazy_activity_snapshot',lambda:outcome('ERROR',None,[],[],None))
    monkeypatch.setattr(audit_build07a,'load_nrel',lambda:outcome('SUCCESS',nrel,[],[],None))
    audit_build07a.main()
    assert json.loads(capsys.readouterr().out)['status']=='ERROR'
