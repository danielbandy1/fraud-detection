"""Fake-model API history indicators, no server or model files. AUTHORED BY ORION."""
import numpy as np
import pytest
from api import serve


def row(**changes):
    r=dict(step=5,type='TRANSFER',amount=100.,nameorig='A',namedest='X',oldbalanceorg=100.,newbalanceorig=0.,oldbalancedest=0.,newbalancedest=100.)
    r.update(changes);return r

class Model:
    def predict(self,x,**kwargs):
        return np.array([.3]*len(x))

@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(serve,'_load',lambda:None)
    monkeypatch.setattr(serve,'_model',Model())
    monkeypatch.setattr(serve,'_explainer',None)

@pytest.mark.parametrize('history,expected',[
    ([],False),([row(step=6)],False),([row(step=5)],False),
    ([row(step=2,nameorig='B',namedest='Y')],False),
    ([row(step=2,namedest='Y')],True),
    ([row(step=2,nameorig='B')],True),
])
def test_indicator_requires_usable_prior_context(history,expected):
    result=serve._score_df(history+[row()])
    assert result.graph_features_available is expected

def test_indicator_uses_selected_target_not_last_row():
    result=serve._score_df([row(step=1),row(step=2)],target_idx=0)
    assert result.graph_features_available is False

def test_window_http_contract():
    from fastapi.testclient import TestClient
    with TestClient(serve.app) as client:
        future=client.post('/score/window',json=[row(step=6),row()])
        prior=client.post('/score/window',json=[row(step=2,nameorig='B'),row()])
        single=client.post('/score',json=row())
        assert future.status_code==prior.status_code==single.status_code==200
        assert future.json()['graph_features_available'] is False
        assert prior.json()['graph_features_available'] is True
        assert single.json()['graph_features_available'] is False
        assert prior.json()['fraud_probability']==.3
