"""Hash/order/threshold fail-closed loader tests. AUTHORED BY ORION."""
import hashlib,json
import pytest
from src import predict
from src.features import FEATURE_COLS

@pytest.fixture
def artifact(tmp_path,monkeypatch):
    (tmp_path/'src').mkdir();(tmp_path/'full_eval_results').mkdir()
    (tmp_path/'src/features.py').write_bytes(b'synthetic feature implementation')
    (tmp_path/'full_eval_results/model.txt').write_bytes(b'synthetic model')
    report={'files':{'model.txt':hashlib.sha256(b'synthetic model').hexdigest()},
      'features_sha256':hashlib.sha256(b'synthetic feature implementation').hexdigest(),
      'features':FEATURE_COLS,'threshold_selection':{'threshold':.7,'test_used_for_selection':False}}
    (tmp_path/'FULL_EVAL_ORION_20261024.json').write_text(json.dumps(report))
    class Fake:
        def __init__(self,**kwargs):assert kwargs['model_str']=='synthetic model'
        def feature_name(self):return FEATURE_COLS
    monkeypatch.setattr(predict.lgb,'Booster',Fake)
    return tmp_path

def test_valid_verified_artifact(artifact):
    _,t,p=predict.load_evaluated_fraud_model(artifact)
    assert t==.7 and len(p['model_sha256'])==64

@pytest.mark.parametrize('file',['src/features.py','full_eval_results/model.txt'])
def test_changed_artifact_rejected(artifact,file):
    (artifact/file).write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):predict.load_evaluated_fraud_model(artifact)

@pytest.mark.parametrize('change',['order','test_labels','threshold'])
def test_changed_contract_rejected(artifact,change):
    p=artifact/'FULL_EVAL_ORION_20261024.json';r=json.loads(p.read_text())
    if change=='order':r['features']=list(reversed(FEATURE_COLS))
    elif change=='test_labels':r['threshold_selection']['test_used_for_selection']=True
    else:r['threshold_selection']['threshold']=float('nan')
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):predict.load_evaluated_fraud_model(artifact)

def test_default_inference_uses_verified_loader(monkeypatch):
    marker=object()
    monkeypatch.setattr(predict,'load_evaluated_fraud_model',lambda:(marker,.7,{}))
    assert predict.load_fraud_model() is marker
