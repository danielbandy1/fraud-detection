"""Synthetic artifact/partition checks for evaluated view. AUTHORED BY ORION."""
import hashlib,json
import pandas as pd
import pytest
from src import evaluation_artifacts as view

@pytest.fixture
def saved(tmp_path,monkeypatch):
    (tmp_path/'full_eval_results').mkdir();files={};parts={}
    for split,start,positions in [('validation',501,[0,1]),('test',601,[2,3])]:
        frame=pd.DataFrame({'retained_row_position':positions,'step':[start,start+1],'isfraud':[0,1],'probability':[.1,.9]})
        p=tmp_path/'full_eval_results'/f'{split}_predictions.parquet';frame.to_parquet(p,index=False)
        files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        parts[split]={'rows':2,'fraud':1,'average_precision':1.,'auroc':1.,'confusion_matrix_true_rows_0_1_pred_columns_0_1':[[1,0],[0,1]]}
    r=dict(parts,files=files);p=tmp_path/'FULL_EVAL_ORION_20261024.json';p.write_text(json.dumps(r))
    monkeypatch.setattr(view,'load_evaluated_fraud_model',lambda root:(None,.5,{'evaluation_report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
    return tmp_path

def test_valid_partitions(saved):
    _,frames,_=view.load_evaluation(saved)
    assert len(frames['test'])==2

def test_tampered_prediction_rejected(saved):
    p=saved/'full_eval_results/test_predictions.parquet';p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):view.load_evaluation(saved)

@pytest.mark.parametrize('change',['overlap','future','probability','metric'])
def test_inconsistent_contents_rejected(saved,change):
    p=saved/'full_eval_results/test_predictions.parquet';frame=pd.read_parquet(p)
    report=saved/'FULL_EVAL_ORION_20261024.json';r=json.loads(report.read_text())
    if change=='overlap':frame['retained_row_position']=[0,1]
    elif change=='future':frame['step']=[900,901]
    elif change=='probability':frame['probability']=[.1,1.1]
    else:r['test']['auroc']=.5
    frame.to_parquet(p,index=False);r['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();report.write_text(json.dumps(r))
    with pytest.raises(ValueError):view.load_evaluation(saved)

def test_validation_scenario_exact_cost():
    f=pd.DataFrame({'isfraud':[1,0,0],'probability':[.1,.9,.9]})
    r=view.validation_cost_scenario(f,1,2)
    assert r['cost']==1 and r['fp']==r['tp']==0 and r['fn']==1
