"""Check repaired feature invariants and trace historical PaySim artifacts.
AUTHORED BY ORION. Synthetic unit fixtures do NOT reproduce PaySim performance.
"""
from pathlib import Path
import hashlib,json,numpy as np,pandas as pd
from src.features import add_graph_features
ROOT=Path(__file__).resolve().parent

def main():
    rows=pd.DataFrame({'step':[2,1,2,3],'nameorig':['A','A','A','B'],'namedest':['X','X','Y','X']},index=[10,11,12,13])
    result=add_graph_features(rows)
    assert result.index.tolist()==rows.index.tolist()
    assert result.orig_tx_count.tolist()==[1,0,1,0]
    assert result.dest_recv_count.tolist()==[1,0,0,2]
    assert result.orig_unique_dest.tolist()==[1,0,1,0]
    future=pd.concat([rows,pd.DataFrame({'step':[999],'nameorig':['A'],'namedest':['X']},index=[14])])
    pd.testing.assert_frame_equal(result,add_graph_features(future).loc[rows.index])
    legacy=json.loads((ROOT/'models/results.json').read_text());m=legacy['lgbm'];d=legacy['data']
    assert m['tp']+m['fn']==d['n_fraud']==8213
    assert sum(m[k] for k in ['tp','fp','fn','tn'])==d['n_rows']==2770409
    assert np.isclose(m['recall'],m['tp']/(m['tp']+m['fn']))
    assert round(m['auprc'],3)==.880 and round(m['auroc'],3)==.989
    report={'artifact_sha256':hashlib.sha256((ROOT/'models/results.json').read_bytes()).hexdigest(),'legacy_artifact':legacy,'historical_metrics_recomputed_from_predictions':False,'legacy_validation_deployment_valid':False,'reason':'Original full-window graph counts leak future rows; random CV and threshold selected on evaluated labels. Historical result retained for traceability, not endorsed as deployment performance.','repaired_history_tests':'earlier-step only, same-step isolation, future invariance, original row order','fixture':'4 hand-constructed synthetic rows, not PaySim training data','new_model_performance':None}
    out=ROOT/'portfolio_results';out.mkdir(exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
