"""Independent rank/threshold reconstruction from saved holdout predictions.
AUTHORED BY ORION. Does not import the model-training or evaluation helpers.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent

def main():
    result=json.loads((ROOT/'FORWARD_TIME_EVAL_ORION_20261023.json').read_text())
    file=ROOT/'forward_time_results/test_predictions.parquet'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==result['prediction_sha256']
    pred=pd.read_parquet(file);y=pred.isfraud.to_numpy();p=pred.probability.to_numpy()
    assert pred.step.gt(600).all() and len(pred)==result['split']['test']['rows']
    assert pred.retained_row_position.is_unique
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    cumulative=np.cumsum(ys)[ends]
    added=np.diff(np.r_[0,cumulative])
    ap=float(np.sum((added/y.sum())*(cumulative/(ends+1))))
    ranks=rankdata(p,method='average');npos=int(y.sum());nneg=len(y)-npos
    auc=float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    threshold=result['threshold'];flag=p>=threshold
    tp=int(np.sum(flag&(y==1)));fp=int(np.sum(flag&(y==0)));fn=int(np.sum(~flag&(y==1)));tn=int(np.sum(~flag&(y==0)))
    metrics={'average_precision':ap,'auroc':auc,'base_rate':float(y.mean()),'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'rows':len(y),'predicted_positive':tp+fp}
    for key,value in metrics.items():assert math.isclose(value,result['test'][key],rel_tol=1e-10,abs_tol=1e-12),(key,value,result['test'][key])
    assert result['whole_source_used'] is True and result['subsample'] is None
    assert result['resource']['peak_rss_mb']<6144
    print(json.dumps({'authored_by':'ORION','verification':'PASS','metrics_reconstructed_independently':metrics},indent=2))
if __name__=='__main__':main()
