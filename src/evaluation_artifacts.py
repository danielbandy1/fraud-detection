"""Verified aggregate evaluation view inputs. AUTHORED BY ORION."""
from pathlib import Path
import hashlib,io,json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
from src.predict import load_evaluated_fraud_model
from src.models import optimal_threshold


def validation_cost_scenario(frame,cost_fn,cost_fp):
    """Exploratory validation-only cost selection, never accepts test data implicitly."""
    y=frame.isfraud.to_numpy();p=frame.probability.to_numpy()
    t=optimal_threshold(y,p,cost_fn,cost_fp)
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=t,labels=[0,1]).ravel())
    return dict(threshold=t,tn=tn,fp=fp,fn=fn,tp=tp,cost=cost_fn*fn+cost_fp*fp)


def load_evaluation(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    _,threshold,provenance=load_evaluated_fraud_model(root)
    raw=(root/'FULL_EVAL_ORION_20261024.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=provenance['evaluation_report_sha256']:
        raise ValueError('Evaluation report changed during loading')
    report=json.loads(raw);frames={}
    for split,low,high in [('validation',501,600),('test',601,743)]:
        name=split+'_predictions.parquet';content=(root/'full_eval_results'/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=report['files'][name]:raise ValueError('Prediction artifact hash mismatch')
        frame=pd.read_parquet(io.BytesIO(content))
        if list(frame)!=['retained_row_position','step','isfraud','probability']:raise ValueError('Prediction schema mismatch')
        q=report[split];y=frame.isfraud.to_numpy();p=frame.probability.to_numpy()
        if len(frame)!=q['rows'] or not frame.retained_row_position.is_unique or not frame.step.between(low,high).all():raise ValueError('Prediction population mismatch')
        if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid outcomes/probabilities')
        if int(y.sum())!=q['fraud']:raise ValueError('Fraud count mismatch')
        for metric,key in [(average_precision_score,'average_precision'),(roc_auc_score,'auroc')]:
            if abs(metric(y,p)-q[key])>1e-12:raise ValueError('Metric mismatch')
        if confusion_matrix(y,p>=threshold,labels=[0,1]).tolist()!=q['confusion_matrix_true_rows_0_1_pred_columns_0_1']:raise ValueError('Confusion matrix mismatch')
        frames[split]=frame
    if set(frames['validation'].retained_row_position)&set(frames['test'].retained_row_position):raise ValueError('Validation/test overlap')
    return report,frames,provenance
