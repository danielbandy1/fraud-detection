"""Independent reconstruction from saved predictions. AUTHORED BY ORION."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
R=Path(__file__).resolve().parent
m=json.loads((R/'FULL_EVAL_ORION_20261024.json').read_text())
v=pd.read_parquet(R/'full_eval_results/validation_predictions.parquet');t=pd.read_parquet(R/'full_eval_results/test_predictions.parquet')
y=v.isfraud.to_numpy();p=v.probability.to_numpy();o=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(o)-1];tp=np.cumsum(y[o])[ends];n=ends+1;f1=2*tp/(y.sum()+n);i=np.flatnonzero(f1==f1.max())[0];threshold=float(p[o][ends[i]])
assert threshold==m['threshold_selection']['threshold']
assert abs(float(f1[i])-m['threshold_selection']['validation_f1'])<1e-12
assert v.step.min()==501 and v.step.max()==600 and t.step.min()==601 and t.step.max()==743
for label,thr in [('test',threshold),('test_fixed_0_5',.5)]:
 y=t.isfraud.to_numpy();p=t.probability.to_numpy();q=m[label];cm=confusion_matrix(y,p>=thr,labels=[0,1]);assert cm.tolist()==q['confusion_matrix_true_rows_0_1_pred_columns_0_1']
 assert abs(average_precision_score(y,p)-q['average_precision'])<1e-12
 assert abs(roc_auc_score(y,p)-q['auroc'])<1e-12
 ranks=rankdata(p,method='average');P=y.sum();N=len(y)-P;auc=(ranks[y==1].sum()-P*(P+1)/2)/(P*N);assert abs(auc-q['auroc'])<1e-12
 o=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[o])!=0),len(o)-1];cp=np.cumsum(y[o])[ends];ap=np.sum(cp/(ends+1)*np.diff(np.r_[0,cp]))/P;assert abs(ap-q['average_precision'])<1e-12
for name,hashval in m['files'].items():assert hashlib.sha256((R/'full_eval_results'/name).read_bytes()).hexdigest()==hashval
assert m['resource']['actual_nice']>=15 and m['resource']['peak_rss_mib']<6144 and m['resource']['minimum_memavailable_gib']>=3
assert m['raw_rows']==6362620 and m['eligible_rows']==2770409
for name in ['primary_bootstrap','secondary_bootstrap']:
 for metric in ['ap_95ci','auroc_95ci']:
  lo,hi=m[name][metric];assert 0<=lo<=hi<=1
print('PASS: validation-only threshold independently reconstructed; test AP/AUROC/tied ranks/confusion/split/files/resources match.')
