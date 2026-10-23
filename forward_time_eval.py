"""Resource-bounded, prespecified full PaySim forward-time experiment.
AUTHORED BY ORION. Run with nice; no network or external submission in this script.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import resource
# Bound address space more strictly than the requested6GB RSS ceiling.
resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
from pathlib import Path
import time,hashlib,json,zipfile,gc,sys
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_score,recall_score
from src.features import clean,build_features,FEATURE_COLS
ROOT=Path(__file__).resolve().parent
SHA='f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d'
MODEL_PARAMS={'objective':'binary','metric':'average_precision','learning_rate':.05,'num_leaves':31,'min_child_samples':50,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':5,'verbosity':-1,'seed':42,'num_threads':2,'deterministic':True,'force_col_wise':True}

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def main():
    start=time.monotonic();phases=[]
    def phase(name):
        rec={'phase':name,'seconds':time.monotonic()-start,'peak_rss_mb':rss()};phases.append(rec);print(json.dumps(rec),flush=True)
    archive=ROOT/'data/raw/paysim_source.zip';h=hashlib.sha256()
    with archive.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    if h.hexdigest()!=SHA:raise ValueError('Wrong public source archive')
    retained=[];raw_rows=0;raw_fraud=0;raw_min=10**9;raw_max=-1
    types={'step':'int16','type':'category','amount':'float64','nameOrig':'string','oldbalanceOrg':'float64','newbalanceOrig':'float64','nameDest':'string','oldbalanceDest':'float64','newbalanceDest':'float64','isFraud':'int8','isFlaggedFraud':'int8'}
    with zipfile.ZipFile(archive) as z,z.open('PS_20174392719_1491204439457_log.csv') as source:
        for chunk in pd.read_csv(source,dtype=types,chunksize=200000):
            raw_rows+=len(chunk);raw_fraud+=int(chunk.isFraud.sum());raw_min=min(raw_min,int(chunk.step.min()));raw_max=max(raw_max,int(chunk.step.max()))
            selected=chunk[chunk.type.isin(['CASH_OUT','TRANSFER'])].copy()
            # Type categories vary by chunk; use a common two-value category universe.
            selected['type']=selected['type'].astype(pd.CategoricalDtype(['CASH_OUT','TRANSFER']))
            retained.append(selected)
    del chunk,selected
    frame=pd.concat(retained,ignore_index=True);del retained;gc.collect()
    frame.columns=[c.lower() for c in frame.columns]
    before=len(frame);frame=clean(frame);removed_duplicates=before-len(frame)
    assert raw_rows==6362620 and int(frame.isfraud.sum())==8213
    phase('source_read_and_population_fixed')
    counts={}
    masks={'train':frame.step.le(500),'validation':frame.step.between(501,600),'test':frame.step.gt(600)}
    for key,mask in masks.items():
        part=frame.loc[mask];counts[key]={'rows':len(part),'fraud':int(part.isfraud.sum()),'base_rate':float(part.isfraud.mean()),'min_step':int(part.step.min()),'max_step':int(part.step.max())}
    assert all(c['rows']>0 and 0<c['fraud']<c['rows'] for c in counts.values())
    # Account IDs are non-feature equivalence classes; integerization preserves exact equality.
    for c in ['nameorig','namedest']:frame[c]=pd.factorize(frame[c],sort=False)[0].astype(np.int32)
    y=frame.isfraud.to_numpy(dtype=np.int8);steps=frame.step.to_numpy(copy=True)
    engineered=build_features(frame);del frame;gc.collect()
    X=engineered[FEATURE_COLS].astype(np.float64);del engineered;gc.collect()
    assert np.isfinite(X.to_numpy()).all()
    phase('strict_prior_step_features_built')
    tr,va,te=(np.flatnonzero(masks[k].to_numpy()) for k in ['train','validation','test'])
    del masks;gc.collect()
    params={**MODEL_PARAMS,'scale_pos_weight':counts['train']['rows']/counts['train']['fraud']-1}
    # ONLY training and validation arrays are supplied to LightGBM. Test is scored once after selection.
    training=lgb.Dataset(X.iloc[tr],label=y[tr],free_raw_data=True)
    validation=lgb.Dataset(X.iloc[va],label=y[va],reference=training,free_raw_data=True)
    model=lgb.train(params,training,num_boost_round=200,valid_sets=[validation],valid_names=['validation'],callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(period=50)])
    best_iteration=int(model.best_iteration or model.current_iteration())
    phase('validation_selected_model_fitted')
    threshold=.5
    probs=model.predict(X.iloc[te],num_iteration=best_iteration);truth=y[te]
    assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
    pred=(probs>=threshold).astype(np.int8)
    tn,fp,fn,tp=(int(v) for v in confusion_matrix(truth,pred,labels=[0,1]).ravel())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn)
    assert abs(precision-precision_score(truth,pred,zero_division=0))<1e-12
    assert abs(recall-recall_score(truth,pred))<1e-12
    assert tp+fn==counts['test']['fraud'] and tp+fp+tn+fn==len(te)
    out=ROOT/'forward_time_results';out.mkdir(exist_ok=True)
    # Holdout predictions permit independent metric reconstruction; IDs are source positions, not accounts.
    pd.DataFrame({'retained_row_position':te,'step':steps[te],'isfraud':truth,'probability':probs}).to_parquet(out/'test_predictions.parquet',index=False)
    model.save_model(str(out/'model.txt'),num_iteration=best_iteration)
    legacy=json.loads((ROOT/'models/results.json').read_text())
    phase('test_scored_once_and_predictions_saved')
    report={'authored_by':'ORION','source_url':'https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1','source_zip_sha256':SHA,'plan_sha256':hashlib.sha256((ROOT/'FORWARD_TIME_PLAN_ORION_20261023.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((ROOT/'src/features.py').read_bytes()).hexdigest(),'raw_rows':raw_rows,'raw_fraud':raw_fraud,'source_min_step':raw_min,'source_max_step':raw_max,'eligible_rows_before_dedup':before,'retained_rows':len(X),'exact_duplicates_removed':removed_duplicates,'whole_source_used':True,'subsample':None,'feature_names':FEATURE_COLS,'split':counts,'params':params,'max_boost_rounds':200,'validation_early_stopping_rounds':20,'best_iteration':best_iteration,'threshold':threshold,'test':{'average_precision':float(average_precision_score(truth,probs)),'auroc':float(roc_auc_score(truth,probs)),'base_rate':counts['test']['base_rate'],'precision':precision,'recall':recall,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'rows':len(te),'predicted_positive':tp+fp},'legacy_random_cv':legacy['lgbm'],'evaluation_scope':'Forward-time post-transaction simulated fraud detection; same earlier-step unlabelled history available across windows, no same-step lookahead; no external deployment validation','resource':{'nice_requested':10,'actual_nice':os.nice(0),'threads':2,'address_space_limit_bytes':6*1024**3,'wall_seconds':time.monotonic()-start,'peak_rss_mb':rss(),'phases':phases},'prediction_sha256':hashlib.sha256((out/'test_predictions.parquet').read_bytes()).hexdigest()}
    assert report['resource']['peak_rss_mb']<6144
    (ROOT/'FORWARD_TIME_EVAL_ORION_20261023.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
