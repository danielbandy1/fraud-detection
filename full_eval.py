"""Full-source validation-threshold and fixed-model bootstrap evaluation.
AUTHORED BY ORION. See FULL_EVAL_PLAN_ORION_20261024.md before execution.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
import resource
resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
import threading,time,json,hashlib,zipfile,gc,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
samples=[]
def available():
    lines=Path('/proc/meminfo').read_text().splitlines()
    return int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024

def watch():
    while True:
        av=available();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        samples.append((av,rss))
        if av<3*1024**3 or rss>=6*1024**3:
            (ROOT/'FULL_EVAL_RESOURCE_STOP_ORION_20261024.json').write_text(json.dumps({'reason':'resource floor/cap','available_bytes':av,'peak_rss_bytes':rss,'time':time.time()}))
            print('STOP: resource floor/cap',av,rss,flush=True);os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,confusion_matrix
from src.features import clean,build_features,FEATURE_COLS
from forward_time_eval import SHA,MODEL_PARAMS

def hfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def select_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-300)
    best=np.flatnonzero(f1==f1.max())[-1]
    return float(thresholds[best]),float(f1[best])

def measures(y,p,threshold):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=threshold,labels=[0,1]).ravel())
    return {'average_precision':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),
     'threshold':float(threshold),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'confusion_matrix_true_rows_0_1_pred_columns_0_1':[[tn,fp],[fn,tp]],
     'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'rows':len(y),'fraud':int(y.sum()),'base_rate':float(y.mean())}

def weighted_scores(pos,neg):
    # Score bins descending. Correctly includes half of tied positive/negative pairs.
    cp=np.cumsum(pos,axis=-1);cn=np.cumsum(neg,axis=-1);P=cp[...,-1];N=cn[...,-1]
    ap=np.sum(np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=(cp+cn)>0)*pos,axis=-1)/P
    auc=np.sum(pos*(N[...,None]-cn+.5*neg),axis=-1)/(P*N)
    return ap,auc

def bootstrap(y,p,steps,B=2000):
    scores,group=np.unique(-p,return_inverse=True)
    timeline=np.arange(int(steps.min()),int(steps.max())+1);stepidx=steps-timeline[0]
    pos=np.zeros((len(timeline),len(scores)));neg=np.zeros_like(pos)
    np.add.at(pos,(stepidx,group),y);np.add.at(neg,(stepidx,group),1-y)
    rng=np.random.default_rng(20261024);w=np.zeros((B,len(timeline)));L=24
    for i in range(B):
        starts=rng.integers(len(timeline),size=int(np.ceil(len(timeline)/L)))
        selected=((starts[:,None]+np.arange(L))%len(timeline)).ravel()[:len(timeline)]
        w[i]=np.bincount(selected,minlength=len(timeline))
    pb=w@pos;nb=w@neg;valid=(pb.sum(1)>0)&(nb.sum(1)>0)
    # Independent sklearn weighted checks validate ranking/AP ties on actual test scores.
    for i in range(min(10,B)):
        if not valid[i]:continue
        samplew=w[i,stepidx];a,u=weighted_scores(pb[i],nb[i])
        assert abs(a-average_precision_score(y,p,sample_weight=samplew))<1e-10
        assert abs(u-roc_auc_score(y,p,sample_weight=samplew))<1e-10
    def ci(a):return [float(v) for v in np.quantile(a,[.025,.975])]
    ap,auc=weighted_scores(pb[valid],nb[valid]);main={'method':'circular moving-block bootstrap of complete hourly steps','block_steps':L,'timeline_steps':len(timeline),'replicates':B,'valid':int(valid.sum()),'invalid_one_class':int((~valid).sum()),'seed':20261024,'ap_95ci':ci(ap),'auroc_95ci':ci(auc)}
    pp=pos.sum(0);nn=neg.sum(0);rng=np.random.default_rng(20261025)
    pr=rng.multinomial(int(y.sum()),pp/pp.sum(),size=B);nr=rng.multinomial(int((1-y).sum()),nn/nn.sum(),size=B)
    ap2,auc2=weighted_scores(pr,nr)
    return main,{'method':'class-stratified iid row bootstrap, ignores temporal/account dependence','replicates':B,'seed':20261025,'ap_95ci':ci(ap2),'auroc_95ci':ci(auc2)}

def main():
    assert os.nice(0)>=15,'Run with nice -n 15'
    started=time.monotonic();archive=ROOT/'data/raw/paysim_source.zip';assert hfile(archive)==SHA
    parts=[];raw=0;rawfraud=0
    types={'step':'int16','type':'category','amount':'float64','nameOrig':'string','oldbalanceOrg':'float64','newbalanceOrig':'float64','nameDest':'string','oldbalanceDest':'float64','newbalanceDest':'float64','isFraud':'int8','isFlaggedFraud':'int8'}
    with zipfile.ZipFile(archive) as z,z.open('PS_20174392719_1491204439457_log.csv') as f:
        for chunk in pd.read_csv(f,dtype=types,chunksize=200000):
            raw+=len(chunk);rawfraud+=int(chunk.isFraud.sum());s=chunk[chunk.type.isin(['CASH_OUT','TRANSFER'])].copy();s['type']=s['type'].astype(pd.CategoricalDtype(['CASH_OUT','TRANSFER']));parts.append(s)
    del chunk,s
    d=pd.concat(parts,ignore_index=True);del parts;gc.collect();d.columns=[c.lower() for c in d.columns];before=len(d);d=clean(d)
    assert raw==6362620 and int(d.isfraud.sum())==8213
    y=d.isfraud.to_numpy(dtype=np.int8);steps=d.step.to_numpy(copy=True)
    ids={'train':np.flatnonzero(steps<=500),'validation':np.flatnonzero((steps>=501)&(steps<=600)),'test':np.flatnonzero(steps>600)}
    split={k:{'rows':len(i),'fraud':int(y[i].sum()),'base_rate':float(y[i].mean()),'min_step':int(steps[i].min()),'max_step':int(steps[i].max())} for k,i in ids.items()}
    for c in ['nameorig','namedest']:d[c]=pd.factorize(d[c],sort=False)[0].astype(np.int32)
    eng=build_features(d);del d;gc.collect();X=eng[FEATURE_COLS].astype('float64');del eng;gc.collect();assert np.isfinite(X.to_numpy()).all()
    tr,va,te=[ids[k] for k in ['train','validation','test']]
    params={**MODEL_PARAMS,'scale_pos_weight':split['train']['rows']/split['train']['fraud']-1}
    train=lgb.Dataset(X.iloc[tr],label=y[tr],free_raw_data=True);valid=lgb.Dataset(X.iloc[va],label=y[va],reference=train,free_raw_data=True)
    m=lgb.train(params,train,num_boost_round=200,valid_sets=[valid],valid_names=['validation'],callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(period=50)])
    best=int(m.best_iteration or m.current_iteration());pv=m.predict(X.iloc[va],num_iteration=best)
    threshold,valf1=select_threshold(y[va],pv)
    # Threshold is frozen before test scoring in this run. Prior holdout exposure disclosed.
    pt=m.predict(X.iloc[te],num_iteration=best)
    assert np.isfinite(pt).all() and ((pt>=0)&(pt<=1)).all()
    out=ROOT/'full_eval_results';out.mkdir(exist_ok=True)
    for name,index,p in [('validation',va,pv),('test',te,pt)]:pd.DataFrame({'retained_row_position':index,'step':steps[index],'isfraud':y[index],'probability':p}).to_parquet(out/f'{name}_predictions.parquet',index=False)
    m.save_model(str(out/'model.txt'),num_iteration=best)
    primary,secondary=bootstrap(y[te],pt,steps[te]);legacy=json.loads((ROOT/'models/results.json').read_text())['lgbm']
    result={'authored_by':'ORION','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source':'ealaxi/paysim1','license':'CC BY-SA 4.0 (as source task specifies)','source_zip_sha256':SHA,'raw_rows':raw,'raw_fraud':rawfraud,'eligible_rows':len(X),'duplicates_removed':before-len(X),'split':split,'features':FEATURE_COLS,'features_sha256':hfile(ROOT/'src/features.py'),'script_sha256':hfile(__file__),'plan_sha256':hfile(ROOT/'FULL_EVAL_PLAN_ORION_20261024.md'),'params':params,'best_iteration':best,'threshold_selection':{'criterion':'maximize validation F1; ties choose higher threshold','threshold':threshold,'validation_f1':valf1,'test_used_for_selection':False},'validation':measures(y[va],pv,threshold),'test':measures(y[te],pt,threshold),'test_fixed_0_5':measures(y[te],pt,.5),'primary_bootstrap':primary,'secondary_bootstrap':secondary,'legacy_random_cv':legacy,'caveats':['Test window already evaluated on 2026-10-23; not a pristine new holdout.','Fixed-model bootstrap CIs do not include fitting/selection or future drift uncertainty.','24-hour block stationarity is approximate in a short 143-hour test window.','Post-transaction simulated detection; not pre-payment prevention.','Whole raw source used but analysis population limited to CASH_OUT/TRANSFER.','AP is average precision, not trapezoidal PR area; old random-CV prevalence/design differ.'],'resource':{'actual_nice':os.nice(0),'threads':2,'address_space_limit_bytes':6*1024**3,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'minimum_memavailable_gib':min(x[0] for x in samples)/1024**3,'monitor_samples':len(samples),'monitor_interval_seconds':.2,'elapsed_seconds':time.monotonic()-started},'files':{f.name:hfile(f) for f in out.glob('*') if f.is_file()}}
    assert result['resource']['peak_rss_mib']<6144 and result['resource']['minimum_memavailable_gib']>=3
    (ROOT/'FULL_EVAL_ORION_20261024.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
