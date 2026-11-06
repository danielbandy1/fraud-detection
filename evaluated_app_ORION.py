"""Saved forward-time results; no fitting or public deployment. AUTHORED BY ORION."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve,roc_curve
from src.evaluation_artifacts import load_evaluation,validation_cost_scenario

st.set_page_config(page_title='PaySim forward-time evaluation',layout='wide')
st.title('PaySim forward-time evaluation')
st.caption('Public simulated transactions • post-transaction detection • local educational results')
# Avoid caching away artifact changes: each rerun verifies source/model/prediction bytes.
try:
    report,frames,provenance=load_evaluation()
except (OSError,ValueError,KeyError) as error:
    st.error(f'Evaluation artifacts could not be verified: {error}')
    st.stop()
test=frames['test'];q=report['test'];threshold=report['threshold_selection']['threshold']
st.warning('This is one previously examined temporal holdout, not external deployment validation. Supplied API history may be less complete than benchmark history.')
a,b,c,d=st.columns(4)
a.metric('Test transactions',f"{q['rows']:,}")
b.metric('Test fraud prevalence',f"{q['base_rate']:.2%}")
c.metric('Test average precision',f"{q['average_precision']:.6f}")
d.metric('Test ROC-AUC',f"{q['auroc']:.6f}")
st.caption('Train: steps ≤500; validation: 501–600; test: 601–743. CASH_OUT/TRANSFER only. Histories use strictly earlier steps.')
ci=report['primary_bootstrap']
st.write(f"Fixed-model 95% intervals from 24-hour block bootstrap: AP [{ci['ap_95ci'][0]:.4f}, {ci['ap_95ci'][1]:.4f}]; ROC-AUC [{ci['auroc_95ci'][0]:.4f}, {ci['auroc_95ci'][1]:.4f}]. These omit fitting/selection and future-drift uncertainty.")
st.subheader('Frozen test operating point')
st.caption(f'Threshold {threshold:.10f}: selected for validation F1, fixed before scoring test in that run. Not cost-optimised.')
st.table({'Actual / predicted':['Legitimate','Fraud'],'Predicted legitimate':[q['tn'],q['fn']],'Predicted fraud':[q['fp'],q['tp']]})
st.write(f"Precision {q['precision']:.2%} • Recall {q['recall']:.2%} • F1 {q['f1']:.4f}")
left,right=st.columns(2)
y=test.isfraud.to_numpy();p=test.probability.to_numpy()
with left:
    precision,recall,_=precision_recall_curve(y,p)
    fig=go.Figure(go.Scatter(x=recall,y=precision,name='Fixed test predictions',mode='lines'))
    fig.add_hline(y=q['base_rate'],line_dash='dash',annotation_text='Test prevalence')
    fig.update_layout(title='Test precision–recall',xaxis_title='Recall',yaxis_title='Precision',xaxis_range=[0,1],yaxis_range=[0,1])
    st.plotly_chart(fig,width='stretch')
with right:
    fpr,tpr,_=roc_curve(y,p)
    fig=go.Figure(go.Scatter(x=fpr,y=tpr,name='Fixed test predictions',mode='lines'))
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line_dash='dash')
    fig.update_layout(title='Test ROC',xaxis_title='False-positive rate',yaxis_title='True-positive rate',xaxis_range=[0,1],yaxis_range=[0,1])
    st.plotly_chart(fig,width='stretch')
st.subheader('Validation-only cost exploration')
st.caption('Hypothetical relative costs. Changing these controls uses validation labels only and never changes the frozen test metrics above. Reusing validation for exploration adds selection uncertainty.')
fn=st.slider('Relative missed-fraud cost',1,50,10,key='cost_fn')
fp=st.slider('Relative false-alert cost',1,10,1,key='cost_fp')
scenario=validation_cost_scenario(frames['validation'],fn,fp)
st.metric('Validation-selected cost threshold',f"{scenario['threshold']:.10f}")
st.write(f"Validation FN {scenario['fn']:,} • FP {scenario['fp']:,} • empirical relative cost {scenario['cost']:,}. No dollar savings or test-optimality claim.")
with st.expander('Provenance and limitations'):
    st.json(provenance)
    for caveat in report['caveats']:st.write(caveat)
    st.caption('Artifact checks establish consistency with the local evaluation report, not signed authenticity. No raw account records are displayed.')
