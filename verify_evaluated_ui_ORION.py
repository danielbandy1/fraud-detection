"""Real saved-artifact view controls; no network server. AUTHORED BY ORION."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parent

def main():
    app=AppTest.from_file(str(ROOT/'evaluated_app_ORION.py'),default_timeout=60).run()
    assert not app.exception,[e.message for e in app.exception]
    assert not app.error,[e.value for e in app.error]
    before={m.label:m.value for m in app.metric}
    assert before['Test transactions']=='43,550'
    assert before['Test average precision']=='0.889492'
    assert before['Test ROC-AUC']=='0.994632'
    assert len(app.get('plotly_chart'))==2
    app.slider(key='cost_fn').set_value(50).run()
    assert not app.exception and not app.error
    assert {m.label:m.value for m in app.metric if m.label.startswith('Test')}=={k:v for k,v in before.items() if k.startswith('Test')}
    print(json.dumps({'authored_by':'ORION','status':'PASS','checks':['real verified artifacts','expected frozen test metrics','two curve elements','validation slider leaves test metrics unchanged'],'scope':'AppTest runtime/control/serialization only; no browser pixel inspection or deployment'}))
if __name__=='__main__':main()
