"""Future and same-hour observations cannot influence earlier features.
AUTHORED BY ORION
"""
import pandas as pd
from src.features import add_graph_features

def test_future_rows_do_not_change_past_and_row_order_stays():
    base=pd.DataFrame({'step':[2,1,2],'nameorig':['A','A','A'],'namedest':['X','X','Y']},index=[5,3,8])
    old=add_graph_features(base)
    future=pd.concat([base,pd.DataFrame({'step':[5],'nameorig':['A'],'namedest':['X']},index=[9])])
    pd.testing.assert_frame_equal(old,add_graph_features(future).loc[base.index])
    assert old.orig_tx_count.tolist()==[1,0,1]
    assert old.orig_unique_dest.tolist()==[1,0,1]
    assert old.index.tolist()==[5,3,8]

def test_same_hour_transactions_cannot_see_one_another():
    base=pd.DataFrame({'step':[1,1,2],'nameorig':['A','A','A'],'namedest':['X','Y','X']})
    out=add_graph_features(base)
    assert out.orig_tx_count.tolist()==[0,0,2]
    assert out.dest_recv_count.tolist()==[0,0,1]

def test_vectorized_history_matches_independent_small_reference():
    import numpy as np
    rng=np.random.default_rng(99)
    frame=pd.DataFrame({'step':rng.integers(1,8,100),'nameorig':rng.integers(0,6,100),'namedest':rng.integers(0,7,100)})
    out=add_graph_features(frame)
    for i,row in frame.iterrows():
        before=frame[frame.step<row.step]
        sent=before[before.nameorig.eq(row.nameorig)]
        assert out.loc[i,'orig_tx_count']==len(sent)
        assert out.loc[i,'orig_unique_dest']==sent.namedest.nunique()
        assert out.loc[i,'dest_recv_count']==before.namedest.eq(row.namedest).sum()
        expected=min(row.step-sent.step.max(),999) if len(sent) else 999
        assert out.loc[i,'orig_step_gap']==expected
