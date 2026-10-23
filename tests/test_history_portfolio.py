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
