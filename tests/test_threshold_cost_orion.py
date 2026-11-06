"""Threshold selection checked by exhaustive confusion counts. AUTHORED BY ORION."""
import numpy as np
import pytest
from src.models import optimal_threshold


def cost(y,p,t,fn=1.,fp=.1):
    positive=p>=t
    return fn*np.sum((y==1)&~positive)+fp*np.sum((y==0)&positive)


def test_can_choose_no_alerts():
    y=np.array([1,0,0]);p=np.array([.1,.9,.9])
    t=optimal_threshold(y,p,cost_fn=1,cost_fp=2)
    assert cost(y,p,t,1,2)==1
    assert not (p>=t).any()


def test_matches_exhaustive_cost_with_ties():
    rng=np.random.default_rng(28)
    for _ in range(30):
        y=np.r_[0,1,rng.integers(2,size=18)]
        p=rng.choice([0.,.2,.5,.9,1.],size=len(y))
        for fn,fp in [(1,.1),(1,2),(0,1),(1,0)]:
            t=optimal_threshold(y,p,cost_fn=fn,cost_fp=fp)
            candidates=np.r_[np.unique(p),np.nextafter(max(p),np.inf)]
            assert cost(y,p,t,fn,fp)==min(cost(y,p,c,fn,fp) for c in candidates)


@pytest.mark.parametrize('fn,fp',[(-1,1),(1,-1),(float('nan'),1),(1,float('inf'))])
def test_reject_invalid_costs(fn,fp):
    with pytest.raises(ValueError):optimal_threshold(np.array([0,1]),np.array([.2,.8]),fn,fp)


@pytest.mark.parametrize('y', [[0,0,0],[1,1,1]])
def test_single_class_cost_optimum(y):
    y=np.array(y);p=np.array([0.,.5,1.])
    t=optimal_threshold(y,p)
    assert cost(y,p,t)==0


@pytest.mark.parametrize('y,p', [([],[]),([0,1],[.5]),([0,2],[.1,.9]),([0,1],[.1,float('nan')]),([0,1],[-.1,.9])])
def test_reject_bad_inputs(y,p):
    with pytest.raises(ValueError):optimal_threshold(y,p)
