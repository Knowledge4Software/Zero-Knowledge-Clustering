"""Check information isolation and exact costs of request-local affine banks."""
import copy
from campaigns.run_online_affine_control import bank_input
from commitjl.family_affine import prepare,query


def source():
    old=[[0,0],[4,4]]
    return dict(model=dict(n=16,k=2,centers=old,basis=[]),x=[[i%5,i%3] for i in range(16)],reuse_centers=[old,[[1,0],[5,4]],[[2,1],[6,5]]],comparison_base=old)


def test_future_requests_do_not_enter_earlier_preprocessing():
    first=source();changed=copy.deepcopy(first);changed['reuse_centers'][2]=[[100,100],[-100,-100]]
    assert bank_input(first,0)==bank_input(changed,0)
    assert bank_input(first,1)==bank_input(changed,1)
    assert bank_input(first,2)!=bank_input(changed,2)
    assert len(bank_input(first,1)['reuse_centers'])==1


def test_every_request_local_bank_matches_original_space_oracle():
    s=source()
    for i,centers in enumerate(s['reuse_centers']):
        inp=bank_input(s,i);assert inp['model']['centers']==centers
        prepared=prepare(inp['x'],inp['reuse_centers'])
        exact=sum(min(sum((x-y)**2 for x,y in zip(row,c)) for c in centers) for row in s['x'])
        assert query(prepared,centers)['total']==exact
