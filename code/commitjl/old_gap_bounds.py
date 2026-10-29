"""Native feasibility of authenticated old gaps plus small center movements.

This is a numerical reference, not an implemented proof relation. A proof
profile would have to bind each old gap to its original row during preprocessing
and authenticate it with the stored H/E/group leaf on every future query.
"""
import numpy as np
from .group_change_bounds import exact_old_correction_bounds
from .paired_certificates import change_parameters
from .reuse_certificates import integer_stats,ceil_sqrt
from .block_certificates import residual_vectors


def old_gap_correction_bounds(model,rows,prepared,old,new,*,minimum_only=False):
    base=exact_old_correction_bounds(model,rows,prepared,old,new)
    x=np.asarray(rows,dtype=np.int64);old=np.asarray(old,dtype=np.int64);new=np.asarray(new,dtype=np.int64)
    groups=np.asarray(prepared['groups'],dtype=int);k=len(old);den=base['cost_denominator']
    maximum=max(abs(int(v)) for v in np.concatenate((x,old)).flat)
    safe=4*x.shape[1]*maximum**2<2**63
    dist=((x[:,None,:]-old[None,:,:])**2).sum(2) if safe else np.asarray([[sum((int(a)-int(b))**2 for a,b in zip(row,c)) for c in old] for row in x],dtype=object)
    gaps=dist.astype(object)-dist[np.arange(len(x)),groups,None].astype(object)
    assert all(int(v)>=0 for v in gaps.flat)
    lower_gaps=gaps.copy()
    if minimum_only:
        for i,g in enumerate(groups):
            margin=min(int(gaps[i,j]) for j in range(k) if j!=g)
            for j in range(k):lower_gaps[i,j]=0 if j==g else margin
    params=change_parameters(model,old,new);diagonal=[j*k+j for j in range(k)]
    coefficients=np.asarray(params['coefficients'],dtype=object)[diagonal];constants=np.asarray(params['constants'],dtype=object)[diagonal]
    h,e=integer_stats(model,rows);radius=np.asarray([ceil_sqrt(v) for v in e],dtype=object)
    movement=residual_vectors(model,new).astype(object)-residual_vectors(model,old).astype(object)
    scores=h.astype(object)@coefficients.T+constants
    centered=scores-scores[np.arange(len(x)),groups,None]
    table=np.asarray([[ceil_sqrt(sum(int(v)**2 for v in movement[j]-movement[g])) for j in range(k)] for g in range(k)],dtype=object)
    errors=2*int(model['basis_scale'])**2*radius[:,None]*table[groups]
    lower=(lower_gaps*den+centered-errors).min(1)
    upper=(gaps*den+centered+errors).min(1)
    lo=np.asarray([max(int(a),int(b)) for a,b in zip(base['row_lower'],lower)],dtype=object)
    hi=base['row_upper'] if minimum_only else np.asarray([min(int(a),int(b)) for a,b in zip(base['row_upper'],upper)],dtype=object)
    assert all(int(a)<=int(b)<=0 for a,b in zip(lo,hi))
    return dict(**{key:value for key,value in base.items() if key not in ['row_lower','row_upper','change_lower_scaled','change_upper_scaled']},row_lower=lo,row_upper=hi,
        change_lower_scaled=base['assigned_scaled']+sum(map(int,lo)),change_upper_scaled=base['assigned_scaled']+sum(map(int,hi)),
        additional_private_scalars_per_row=1 if minimum_only else k-1,proof_implemented=False)
