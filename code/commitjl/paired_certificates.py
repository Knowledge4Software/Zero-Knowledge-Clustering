"""Exact-integer paired bounds for nearest-center cost changes.

The stored statistics are unchanged: H=T*B*(x-anchor), E=||R||^2,
R=(T^2 I-B^T B)*(x-anchor). For G=B B^T-T^2 I, the identity is

 T^2 D(x,c)-T^6 ||x-c||^2 = 2 T^2 <R(x),R(c)> + (H(x)-H(c))^T G (H(x)-H(c)).

Subtract two center distances before bounding the residual term. The quadratic
Gram difference becomes affine in H(x), so rounded-basis correction is exact.
The remaining uncertainty is proportional to ||R(c_new)-R(c_old)||.

For exact pair differences F[a,b]=f_new[a]-f_old[b], both min_a max_b F and
max_b min_a F equal min f_new-min f_old. Therefore applying the first ordering
to lower bounds and the second to upper bounds gives a valid change interval.
This module is a numerical reference, not a cryptographic proof by itself.
"""
from __future__ import annotations

import numpy as np
from .reuse_certificates import integer_stats, ceil_sqrt


def change_parameters(model, old_centers, new_centers):
    b=np.asarray(model['basis'],dtype=object)
    old=np.asarray(old_centers,dtype=object);new=np.asarray(new_centers,dtype=object)
    if b.ndim!=2 or old.ndim!=2 or new.ndim!=2 or min(len(old),len(new))<2:
        raise ValueError('nonempty basis and at least two centers required')
    if old.shape[1]!=b.shape[1] or new.shape[1]!=b.shape[1]:
        raise ValueError('center feature count differs from basis')
    if 'input_bits' in model:
        limit=2**int(model['input_bits'])
        if any(abs(int(v))>limit for v in np.concatenate((old,new)).flat):
            raise ValueError('center outside approved input range')
    t=int(model['basis_scale']);t2=t*t
    ho,eo=integer_stats(model,old);hn,en=integer_stats(model,new)
    ho=ho.astype(object);hn=hn.astype(object)
    # Rounded basis entries are small enough for an exact int64 Gram product
    # on ordinary workloads. Keep the arbitrary-integer fallback for extremes.
    bound=max(abs(int(v)) for v in b.flat)
    if b.shape[1]*bound*bound<2**63:
        bi=np.asarray(b,dtype=np.int64);gram=(bi@bi.T).astype(object)
    else:gram=b@b.T
    gram-=t2*np.eye(len(b),dtype=object)
    shifted=gram-t2*np.eye(len(b),dtype=object)
    anchor=np.asarray(model['centers'][0],dtype=object)
    def residual(c):
        w=c-anchor
        return t2*w-(w@b.T)@b
    ro,rn=residual(old),residual(new)
    # These center-only quadratic terms used to be repeated for every pair.
    qo=[int(row@gram@row) for row in ho];qn=[int(row@gram@row) for row in hn]
    no=[sum(int(v)**2 for v in row)+int(e) for row,e in zip(ho,eo)]
    nn=[sum(int(v)**2 for v in row)+int(e) for row,e in zip(hn,en)]
    coefficients=[];constants=[];radii=[]
    for a in range(len(new)):
        for j in range(len(old)):
            dh=hn[a]-ho[j]
            coefficients.append((2*(shifted@dh)).tolist())
            constants.append(t2*(nn[a]-no[j])-(qn[a]-qo[j]))
            radii.append(ceil_sqrt(sum(int(v)**2 for v in rn[a]-ro[j])))
    return dict(coefficients=coefficients,constants=constants,residual_delta_radii=radii,
                old_count=len(old),new_count=len(new),cost_denominator=t**6)


def paired_change(model, rows, old_centers, new_centers, *, row_outputs=False):
    if not len(rows):raise ValueError('empty audit catalog')
    p=change_parameters(model,old_centers,new_centers)
    h,e=integer_stats(model,rows)
    score=h.astype(object)@np.asarray(p['coefficients'],dtype=object).T+np.asarray(p['constants'],dtype=object)
    radii=np.asarray([ceil_sqrt(v) for v in e],dtype=object)
    errors=2*int(model['basis_scale'])**2*radii[:,None]*np.asarray(p['residual_delta_radii'],dtype=object)
    shape=(len(rows),p['new_count'],p['old_count'])
    lower=(score-errors).reshape(shape).max(axis=2).min(axis=1)
    upper=(score+errors).reshape(shape).min(axis=1).max(axis=1)
    if any(a>b for a,b in zip(lower,upper)):
        raise AssertionError('paired interval inconsistent')
    lo=sum(int(v) for v in lower);hi=sum(int(v) for v in upper)
    result=dict(change_lower_scaled=lo,change_upper_scaled=hi,cost_denominator=p['cost_denominator'],
                decision='improves' if hi<0 else ('worsens' if lo>0 else 'inconclusive'))
    if row_outputs:result.update(row_lower=lower.tolist(),row_upper=upper.tolist())
    return result
