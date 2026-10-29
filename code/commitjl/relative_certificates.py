"""Exact rounded-basis bounds on within-set distance differences.

Reuses H and E. This avoids adding independent absolute distance errors when
certifying a label. It is not an exact aggregate cost-change certificate.
"""
import numpy as np
from .paired_certificates import change_parameters
from .reuse_certificates import integer_stats,ceil_sqrt


def relative_certificate(model,rows,centers):
    h,e=integer_stats(model,rows);hc,ec=integer_stats(model,centers)
    h=h.astype(object);hc=hc.astype(object)
    score=h@hc.T*(-2)+np.asarray([sum(int(x)**2 for x in row) for row in hc],dtype=object)[None,:]+ec[None,:]
    labels=np.asarray(score.argmin(1),dtype=int)
    p=change_parameters(model,centers,centers)
    pair=h@np.asarray(p['coefficients'],dtype=object).T+np.asarray(p['constants'],dtype=object)
    radii=np.asarray([ceil_sqrt(v) for v in e],dtype=object)
    errors=2*int(model['basis_scale'])**2*radii[:,None]*np.asarray(p['residual_delta_radii'],dtype=object)[None,:]
    lower=(pair-errors).reshape(len(rows),len(centers),len(centers))
    flags=[all(j==winner or int(lower[i,j,winner])>0 for j in range(len(centers))) for i,winner in enumerate(labels)]
    return dict(labels=labels.tolist(),certified_flags=flags,certified_count=sum(flags))
