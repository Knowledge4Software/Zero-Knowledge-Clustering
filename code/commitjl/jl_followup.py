"""Exploratory JL ensembles and training-only covariance range finding.

These are estimates/subspace constructors, not new probability guarantees.
Certificate acceptance continues to use the exact rounded-basis inequalities.
"""
import numpy as np
from .jl_certificate_pilots import jl_map


def ensemble_scores(context, maps, alpha=1., gated=False):
    estimates=[]
    delta=context['center_residual']-context['center_residual'][0]
    for a in maps:
        estimates.append(2*((a@context['row_residual'].T).T@(a@delta.T))/3)
    estimates=np.asarray(estimates)
    median=np.median(estimates,axis=0)
    scores=context['central']-alpha*median
    if gated:
        # Only replace the no-JL decision when all independent estimates agree
        # and their margin exceeds twice the largest score dispersion. No truth.
        individual=(context['central'][None,:,:]-alpha*estimates).argmin(2)
        proposed=scores.argmin(1)
        ordered=np.sort(scores,axis=1)
        spread=np.max(np.median(np.abs(estimates-median[None,:,:]),axis=0),axis=1)
        use=np.all(individual==proposed[None,:],axis=0)&(ordered[:,1]-ordered[:,0]>2*alpha*spread)
        scores=np.where(use[:,None],scores,context['central'])
    return scores


def covariance_jl_basis(model, training, extra, seed, powers, oversample=8):
    """Training covariance times a sparse random test matrix, then Ritz choice.

    Powers=0 is an untrained random extension. Powers>0 uses training only;
    this is randomized subspace iteration, not a stronger JL theorem or a new
    algorithm. Future centers/audit rows never enter this constructor.
    """
    original=np.asarray(model['basis'],dtype=np.int64);t=int(model['basis_scale'])
    d=original.shape[1];width=extra+(oversample if powers else 0)
    if extra<1 or len(original)+extra>min(128,d) or powers not in (0,1,2):
        raise ValueError('unsupported extension')
    q=np.linalg.qr(original.T.astype(float),mode='reduced')[0]
    residual=np.asarray(training,dtype=float)-np.asarray(model['centers'][0],dtype=float)
    residual-=residual@q@q.T
    y=jl_map(d,width,seed).toarray().T.astype(float)
    for _ in range(2):y-=q@(q.T@y)
    for _ in range(powers):
        y=residual.T@(residual@y)
        for __ in range(2):y-=q@(q.T@y)
        y,tri=np.linalg.qr(y,mode='reduced')
        if np.any(np.abs(np.diag(tri))<1e-10):raise ValueError('degenerate training sketch; do not search seeds')
    y,tri=np.linalg.qr(y,mode='reduced')
    if np.any(np.abs(np.diag(tri))<1e-10):raise ValueError('degenerate sketch')
    if powers:
        _,_,v=np.linalg.svd(residual@y,full_matrices=False)
        y=y@v[:extra].T
    else:y=y[:,:extra]
    for j in range(extra):
        if y[np.argmax(np.abs(y[:,j])),j]<0:y[:,j]*=-1
    assert np.allclose(q.T@y,0,atol=1e-9)
    assert np.allclose(y.T@y,np.eye(extra),atol=1e-9)
    return dict(model,basis=np.concatenate((original,np.rint(y.T*t).astype(np.int64))).tolist())
