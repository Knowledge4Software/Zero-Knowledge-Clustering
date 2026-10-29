"""Float64 RaBitQ one-bit estimator for exhaustive fixed-bank audit queries.

Gao and Long, SIGMOD2024, DOI10.1145/3654970. This is an independent
reference implementation of the one-bit estimator, not their SIMD ANN engine.
Old exact groups are used as IVF anchors; no catalog row or candidate center
is discarded by search pruning. Numerical intervals are not ZK proofs.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import betaincinv

from .published_audit_baselines import distances


class RaBitQAudit:
    def __init__(self, rows, old_centers, seed, total_pair_budget,
                 family_failure_probability=.01, padded_dimension=None):
        x=np.asarray(rows,dtype=np.int64);old=np.asarray(old_centers,dtype=np.int64)
        self.groups=distances(x,old).argmin(1)
        self.old=old.astype(float);self.n,self.d=x.shape
        self.dimension=padded_dimension or ((self.d+63)//64)*64
        if self.dimension<self.d or self.dimension<3 or total_pair_budget<self.n*len(old):
            raise ValueError('invalid dimension or finite-query budget')
        if not 0<family_failure_probability<1:
            raise ValueError('invalid failure probability')
        rng=np.random.default_rng(seed)
        q,r=np.linalg.qr(rng.standard_normal((self.dimension,self.dimension)))
        # Positive R diagonal produces Haar orthogonal Q from a Gaussian matrix.
        q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
        self.rotation=q[:self.d]
        residual=x.astype(float)-self.old[self.groups]
        self.norms=np.linalg.norm(residual,axis=1)
        self.rotated_anchors=self.old@self.rotation
        rotated=residual@self.rotation
        code=np.where(rotated>0,1.,-1.)/math.sqrt(self.dimension)
        rho=np.divide(np.sum(code*rotated,axis=1),self.norms,
                      out=np.ones(self.n),where=self.norms!=0)
        self.rho=np.clip(rho,np.finfo(float).eps,1.)
        self.weighted_code=code*(self.norms/self.rho)[:,None]
        self.error_factor=np.sqrt(np.maximum(0.,1.-self.rho**2))/self.rho
        self.family_failure_probability=family_failure_probability
        self.total_pair_budget=total_pair_budget
        # A coordinate squared on S^(D-2) is Beta(1/2,(D-2)/2).
        # RaBitQ's rotational-invariance error identity is bounded by that
        # coordinate. Union bound covers the predeclared fixed finite queries.
        per_pair=family_failure_probability/total_pair_budget
        self.component_bound=math.sqrt(float(betaincinv(.5,(self.dimension-2)/2,1-per_pair)))
        self.old_cost=sum(map(int,distances(x,old).min(1)))
        self.code_bytes=(self.n*self.dimension+7)//8

    def query(self, centers, deterministic_component_bound=False):
        centers=np.asarray(centers,dtype=float)
        rotated=centers@self.rotation
        estimates=np.empty((self.n,len(centers)));errors=np.empty_like(estimates)
        component=1. if deterministic_component_bound else self.component_bound
        for g,anchor in enumerate(self.old):
            ids=np.flatnonzero(self.groups==g)
            if not len(ids):continue
            query=rotated-self.rotated_anchors[g]
            norms=np.linalg.norm(centers-anchor,axis=1)
            estimates[ids]=self.norms[ids,None]**2+norms[None,:]**2-2*self.weighted_code[ids]@query.T
            errors[ids]=2*self.norms[ids,None]*norms[None,:]*self.error_factor[ids,None]*component
        lower=np.maximum(0.,estimates-errors);upper=np.maximum(0.,estimates+errors)
        # The input reference cost is obtained by exact native preprocessing;
        # it is not claimed to be independently authenticated here.
        return dict(cost=float(estimates.min(1).sum()),labels=estimates.argmin(1),
                    lower=float(lower.min(1).sum()-self.old_cost),
                    upper=float(upper.min(1).sum()-self.old_cost),
                    distance_lower=lower,distance_upper=upper,
                    numerical_only=True,cryptographic_proof=False)
