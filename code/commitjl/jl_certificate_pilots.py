"""JL research helpers: deterministic envelopes and untrusted refinement hints.

No new proof protocol is implemented here. The randomized-basis envelopes use
the existing exact rounded-basis algebra, not an assumed probabilistic JL event.
Hints affect only work ordering; selected corrections must still be verified.
"""
import numpy as np
from .sparse_gen import sparse_matrix_gen_csr
from .reuse_certificates import integer_stats,ceil_sqrt
from .paired_certificates import change_parameters
from .block_certificates import residual_vectors


def jl_map(d,width,seed,q=3):
    seed_bytes=('jl-certificate-pilot-v1:'+str(seed)).encode()
    return sparse_matrix_gen_csr(seed_bytes,width,d,q)[1]


def random_residual_basis(model,extra,seed):
    """Orthogonalize sparse-JL measurements against the frozen head span.

    QR preserves the measurement subspace, but the result must be described as
    an orthogonalized sparse sketch; no textbook sparse-JL distortion theorem
    is claimed for this transformed/rounded basis.
    """
    original=np.asarray(model['basis'],dtype=np.int64);t=int(model['basis_scale'])
    if extra<1 or len(original)+extra>min(128,original.shape[1]):raise ValueError('invalid extension size')
    a=jl_map(original.shape[1],extra,seed,min(3,extra));q=np.linalg.qr(original.T.astype(float),mode='reduced')[0];directions=a.toarray().T.astype(float)
    for _ in range(2):directions-=q@(q.T@directions)
    extension,triangular=np.linalg.qr(directions,mode='reduced')
    if np.any(np.abs(np.diag(triangular))<1e-10):raise ValueError('degenerate fixed sketch; report it rather than searching seeds')
    basis=np.concatenate((original,np.rint(extension.T*t).astype(np.int64)))
    return dict(model,basis=basis.tolist())


def hint_context(model,rows,new,groups):
    """Floating estimates only; never use these as certified endpoints."""
    h,_=integer_stats(model,rows);k=len(new);p=change_parameters(model,new,new);indices=[j*k for j in range(k)];den=p['cost_denominator']
    central=h.astype(object)@np.asarray(p['coefficients'],dtype=object)[indices].T+np.asarray(p['constants'],dtype=object)[indices]
    t2=int(model['basis_scale'])**2
    return dict(central=np.asarray(central/den,dtype=float),row_residual=np.asarray(residual_vectors(model,rows),float)/t2,
        center_residual=np.asarray(residual_vectors(model,new),float)/t2,groups=np.asarray(groups,dtype=int))


def sketch_hints(context,a,q=3):
    row=(a@context['row_residual'].T).T;delta=context['center_residual']-context['center_residual'][0];centers=(a@delta.T).T
    scores=context['central']-2*(row@centers.T)/q
    group=scores[np.arange(len(scores)),context['groups']]
    return scores.min(1)-group,scores.argmin(1)


def hint_order(bounds,estimated,mode='predicted_gain'):
    lo,hi=bounds['row_lower'],bounds['row_upper'];den=bounds['cost_denominator'];a=bounds['assigned_scaled']
    uncertain=[i for i,(l,u) in enumerate(zip(lo,hi)) if int(l)<int(u)]
    if mode=='width':return sorted(uncertain,key=lambda i:(-(int(hi[i])-int(lo[i])),i))
    if mode!='predicted_gain' or len(estimated)!=len(lo):raise ValueError('invalid hint ordering')
    if not np.isfinite(estimated).all():return hint_order(bounds,None,'width')
    # Clipping to valid intervals affects heuristics only. No oracle truth is
    # available while choosing direction/order. Index ties are deterministic.
    values=[max(int(l),min(int(u),int(round(float(v)*den)))) for l,u,v in zip(lo,hi,estimated)]
    worsening=a+sum(values)>=0
    gain=[values[i]-int(lo[i]) if worsening else int(hi[i])-values[i] for i in range(len(lo))]
    return sorted(uncertain,key=lambda i:(-gain[i],-(int(hi[i])-int(lo[i])),i))


def refinement_work(bounds,exact,order,max_rows):
    if len(set(order))!=len(order) or any(i<0 or i>=len(exact) for i in order):raise ValueError('invalid row coverage')
    lo,hi=bounds['change_lower_scaled'],bounds['change_upper_scaled']
    def decision():return 'worsens' if lo>0 else 'improves' if hi<0 else 'equal' if lo==hi==0 else 'inconclusive'
    if decision()!='inconclusive':return dict(exact_rows=0,decision=decision(),lower=lo,upper=hi)
    used=0
    for i in order[:max_rows]:
        value=int(exact[i]);l,u=int(bounds['row_lower'][i]),int(bounds['row_upper'][i])
        if not l<=value<=u:raise ValueError('unverified correction outside old interval')
        lo+=value-l;hi+=value-u;used+=1
        if decision()!='inconclusive':break
    return dict(exact_rows=used,decision=decision(),lower=lo,upper=hi)


def aggregate_stats(model,prepared):
    counts=np.asarray(prepared['counts'],dtype=object);sums=np.asarray(prepared['sums'],dtype=object);anchor=np.asarray(model['centers'][0],dtype=object)
    # Artificial rows produce centered vectors S_g-n_g*anchor, not S_g-anchor.
    points=sums-(counts[:,None]-1)*anchor
    # integer_stats accepts int64 original coordinates; these aggregate points
    # are bounded by a small multiple of N*X on the current fixed catalog.
    if any(abs(int(v))>=2**63 for v in points.flat):raise ValueError('aggregate outside int64 reference input range')
    h,e=integer_stats(model,np.asarray(points,dtype=np.int64))
    return dict(heads=h,energy=e,counts=counts)


def compressed_aggregate_bounds(base,model,stats,old,new):
    """Exact norm of each group residual sum, unlike sqrt(n*sum row energies).

    Authenticated construction would require a one-time proof binding these
    aggregate H/E statistics to full raw moments. Mixed row/group bases are
    allowed at the SAME fixed-point scale. No speed claim is inferred here.
    """
    p=change_parameters(model,old,new);den=int(model['basis_scale'])**6
    if den!=base['cost_denominator']:raise ValueError('mixed scale unsupported')
    k=len(old);center=0;error=0
    for g in range(k):
        index=g*k+g;center+=int(stats['counts'][g])*int(p['constants'][index])+sum(int(a)*int(b) for a,b in zip(stats['heads'][g],p['coefficients'][index]))
        error+=2*int(model['basis_scale'])**2*ceil_sqrt(stats['energy'][g])*int(p['residual_delta_radii'][index])
    assert center-error<=base['assigned_scaled']<=center+error
    lower=base['change_lower_scaled']+center-error-base['assigned_scaled'];upper=base['change_upper_scaled']+center+error-base['assigned_scaled']
    return dict(lower=lower,upper=upper,assigned_lower=center-error,assigned_upper=center+error,cost_denominator=den,private_aggregate_scalars=k*(len(model['basis'])+2),proof_implemented=False)
