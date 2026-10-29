"""Group-aggregate cost-change intervals with bounded switching corrections.

The fixed group assignment need not be a true nearest-center assignment.
Group moments must be authenticated against the complete audit catalog in a
proof implementation. This module supplies exact-integer numerical bounds only.
"""
from math import ceil
import numpy as np
from .paired_certificates import change_parameters
from .reuse_certificates import integer_stats,ceil_sqrt
from .block_certificates import residual_vectors


def advantage_bounds(model,heads,energies,centers,groups):
    """Bound min_j ||x-c_j||² - ||x-c_group||² in T^6 units."""
    params=change_parameters(model,centers,centers);k=len(centers)
    groups=np.asarray(groups,dtype=int)
    if len(groups)!=len(heads) or any(g<0 or g>=k for g in groups):
        raise ValueError('invalid fixed group labels')
    radii=np.asarray([ceil_sqrt(e) for e in energies],dtype=object)
    # Each record needs only its authenticated group's column, not all k*k
    # center pairs. Grouping is public numerical work; the relation is unchanged.
    lower=np.empty(len(groups),dtype=object);upper=np.empty(len(groups),dtype=object)
    coefficients=np.asarray(params['coefficients'],dtype=object)
    constants=np.asarray(params['constants'],dtype=object)
    center_radii=np.asarray(params['residual_delta_radii'],dtype=object)
    for g in range(k):
        selected=np.flatnonzero(groups==g)
        if not len(selected):continue
        pairs=np.arange(g,k*k,k)
        score=heads[selected].astype(object)@coefficients[pairs].T+constants[pairs]
        errors=2*int(model['basis_scale'])**2*radii[selected,None]*center_radii[pairs][None,:]
        lower[selected]=(score-errors).min(axis=1);upper[selected]=(score+errors).min(axis=1)
    assert all(a<=b<=0 for a,b in zip(lower,upper))
    return lower,upper


def assigned_change(prepared,old,new):
    k=len(old)
    if len(new)!=k or len(prepared['counts'])!=k:raise ValueError('fixed center correspondence required')
    return sum(2*sum(int(x)*(int(a)-int(b)) for x,a,b in zip(prepared['sums'][g],old[g],new[g]))
        +int(prepared['counts'][g])*(sum(int(v)**2 for v in new[g])-sum(int(v)**2 for v in old[g])) for g in range(k))


def compressed_old_correction_bounds(model,rows,prepared,old,new):
    """Bound the assigned aggregate with authenticated sums of H and E only.

    For group g, ||sum R_i|| <= sqrt(n_g * sum E_i). Applying the same
    rounded-basis paired identity to the group sum bounds A without storing
    full original-space coordinate sums in the query relation. This is a
    numerical reference; its compressed moments require their own binding.
    """
    bounds=exact_old_correction_bounds(model,rows,prepared,old,new)
    heads,energy=integer_stats(model,rows);groups=np.asarray(prepared['groups'],dtype=int);k=len(old)
    params=change_parameters(model,old,new);center=0;error=0
    for g in range(k):
        selected=groups==g;count=int(selected.sum());hs=heads[selected].sum(0);es=sum(map(int,energy[selected]));index=g*k+g
        center+=count*int(params['constants'][index])+sum(int(a)*int(b) for a,b in zip(hs,params['coefficients'][index]))
        error+=2*int(model['basis_scale'])**2*ceil_sqrt(count*es)*int(params['residual_delta_radii'][index])
    exact_assigned=bounds['assigned_scaled'];assert center-error<=exact_assigned<=center+error
    bounds['change_lower_scaled']+=center-error-exact_assigned;bounds['change_upper_scaled']+=center+error-exact_assigned
    bounds.update(assigned_lower_scaled=center-error,assigned_upper_scaled=center+error,assigned_upper_bound_fast_path=center+error<0,compressed_moments=True)
    return bounds


def exact_old_correction_bounds(model,rows,prepared,old,new):
    """Stronger profile requiring authenticated exact old nearest groups.

    This numerical reference checks that requirement explicitly. A proof
    implementation must additionally enforce it in every preprocessing row.
    With old switching advantage zero, Delta=A+sum(b_new)<=A. A negative A
    therefore certifies improvement without per-row query bounds.
    """
    x=np.asarray(rows,dtype=np.int64);c=np.asarray(old,dtype=np.int64)
    groups=np.asarray(prepared['groups'],dtype=int)
    if x.ndim!=2 or c.ndim!=2 or x.shape[1]!=c.shape[1] or len(x)!=len(groups):raise ValueError('invalid exact-old catalog')
    if any(g<0 or g>=len(c) for g in groups):raise ValueError('invalid fixed groups')
    maximum=max(abs(int(x.min())),abs(int(x.max())),abs(int(c.min())),abs(int(c.max())))
    if 4*x.shape[1]*maximum**2>=2**63:
        dist=np.asarray([[sum((int(a)-int(b))**2 for a,b in zip(row,center)) for center in c] for row in x],dtype=object)
    else:dist=((x[:,None,:]-c[None,:,:])**2).sum(2)
    if not np.all(dist[np.arange(len(x)),groups]==dist.min(1)):raise ValueError('groups are not exact old nearest centers')
    lower,upper=advantage_bounds(model,*integer_stats(model,x),new,groups)
    denominator=int(model['basis_scale'])**6;assigned=assigned_change(prepared,old,new)*denominator
    assert all(int(a)<=int(b)<=0 for a,b in zip(lower,upper))
    return dict(assigned_scaled=assigned,row_lower=lower,row_upper=upper,cost_denominator=denominator,
        change_lower_scaled=assigned+sum(map(int,lower)),change_upper_scaled=assigned+sum(map(int,upper)),
        assigned_upper_bound_fast_path=assigned<0)


def correction_bounds(model,rows,prepared,old,new):
    """The original rows are used only to obtain authenticated H/E statistics."""
    if not len(rows) or len(rows)!=len(prepared['groups']):raise ValueError('complete fixed catalog required')
    h,e=integer_stats(model,rows)
    a,b=advantage_bounds(model,h,e,old,prepared['groups'])
    c,d=advantage_bounds(model,h,e,new,prepared['groups'])
    lower=np.asarray([int(x)-int(y) for x,y in zip(c,b)],dtype=object)
    upper=np.asarray([int(x)-int(y) for x,y in zip(d,a)],dtype=object)
    assert all(x<=y for x,y in zip(lower,upper))
    denominator=int(model['basis_scale'])**6
    assigned=assigned_change(prepared,old,new)*denominator
    return dict(assigned_scaled=assigned,row_lower=lower,row_upper=upper,cost_denominator=denominator,
        change_lower_scaled=assigned+sum(map(int,lower)),change_upper_scaled=assigned+sum(map(int,upper)))


def paired_correction_bounds(model,rows,prepared,old,new):
    """Cancel the fixed group's movement before bounding switching corrections.

    For each group g, the residual direction for pair (a,b) is
    R(new_a)-R(old_b)-R(new_g)+R(old_g). Bounds on the resulting pair
    differences give min_a max_b lower and max_b min_a upper bounds.
    """
    if len(old)!=len(new) or len(rows)!=len(prepared['groups']):raise ValueError('fixed groups/center correspondence required')
    groups=np.asarray(prepared['groups'],dtype=int);k=len(old)
    if any(g<0 or g>=k for g in groups):raise ValueError('invalid fixed group labels')
    params=change_parameters(model,old,new);h,e=integer_stats(model,rows)
    score=h.astype(object)@np.asarray(params['coefficients'],dtype=object).T+np.asarray(params['constants'],dtype=object)
    reference=np.asarray([score[i,g*k+g] for i,g in enumerate(groups)],dtype=object)
    score-=reference[:,None]
    ro=residual_vectors(model,old).astype(object);rn=residual_vectors(model,new).astype(object)
    table=[]
    for g in range(k):
        radii=[]
        for a in range(k):
            for b in range(k):
                direction=rn[a]-ro[b]-rn[g]+ro[g]
                radii.append(ceil_sqrt(sum(int(v)**2 for v in direction)))
        table.append(radii)
    selected=np.asarray([table[g] for g in groups],dtype=object)
    row_radius=np.asarray([ceil_sqrt(v) for v in e],dtype=object)
    errors=2*int(model['basis_scale'])**2*row_radius[:,None]*selected
    lower=(score-errors).reshape(len(rows),k,k).max(2).min(1)
    upper=(score+errors).reshape(len(rows),k,k).min(1).max(1)
    assert all(int(a)<=int(b) for a,b in zip(lower,upper))
    denominator=params['cost_denominator'];assigned=assigned_change(prepared,old,new)*denominator
    return dict(assigned_scaled=assigned,row_lower=lower,row_upper=upper,cost_denominator=denominator,
        change_lower_scaled=assigned+sum(map(int,lower)),change_upper_scaled=assigned+sum(map(int,upper)))


def refine(result,exact_corrections,budget_fraction):
    """A fixed budget, chosen by interval width before consulting exact values."""
    if not 0<=budget_fraction<=1:raise ValueError('budget outside [0,1]')
    lower,upper=result['row_lower'],result['row_upper']
    if len(exact_corrections)!=len(lower):raise ValueError('correction length mismatch')
    uncertain=[i for i,(a,b) in enumerate(zip(lower,upper)) if int(a)<int(b)]
    order=sorted(uncertain,key=lambda i:(-(int(upper[i])-int(lower[i])),i))
    selected=order[:ceil(budget_fraction*len(lower))]
    lo,hi=result['change_lower_scaled'],result['change_upper_scaled']
    for i in selected:
        exact=int(exact_corrections[i])
        if not int(lower[i])<=exact<=int(upper[i]):raise AssertionError('exact correction outside bound')
        lo+=exact-int(lower[i]);hi+=exact-int(upper[i])
    assert lo<=hi
    return dict(change_lower_scaled=lo,change_upper_scaled=hi,selected_indices=selected,
        exact_rows=len(selected),uncertain_rows=len(uncertain),cost_denominator=result['cost_denominator'])
