"""Independent exact-integer reference for rounded-basis audit certificates.

The target is the committed quantized dataset. Floating-point fidelity is a
separate empirical metric. Large arithmetic uses Python integers; the int64
projection fast path is enabled only after checking conservative overflow bounds.
"""
from __future__ import annotations
from math import isqrt
import numpy as np


def ceil_sqrt(value):
    value=int(value)
    if value<0:raise ValueError('negative energy')
    root=isqrt(value)
    return root+(root*root<value)


def exact_squared_norms(values):
    """Exact row energies, with bounded int64 limb products and Python sums.

    Splitting signed coordinates avoids overflow without one Python integer
    multiplication per coordinate. Chunking bounds the temporary working set.
    The object path remains the oracle for values outside signed int64.
    """
    values=np.asarray(values)
    if values.ndim!=2 or not values.shape[1]:raise ValueError('invalid energy shape')
    if values.dtype!=np.dtype('int64'):
        return np.asarray([sum(int(v)**2 for v in row) for row in values],dtype=object)
    width=values.shape[1]
    bits=max(1,min(30,(((2**63-1)//width).bit_length()-1)//2))
    limbs=(64+bits-1)//bits;mask=(1<<bits)-1
    result=np.empty(len(values),dtype=object)
    for start in range(0,len(values),256):
        remaining=values[start:start+256];parts=[]
        for _ in range(limbs-1):
            parts.append(remaining&mask);remaining=remaining>>bits
        parts.append(remaining)
        total=np.zeros(len(remaining),dtype=object)
        for i,a in enumerate(parts):
            for j in range(i,len(parts)):
                term=(a*parts[j]).sum(axis=1,dtype=np.int64).astype(object)
                total+=term*((1 if i==j else 2)<<(bits*(i+j)))
        result[start:start+len(total)]=total
    return result


def integer_stats(model, rows):
    b=np.asarray(model['basis'],dtype=np.int64);t=int(model['basis_scale'])
    w=np.asarray(rows,dtype=np.int64)-np.asarray(model['centers'][0],dtype=np.int64)
    if b.ndim!=2 or w.ndim!=2 or b.shape[1]!=w.shape[1]:raise ValueError('invalid statistic shape')
    wmax=int(np.abs(w).max(initial=0))
    dotmax=wmax*max(sum(abs(int(v)) for v in row) for row in b)
    residualmax=t*t*wmax+dotmax*max(sum(abs(int(v)) for v in col) for col in b.T)
    if max(dotmax*t,residualmax)<2**63:
        dot=w@b.T;heads=t*dot;residual=t*t*w-dot@b
    else:
        w=w.astype(object);b=b.astype(object);dot=w@b.T;heads=t*dot;residual=t*t*w-dot@b
    energy=exact_squared_norms(residual)
    return heads,energy


def gram_defect(model):
    b=np.asarray(model['basis'],dtype=object);t=int(model['basis_scale'])
    gram=b@b.T
    for i in range(len(b)):gram[i,i]-=t*t
    return max(sum(abs(int(v)) for v in row) for row in gram)


def certificate(model, rows, centers):
    if len(centers)<2:raise ValueError('at least two centers required')
    h,e=integer_stats(model,rows);hc,ec=integer_stats(model,centers)
    t=int(model['basis_scale']);g=gram_defect(model);t2=t*t
    radii=[ceil_sqrt(v) for v in e];center_radii=[ceil_sqrt(v) for v in ec]
    labels=[];flags=[];lower=upper=psi=margin=0
    for i,row in enumerate(h):
        distances=[];errors=[]
        for j,center in enumerate(hc):
            h2=sum((int(a)-int(b))**2 for a,b in zip(row,center))
            distances.append(h2+int(e[i])+int(ec[j]))
            errors.append(2*t2*radii[i]*center_radii[j]+g*h2)
        winner=min(range(len(centers)),key=distances.__getitem__)
        scaled=[t2*d for d in distances]
        valid=all(scaled[j]-scaled[winner]>errors[j]+errors[winner]
                  for j in range(len(centers)) if j!=winner)
        labels.append(winner);flags.append(valid)
        lower+=max(0,min(d-err for d,err in zip(scaled,errors)))
        upper+=min(d+err for d,err in zip(scaled,errors))
        psi+=distances[winner]
        ordered=sorted(distances);margin+=ordered[1]-ordered[0]
    return dict(labels=labels,certified_flags=flags,certified_count=sum(flags),
        psi=psi,margin_sum=margin,cost_lower_scaled=lower,cost_upper_scaled=upper,
        cost_denominator=t**6,gram_defect=g)


def exact_reference(rows, centers):
    # Python integers also cover pathological dimensions/precision in tests.
    labels=[];cost=0
    for row in rows:
        distances=[sum((int(a)-int(b))**2 for a,b in zip(row,c)) for c in centers]
        label=min(range(len(centers)),key=distances.__getitem__)
        labels.append(label);cost+=distances[label]
    return labels,cost


def summarize(model, rows, centers):
    c=certificate(model,rows,centers);labels,phi=exact_reference(rows,centers)
    wrong=sum(flag and a!=b for flag,a,b in zip(c['certified_flags'],c['labels'],labels))
    truth=phi*c['cost_denominator']
    if wrong or not c['cost_lower_scaled']<=truth<=c['cost_upper_scaled']:
        raise AssertionError('certificate contradicts exact original-space reference')
    lo,hi=c['cost_lower_scaled'],c['cost_upper_scaled']
    return dict(n=len(rows),agreement_quantized=sum(a==b for a,b in zip(c['labels'],labels))/len(rows),
        certified_count=c['certified_count'],certified_fraction=c['certified_count']/len(rows),
        certified_wrong_rows=wrong,interval_contains_exact_cost=True,
        interval_relative_width=(hi-lo)/lo if lo else None,
        decoded_cost_lower=lo/c['cost_denominator'],decoded_cost_upper=hi/c['cost_denominator'],
        quantized_true_cost=phi,gram_defect=c['gram_defect'],
        exact_label_policy_accept=c['certified_count']==len(rows),
        coverage99_cost1_policy_accept=100*c['certified_count']>=99*len(rows) and lo>0 and 100*(hi-lo)<=lo)
