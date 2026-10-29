"""Lightweight algebra, source-preservation and PDF checks for the new section."""
from pathlib import Path
import json,shutil
import numpy as np
from pypdf import PdfReader
from campaigns.local_fast_tradeoff import sha,save
from commitjl.reuse_certificates import integer_stats,ceil_sqrt
from commitjl.paired_certificates import change_parameters
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'paper_results/supervisor_methodology_20261027'
BUILD=REPO/'tmp/pdfs/supervisor_methodology_20261027'


def algebra():
    rng=np.random.default_rng(20261027);pairs=0;aggregates=0
    for case in range(12):
        n,d,k,s=4,7,3,3;t=16 if case%2 else 32
        x=rng.integers(-8,9,(n,d));old=rng.integers(-8,9,(k,d));new=rng.integers(-8,9,(k,d))
        b=rng.integers(-t,t+1,(s,d)).astype(object)
        model=dict(n=n,d=d,k=k,input_bits=4,basis=b.tolist(),basis_scale=t,centers=old.tolist(),
            method='learned_energy',tail_width=0,tail_rows=[],tail_signs=[])
        h,e=integer_stats(model,x);hc,ec=integer_stats(model,new);gram=b@b.T-t*t*np.eye(s,dtype=object)
        row_res=t*t*(x-old[0]).astype(object)-((x-old[0]).astype(object)@b.T)@b
        center_res=t*t*(new-old[0]).astype(object)-((new-old[0]).astype(object)@b.T)@b
        params=change_parameters(model,new,new)
        for i in range(n):
            for j in range(k):
                delta=h[i].astype(object)-hc[j].astype(object)
                compact=sum(int(v)**2 for v in delta)+int(e[i])+int(ec[j])
                distance=sum(int(v)**2 for v in x[i]-new[j])
                lhs=t*t*compact-t**6*distance
                rhs=2*t*t*int(row_res[i]@center_res[j])+int(delta@gram@delta)
                assert lhs==rhs
                for g in range(k):
                    dh=hc[j].astype(object)-hc[g].astype(object)
                    q=2*int(h[i].astype(object)@(gram-t*t*np.eye(s,dtype=object))@dh)
                    q+=t*t*(sum(int(v)**2 for v in hc[j])-sum(int(v)**2 for v in hc[g])+int(ec[j])-int(ec[g]))
                    q-=int(hc[j].astype(object)@gram@hc[j].astype(object)-hc[g].astype(object)@gram@hc[g].astype(object))
                    code_q=int(np.asarray(params['coefficients'][j*k+g],dtype=object)@h[i].astype(object)+params['constants'][j*k+g])
                    assert q==code_q
                    truth=t**6*(sum(int(v)**2 for v in x[i]-new[j])-sum(int(v)**2 for v in x[i]-new[g]))
                    residual=2*t*t*int(row_res[i]@(center_res[j]-center_res[g]))
                    assert truth==q-residual
                    error=2*t*t*ceil_sqrt(e[i])*params['residual_delta_radii'][j*k+g]
                    assert q-error<=truth<=q+error
                    pairs+=1
        od=((x[:,None,:]-old[None,:,:])**2).sum(2);nd=((x[:,None,:]-new[None,:,:])**2).sum(2)
        groups=od.argmin(1);prepared=prepare_groups(x,groups.tolist(),k)
        result=exact_old_correction_bounds(model,x,prepared,old,new)
        exact=[t**6*int(nd[i].min()-nd[i,groups[i]]) for i in range(n)]
        change=t**6*int(nd.min(1).sum()-od.min(1).sum())
        assert result['assigned_scaled']+sum(exact)==change
        lo,hi=result['change_lower_scaled'],result['change_upper_scaled'];width=hi-lo
        order=sorted(range(n),key=lambda i:(-(result['row_upper'][i]-result['row_lower'][i]),i))
        removed=0
        for i in order:
            lo+=exact[i]-int(result['row_lower'][i]);hi+=exact[i]-int(result['row_upper'][i])
            removed+=int(result['row_upper'][i])-int(result['row_lower'][i])
            assert lo<=change<=hi and hi-lo==width-removed
        assert lo==hi==change
        aggregates+=1
    return dict(cases=12,pairwise_checks=pairs,aggregate_checks=aggregates,rounded_identity=True,
        affine_coefficients_match_code=True,refinement_containment_and_width_identity=True,
        interpretation='Small synthetic algebra checks, not new performance or application evidence.')


def main():
    received=json.loads((ROOT/'received_sources_manifest.json').read_text())
    assert all(sha(REPO/name)==digest for name,digest in received['received_archives'].items())
    assert all(sha(ROOT/name)==digest for name,digest in received['sources'].items())
    log=(BUILD/'methodology_preview.log').read_text(errors='replace')
    warnings=[line for line in log.splitlines() if any(token in line for token in ['Overfull','undefined references','undefined citations','Float too large','multiply defined','LaTeX Error'])]
    assert not warnings,warnings
    source=BUILD/'methodology_preview.pdf';reader=PdfReader(source)
    assert not reader.is_encrypted
    text='\n'.join(page.extract_text() or '' for page in reader.pages)
    for phrase in ['Problem Formulation','Stage 1:','Stage 2:','Stage 3:','Stage 4:','Conditional interval correctness','Disclosure boundary']:
        assert phrase in text,'Missing '+phrase
    assert '[TODO' not in text and '[RESULT' not in text
    pdf=REPO/'output/pdf/prototype_update_methodology_20261027.pdf';shutil.copyfile(source,pdf)
    code=['code/commitjl/reuse_certificates.py','code/commitjl/enriched_jl.py','code/experiments/rank79_independent_replication.py',
          'code/commitjl/paired_certificates.py','code/commitjl/group_change_bounds.py','code/commitjl/relative_certificates.py',
          'code/zk/circuit/bound_group_compact.go','code/zk/circuit/bound_group_fingerprint.go','code/zk/circuit/bound_group_refinement.go',
          'code/zk/circuit/bound_group_switch.go']
    result=dict(status='awaiting_visual_review',pdf=pdf.relative_to(REPO).as_posix(),pdf_sha256=sha(pdf),pages=len(reader.pages),
        latex_warnings=[],algebra=algebra(),received_sources_unchanged=True,supervisor_abstract_introduction_unchanged=True,
        methodology_uses_jl=False,current_profile=dict(prediction_rank=63,certificate_rank=79,basis_scale=2**14,tail_width=0,maximum_exact_rows=16),
        code_source_sha256={name:sha(REPO/name) for name in code},source_sha256={p.name:sha(p) for p in ROOT.glob('*.tex')},
        visual_review='PENDING',not_a_full_security_audit=True,compatible_abstract_introduction_required_before_full_paper_merge=True)
    save(ROOT/'methodology_checks.json',result);print(json.dumps(dict(pages=result['pages'],pdf_sha256=result['pdf_sha256'],algebra=result['algebra'])))


if __name__=='__main__':main()
