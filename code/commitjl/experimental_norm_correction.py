"""Experimental nonlinear postprocessing; not the canonical linear/ZK method."""
import numpy as np


def residual_norm_corrected_transform(model,points):
    """Retain protected coordinates and restore each original residual norm.

    Requires reading original point norms, which must be charged to conversion.
    Pairwise distances are not guaranteed exact. Original protected-center
    distances are exact in real arithmetic. No linear JL or proof claim applies.
    An annihilated positive residual is rejected, never replaced silently.
    """
    if model.method!='hybrid':raise ValueError('Requires a protected/JL hybrid')
    x=np.asarray(points,dtype=float);z=model.transform(x)
    rank=model.basis.shape[1];shifted=x-model.origin
    original_norm=np.einsum('ij,ij->i',shifted,shifted)
    head_norm=np.einsum('ij,ij->i',z[:,:rank],z[:,:rank])
    residual=original_norm-head_norm
    tolerance=64*np.finfo(float).eps*original_norm
    if np.any(residual < -tolerance):raise ValueError('Invalid orthogonal residual energy')
    residual=np.maximum(residual,0);residual[residual<=tolerance]=0
    tail_norm=np.einsum('ij,ij->i',z[:,rank:],z[:,rank:])
    positive=residual>0
    if np.any(positive & (tail_norm<=np.finfo(float).tiny)):
        raise ValueError('Sketch annihilated a positive residual; norm correction undefined')
    scale=np.zeros(len(x));scale[positive]=np.sqrt(residual[positive])/np.sqrt(tail_norm[positive])
    z[:,rank:]*=scale[:,None]
    if not np.isfinite(z).all():raise ValueError('Norm-corrected projection overflow')
    return z
