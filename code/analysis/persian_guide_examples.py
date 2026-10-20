"""Execute small educational examples and save provenance for the Persian guide.

These examples are demonstrations, not additions to the benchmark campaign.
No existing measurement or mathematical implementation is edited.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from data.loaders import load_dataset
from commitjl.projections import sparse_jl_projection
from commitjl.clustering import nearest_centroid_assign
from commitjl.metrics import clustering_agreement
from commitjl.protocol import run_protocol
from experiments.pipeline import build_projection

OUT=CODE/'results/overview/education/persian_guide'
INK='#193047';TEAL='#008A8A';ORANGE='#D37835'


def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK})
    result={'date':'2026-10-04','kind':'executed_educational_examples_not_benchmark_rows'}
    with threadpool_limits(limits=1):
        toy=np.array([[1.,1.],[2.,2.],[3.,3.]])
        model=PCA(n_components=1,svd_solver='full').fit(toy)
        z=model.transform(toy)
        assert np.allclose(np.abs(model.components_),1/np.sqrt(2))
        assert np.allclose(model.inverse_transform(z),toy)
        result['toy_pca']={'X':toy.tolist(),'mean':model.mean_.tolist(),'components':model.components_.tolist(),
            'Z':z.tolist(),'explained_variance_ratio':model.explained_variance_ratio_.tolist(),
            'sample_covariance':np.cov(toy,rowvar=False).tolist()}
        fig,ax=plt.subplots(figsize=(7,3.2))
        ax.scatter(toy[:,0],toy[:,1],s=75,c=TEAL)
        ax.plot([.5,3.5],[.5,3.5],color=ORANGE,label='First principal direction')
        for i,x in enumerate(toy):ax.annotate(f'x{i+1}',x+np.array([.08,-.15]))
        ax.set(xlabel='Feature 1',ylabel='Feature 2',xlim=(0,4),ylim=(0,4));ax.legend();ax.set_aspect('equal')
        savefig(fig,'toy_pca')

        specs=[('gaussian_mixture',{'n':1000,'d':256,'k':5,'gamma':.75,'seed':12345}),
            ('dbscan_margin',{'n':1000,'d':128,'seed':12345}),
            ('mnist',{'n_max':1000,'seed':0}),('fashion_mnist',{'n_max':1000,'seed':0}),
            ('gisette',{'n_max':1000,'seed':0}),('arcene',{'n_max':100,'seed':0}),
            ('20ng_tfidf',{'n_max':1000,'max_features':2000,'seed':0})]
        result['datasets']={}
        for name,kwargs in specs:
            data=load_dataset(name,**kwargs);X=data['X'];y=data.get('y')
            info={'shape':list(X.shape),'dtype':str(X.dtype),'first_12_features':X[0,:12].tolist(),
                'first_8_nonzero_index_value':[[int(i),float(X[0,i])] for i in np.flatnonzero(X[0])[:8]],
                'first_label':None if y is None else int(y[0]),'source':data['meta'].get('source'),
                'sha256':data['meta']['sha256'],'source_n':data['meta'].get('source_n'),
                'zero_fraction':float(np.mean(X==0))}
            result['datasets'][name]=info
            if name in ('mnist','fashion_mnist'):
                fig,axes=plt.subplots(1,6,figsize=(9,1.9))
                for i,ax in enumerate(axes):
                    ax.imshow(X[i].reshape(28,28),cmap='gray',vmin=0,vmax=255)
                    ax.set_title(f'row {i}; label {y[i]}',fontsize=8);ax.axis('off')
                savefig(fig,name+'_samples')
            if name=='mnist':
                pca=PCA(n_components=64,svd_solver='full').fit(X)
                Z=pca.transform(X)
                km=KMeans(n_clusters=5,n_init=1,random_state=0).fit(X)
                C=km.cluster_centers_;labels=nearest_centroid_assign(X,C)
                Cz=pca.transform(C);assigned=nearest_centroid_assign(Z,Cz)
                uncentered=nearest_centroid_assign(X@pca.components_.T,C@pca.components_.T)
                assert np.array_equal(assigned,uncentered)
                P=build_projection('sparse_jl',X,64,2,0,65536)['real']
                sparse=nearest_centroid_assign(X@P.T,C@P.T)
                result['mnist_pca']={'n':len(X),'d':X.shape[1],'m':64,'Z_shape':list(Z.shape),
                    'row0_first6_scores':Z[0,:6].tolist(),'variance_retained':float(pca.explained_variance_ratio_.sum()),
                    'pixel_row0_patch':X[0].reshape(28,28)[10:14,10:16].tolist(),
                    'pca_ari_to_full':clustering_agreement(labels,assigned)['ari'],
                    'sparse_ari_to_full':clustering_agreement(labels,sparse)['ari'],
                    'full_labels_first12':labels[:12].tolist(),'pca_labels_first12':assigned[:12].tolist(),
                    'sparse_labels_first12':sparse[:12].tolist(),'true_digits_first12':y[:12].tolist()}
                fig,axes=plt.subplots(1,4,figsize=(8,2.2))
                images=[X[0]]+[pca.mean_+Z[0,:m]@pca.components_[:m] for m in (8,16,64)]
                for ax,image,title in zip(axes,images,['Original','PCA m=8','PCA m=16','PCA m=64']):
                    ax.imshow(image.reshape(28,28),cmap='gray',vmin=0,vmax=255);ax.set_title(title);ax.axis('off')
                savefig(fig,'mnist_reconstruction')
                fig,ax=plt.subplots(figsize=(7,3))
                ax.scatter(Z[:,0],Z[:,1],c=km.labels_,cmap='tab10',s=8,alpha=.65)
                ax.set(xlabel='PCA component 1',ylabel='PCA component 2',title='First two PCA coordinates; colors = five full-space clusters')
                savefig(fig,'mnist_pca_plane')
            del data,X

        docs=['car engine wheel','car wheel road','space rocket orbit','space rocket moon']
        vec=TfidfVectorizer();Xt=vec.fit_transform(docs).toarray()
        result['tfidf_toy']={'documents':docs,'features':vec.get_feature_names_out().tolist(),'X':Xt.tolist()}
        assert Xt.shape[0]==4 and np.allclose(np.linalg.norm(Xt,axis=1),1)

        A=np.array([[1,0,-1,1],[0,1,0,0]],dtype=np.int64)
        x=np.array([2,3,1,4]);result['sparse_manual']={'A':A.tolist(),'x':x.tolist(),'Ax':(A@x).tolist(),'q':1}
        assert np.all(np.count_nonzero(A,axis=0)==1) and (A@x).tolist()==[5,3]
        v=np.array([1.,1.]);collapse=np.array([[1.,-1.]])
        assert np.allclose(collapse@v,0)

        # A tiny, public teaching instance: a genuine proof if the bridge succeeds.
        Xsmall=np.array([[1,1,0,0],[2,1,0,0],[0,0,2,1],[0,0,3,1]],dtype=np.int64)
        Csmall=np.array([[1,1,0,0],[0,0,2,1]],dtype=np.int64)
        started=time.perf_counter()
        protocol=run_protocol(Xsmall,Csmall,m=2,q=1,scale=1,generate_zk_proof=True,
            proof_timeout=180,challenge_fn=lambda:hashlib.sha256(b'educational-guide-challenge').digest())
        proof=protocol['proof_result']
        result['tiny_protocol']={'X':Xsmall.tolist(),'C':Csmall.tolist(),'A':protocol['A_dense'].tolist(),
            'assignments':protocol['assignments'].tolist(),'psi_hat':protocol['Psi_hat'],
            'C_X':str(protocol['C_X']),'C_theta':str(protocol['C_theta']),'C_A':str(protocol['C_A']),
            'rho_hex':protocol['rho'].hex(),'seed_hex':protocol['seed'].hex(),
            'wall_s':time.perf_counter()-started,'stage_times_s':protocol['transcript']['stage_times_s'],
            'proof_ok':bool(proof.ok),'verify_ok':bool(proof.verify_ok),'constraints':int(proof.num_constraints),
            'proof_bytes':int(proof.proof_bytes),'backend':proof.backend,'error':proof.error}
        if not (proof.ok and proof.verify_ok):
            raise RuntimeError(f'Tiny educational proof failed: {proof.error}')
        result['source_fingerprints']={str(p.relative_to(CODE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [CODE.parent/'manuscript.tex',CODE/'results/diagnostics/local_v1/measurements.parquet']}
    (OUT/'examples.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'examples':str(OUT/'examples.json'),'tiny_proof':result['tiny_protocol'],
                     'mnist_pca':result['mnist_pca']},indent=2))


if __name__=='__main__':main()
