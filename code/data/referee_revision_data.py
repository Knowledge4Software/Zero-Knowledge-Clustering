"""Small, pinned public learning datasets for the 27 October referee revision.

The transformer is inference-only. Its embeddings are the committed input;
the audit does not certify execution of the encoder or text confidentiality.
"""
from pathlib import Path
import argparse,hashlib,io,json,time,urllib.request,zipfile
import numpy as np
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
MODEL='sentence-transformers/all-MiniLM-L6-v2'
HAR_URL='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'


def split_indices(count,total,seed):
    if count<total:raise ValueError('Insufficient distinct real records')
    return np.random.default_rng(seed).permutation(count)[:total]


def har(out):
    raw=out/'uci_har_source.zip'
    if not raw.exists():
        with urllib.request.urlopen(HAR_URL,timeout=60) as response,raw.with_suffix('.download').open('wb') as stream:
            while block:=response.read(2**20):stream.write(block)
        raw.with_suffix('.download').replace(raw)
    archive=zipfile.ZipFile(raw)
    nested=next((n for n in archive.namelist() if n.lower().endswith('uci har dataset.zip')),None)
    if nested:archive=zipfile.ZipFile(io.BytesIO(archive.read(nested)))
    def load(suffix,dtype=float):
        names=[n for n in archive.namelist() if n.endswith(suffix)]
        if len(names)!=1:raise ValueError('Ambiguous official HAR file: '+suffix)
        with archive.open(names[0]) as stream:return np.loadtxt(stream,dtype=dtype)
    train=load('train/X_train.txt');test=load('test/X_test.txt')
    train_y=load('train/y_train.txt',int);test_y=load('test/y_test.txt',int)
    train_subject=load('train/subject_train.txt',int);test_subject=load('test/subject_test.txt',int)
    if set(train_subject)&set(test_subject):raise ValueError('Official subject split overlaps')
    a=split_indices(len(train),2000,20261027);b=split_indices(len(test),2048,20261028)
    path=out/'har561.npz';np.savez_compressed(path,training=train[a],audit=test[b],training_y=train_y[a],audit_y=test_y[b],training_subject=train_subject[a],audit_subject=test_subject[b])
    save(out/'har561_source.json',dict(source=HAR_URL,source_zip_sha256=sha(raw),archive_sha256=sha(path),
        dimensions=561,training_records=2000,audit_records=2048,official_subject_split_disjoint=True,
        training_index_sha256=hashlib.sha256(a.tobytes()).hexdigest(),audit_index_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        description='Public sensor-feature benchmark, not private participant data; within-subject windows remain correlated.'))


def neural_text(out):
    from huggingface_hub import HfApi,hf_hub_download
    from transformers import AutoTokenizer
    import onnxruntime as ort
    from sklearn.datasets import fetch_20newsgroups
    meta=HfApi().model_info(MODEL,files_metadata=True);revision=meta.sha
    candidates=[v.rfilename for v in meta.siblings]
    names=['onnx/model_qint8_avx2.onnx','onnx/model_quantized.onnx','onnx/model.onnx']
    filename=next((n for n in names if n in candidates),None)
    if filename is None:raise ValueError('No approved CPU ONNX model file is available')
    model=Path(hf_hub_download(MODEL,filename,revision=revision))
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,trust_remote_code=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    features={};labels={};index_hashes={};text_hashes={}
    for split,total,seed in [('train',2000,20261027),('test',2048,20261028)]:
        source=fetch_20newsgroups(subset=split,remove=('headers','footers','quotes'),data_home=str(out/'20news_source'))
        keep=np.asarray([i for i,t in enumerate(source.data) if t.strip()],dtype=np.int64)
        selected=keep[split_indices(len(keep),total,seed)];values=[]
        corpus_digest=hashlib.sha256()
        for start in range(0,total,16):
            texts=[source.data[int(i)] for i in selected[start:start+16]]
            for t in texts:
                encoded=t.encode('utf-8');corpus_digest.update(len(encoded).to_bytes(8,'big'));corpus_digest.update(encoded)
            batch=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='np')
            inputs={v.name:np.asarray(batch[v.name],dtype=np.int64) for v in session.get_inputs()}
            hidden=session.run(None,inputs)[0];mask=np.asarray(batch['attention_mask'],dtype=np.float32)[:,:,None]
            pooled=(hidden*mask).sum(1)/np.maximum(mask.sum(1),1)
            pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
            values.append(pooled.astype(np.float32))
            if start%256==0:print(json.dumps(dict(stage='neural_text_encoder',split=split,encoded=start,total=total)),flush=True)
        features[split]=np.concatenate(values);labels[split]=np.asarray(source.target)[selected]
        index_hashes[split]=hashlib.sha256(selected.tobytes()).hexdigest();text_hashes[split]=corpus_digest.hexdigest()
    path=out/'text_minilm384.npz';np.savez_compressed(path,training=features['train'],audit=features['test'],training_y=labels['train'],audit_y=labels['test'])
    save(out/'text_minilm384_source.json',dict(dataset='20 Newsgroups, official train/test, nonempty after removing headers/footers/quotes',
        encoder_repository=MODEL,encoder_revision=revision,encoder_file=filename,encoder_sha256=sha(model),archive_sha256=sha(path),
        dimensions=384,training_records=2000,audit_records=2048,token_limit=128,batch=16,pooling='Attention-mask mean, then L2 normalization',
        onnxruntime=ort.__version__,index_sha256=index_hashes,text_sha256=text_hashes,
        scope='Public text surrogate for a committed neural-feature catalog. Audit certificates do not prove encoder inference or prevent inference from disclosed aggregates.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dataset',choices=['har','text','all'],default='all');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);started=time.time()
    for name,action,archive in [('har',har,'har561.npz'),('text',neural_text,'text_minilm384.npz')]:
        if args.dataset not in (name,'all'):continue
        if (out/archive).exists():
            source=out/(Path(archive).stem+'_source.json')
            if not source.exists() or json.loads(source.read_text())['archive_sha256']!=sha(out/archive):raise ValueError('Existing data archive is not pinned')
            continue
        action(out)
    save(out/'acquisition_status.json',dict(complete=True,wall_seconds=time.time()-started,requested=args.dataset))


if __name__=='__main__':main()
