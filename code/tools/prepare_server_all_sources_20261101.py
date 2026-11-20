"""Freeze cached wide datasets and training-only TF-IDF without downloads."""
from pathlib import Path
import json
import shutil
import time

import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from commitjl.pilot_io import array_sha, save, sha

REPO = Path(__file__).resolve().parents[2]
HOME = REPO/'code/.cache/commitjl_data'
TARGET = REPO/'server_inputs/all_sources_20261101'
SEEDS = [701, 709, 719]


def prepare():
    TARGET.mkdir(parents=True, exist_ok=True)
    if (TARGET/'pin.json').exists():
        pin = json.loads((TARGET/'pin.json').read_text())
        for name, digest in pin['files'].items():
            assert sha(TARGET/name) == digest
        return
    provenance = {}
    specifications = [
        ('gisette', HOME/'gisette/gisette_train.data', 3000,
         '7a0d620f585ebbdcd490c01cb6840d4e4a9b127c03e3a3ebd91eaea2711fd60e'),
        ('arcene', HOME/'arcene/ARCENE/arcene_train.data', 100,
         '4fac809d28bd91101cd5cfcc95a2a6ef84ece3bb04960c9b138f9422a923f375')]
    for name, path, rows, expected in specifications:
        started = time.monotonic()
        matrix = np.loadtxt(path)
        ids = (np.random.default_rng(0).choice(len(matrix), rows, replace=False)
               if rows < len(matrix) else np.arange(len(matrix)))
        selected = np.ascontiguousarray(matrix[ids], dtype=np.float64)
        assert array_sha(selected) == expected
        compact = selected.astype(np.uint16)
        assert np.array_equal(selected, compact.astype(np.float64)), 'Lossy feature conversion'
        np.savez_compressed(TARGET/(name+'.npz'), X=compact, original_row_ids=ids)
        provenance[name] = dict(raw_source=str(path.relative_to(REPO)), raw_sha256=sha(path),
             canonical_float64_sha256=expected, shape=list(selected.shape),
             storage='Lossless uint16; restore float64 before checking canonical hash.',
             preparation_seconds=time.monotonic()-started)
        print(json.dumps(dict(prepared=name, shape=selected.shape)), flush=True)
    started = time.monotonic()
    corpus = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'),
                               data_home=str(HOME), download_if_missing=False)
    eligible = np.array([i for i, text in enumerate(corpus.data) if text.strip()], dtype=np.int64)
    for seed in SEEDS:
        tick = time.monotonic()
        ids = eligible[np.random.default_rng(741+seed).permutation(len(eligible))]
        train_ids, audit_ids = ids[:1024], ids[1024:2048]
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        with threadpool_limits(limits=1):
            vectorizer.fit([corpus.data[i] for i in train_ids[:512]])
            training = vectorizer.transform([corpus.data[i] for i in train_ids]).toarray()
            audit = vectorizer.transform([corpus.data[i] for i in audit_ids]).toarray()
        assert training.shape == audit.shape == (1024, 2000)
        filename = f'text_tfidf_seed{seed}.npz'
        np.savez_compressed(TARGET/filename, training=training, audit=audit,
                            training_ids=train_ids, audit_ids=audit_ids)
        terms = vectorizer.get_feature_names_out().tolist()
        save(TARGET/f'text_tfidf_seed{seed}_preprocessing.json', dict(
             seed=seed, corpus='20 Newsgroups official training source; a second representation of the existing text corpus.',
             preprocessing_training_ids=train_ids[:512].tolist(),
             training_ids=train_ids.tolist(), audit_ids=audit_ids.tolist(),
             vocabulary=terms, idf=vectorizer.idf_.tolist(), sklearn_version=sklearn.__version__,
             training_sha256=array_sha(training), audit_sha256=array_sha(audit),
             vectorizer_fitting_scope='Only the 512 past-training documents; incoming and audit documents never fit vocabulary/IDF.',
             preparation_seconds=time.monotonic()-tick))
        print(json.dumps(dict(prepared=f'text_tfidf_seed{seed}')), flush=True)
    provenance['text_tfidf'] = dict(cached_corpus_sha256=sha(HOME/'20news-bydate_py3.pkz'),
         selected_from='Official training corpus, nonempty text; disjoint document IDs within each fit.',
         preparation_seconds=time.monotonic()-started)
    encoder = REPO/'code/validation/outputs/referee_two_reviews_20261027/neural_activity/encoder_seed307.npz'
    shutil.copyfile(encoder, TARGET/'har_encoder_seed307.npz')
    provenance['har_neural256'] = dict(encoder_sha256=sha(encoder),
         encoder_source=str(encoder.relative_to(REPO)),
         scope='Frozen historical encoder trained on official training-source rows; new basis/center fits; encoder inference is outside the proof.',
         representation_only=True)
    save(TARGET/'pin.json', dict(created_unix=time.time(), source_policy='Cached sources only; no download or new paid service.',
         provenance=provenance,
         files={p.name:sha(p) for p in sorted(TARGET.iterdir()) if p.is_file() and p.name!='pin.json'}))


if __name__ == '__main__':
    prepare()
