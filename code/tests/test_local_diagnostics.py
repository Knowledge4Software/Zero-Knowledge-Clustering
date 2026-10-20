"""Independent checks for diagnostic denominators and conditional claims."""
import json
import math

import numpy as np

from campaigns.diagnostic import atomic_json, centroid_counts, jobs, rate, wilson


def test_empty_rates_are_missing_not_zero(tmp_path):
    assert math.isnan(rate(0, 0))
    path=tmp_path/'row.json'
    atomic_json(path, {'rate':rate(0,0), 'changes':np.int64(0)})
    assert json.loads(path.read_text())=={'rate':None,'changes':0}


def test_conditional_certificate_is_not_margin_only():
    X=np.array([[0.1],[9.9]])
    C=np.array([[0.],[10.]])
    # Deliberately falsify the embedding premise and reverse assignments.
    out=centroid_counts(X,C,X[::-1],C,.15)
    assert out['margin_count']==2
    assert out['eligible_count']==0
    assert out['margin_changes_without_premise']==2
    assert math.isnan(out['conditional_violation_rate'])
    identity=centroid_counts(X,C,X,C,.15)
    assert identity['eligible_count']==2
    assert identity['eligible_changes']==0


def test_wilson_accounts_for_finite_samples_at_zero_and_one():
    low,high=wilson(0,32)
    assert 0<=low<1e-12 and 0<high<1
    low,high=wilson(32,32)
    assert 0<low<1 and high==1


def test_regular_plan_retains_ten_seeds_and_three_sources():
    plan=jobs(False)
    assert all(j['seeds']==list(range(10)) for j in plan if j['kind']!='overhead')
    timing=[j for j in plan if j['kind']=='overhead']
    assert len(timing)==90
    assert {j['source'] for j in timing}=={'nonce','beacon','commit_reveal'}
