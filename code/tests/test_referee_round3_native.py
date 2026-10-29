import json
import pytest
from experiments.referee_round3_native import configurations,catalog_sizes,heldout_indices,DESIGN


def test_new_seed_grid_and_large_catalog_are_frozen_without_quality_filters():
    design=json.loads(DESIGN.read_text());grid=configurations(design)
    assert len(grid)==14 and ('mnist',20,173) in grid and ('fashion_mnist',5,223) in grid
    assert catalog_sizes(design,'mnist',5,173)==[128,512,1024,4096,16384]
    assert catalog_sizes(design,'mnist',20,173)==[1024]


def test_real_heldout_rows_are_disjoint_and_never_padded():
    train,audit=heldout_indices(20000,3000,16384,173)
    assert len(train)==3000 and len(audit)==16384
    assert len(set(audit))==16384 and not set(train).intersection(audit)
    with pytest.raises(ValueError,match='padding'):heldout_indices(5000,3000,16384,173)
