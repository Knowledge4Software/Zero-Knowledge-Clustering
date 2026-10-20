import pandas as pd
from validation.check_layout import provenance_flags


def test_preserved_historical_results_are_identified_separately_from_current_source():
    data=pd.DataFrame({'_source_sha256':['old','old']})
    flags=provenance_flags(data,'current','old','original-bytes','original-bytes')
    assert not flags['source_matches']
    assert flags['recorded_source_matches'] and flags['recorded_bytes_match']


def test_historical_record_rejects_changed_bytes_or_mixed_provenance():
    data=pd.DataFrame({'_source_sha256':['old','old']})
    assert not provenance_flags(data,'current','old','changed','original')['recorded_bytes_match']
    data.loc[1,'_source_sha256']='current'
    assert not provenance_flags(data,'current','old','original','original')['recorded_source_matches']
