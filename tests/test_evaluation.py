import json
import pandas as pd
import pytest
from src.data import prepare
from src.evaluate import mcnemar_exact, predict, run, split_data

@pytest.fixture
def data():
    # Create 45 sample resume-job pairs across three classification categories.
    rows = []
    for i in range(15):
        rows.extend([
            {'resume': f'python sql candidate {i}', 'job': f'python sql analyst role {i}', 'label':'Good Fit'},
            {'resume': f'python candidate {i}', 'job': f'python sql analyst role {i}', 'label':'Potential Fit'},
            {'resume': f'nursing candidate {i}', 'job': f'python sql analyst role {i}', 'label':'No Fit'},
        ])
    return pd.DataFrame(rows)

def test_prepare_removes_duplicates_and_invalid():
    df = pd.DataFrame([{'resume':'Python','job':'SQL','label':'Good Fit'},
                       {'resume':' python ','job':' sql ','label':'Good Fit'},
                       {'resume':'','job':'SQL','label':'No Fit'}])
    clean, stats = prepare(df)
    assert len(clean) == 1 and stats['duplicate_pairs_removed'] == 1
    assert stats['invalid_rows_removed'] == 1

def test_conflicting_labels_are_removed():
    df = pd.DataFrame([{'resume':'a','job':'b','label':'Good Fit'},
                       {'resume':'a','job':'b','label':'No Fit'},
                       {'resume':'c','job':'d','label':'Potential Fit'}])
    clean, stats = prepare(df)
    assert len(clean) == 1 and stats['conflicting_label_rows_removed'] == 2

def test_split_is_disjoint(data):
    clean, _ = prepare(data)
    train, val, test = split_data(clean)
    assert not set(train.pair_id) & set(val.pair_id)
    assert not set(train.pair_id) & set(test.pair_id)
    assert not set(val.pair_id) & set(test.pair_id)

def test_threshold_boundaries():
    assert predict(None, .3, .7) == 'No Fit'
    assert predict(.3, .3, .7) == 'Potential Fit'
    assert predict(.7, .3, .7) == 'Good Fit'

def test_mcnemar_exact():
    r = mcnemar_exact(['Good Fit']*4, ['Good Fit']*4, ['No Fit']*4)
    assert r['v1_only_correct'] == 4 and r['v2_only_correct'] == 0
    assert r['p_value'] == pytest.approx(.125)

def test_t8_reproducible_evaluation(data, tmp_path):
    a = run(data, tmp_path/'a', seed=42)
    b = run(data, tmp_path/'b', seed=42)
    assert a == b
    assert (tmp_path/'a'/'evaluation.json').exists()
    assert (tmp_path/'a'/'test_predictions.csv').exists()

def test_unrecognized_label_rejected():
    df = pd.DataFrame([{'resume':'a','job':'b','label':'unknown'}])
    with pytest.raises(ValueError): prepare(df)
