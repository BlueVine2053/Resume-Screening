"""Reproducible evaluation using separate supplied training and test CSV files."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data import prepare
from src.matching import Matcher
from src.evaluate import choose_thresholds, predict, metrics, mcnemar_exact


def normalized_hash(series):
    return series.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip().map(
        lambda x: hashlib.sha256(x.encode('utf8')).hexdigest())


def leakage_audit(train, test):
    result = {}
    for col in ('pair_id', 'resume', 'job'):
        left = train[col] if col == 'pair_id' else normalized_hash(train[col])
        right = test[col] if col == 'pair_id' else normalized_hash(test[col])
        common = set(left) & set(right)
        result[col + '_overlap_unique'] = len(common)
        result[col + '_overlap_test_rows'] = int(right.isin(common).sum())
    return result


def run(train_csv, test_csv, out_dir='results/final', seed=42, test_limit=None):
    training, train_cleaning = prepare(pd.read_csv(train_csv))
    testing, test_cleaning = prepare(pd.read_csv(test_csv))
    audit = leakage_audit(training, testing)
    # Prevent exact duplicated pairs from being used in evaluation.
    testing = testing[~testing.pair_id.isin(set(training.pair_id))].reset_index(drop=True)
    if test_limit is not None:
        testing = testing.head(test_limit)
    fit, validation = train_test_split(training, test_size=.20, random_state=seed, stratify=training.label)
    model = Matcher(fit.job.tolist(), fit.resume.tolist())
    val_scores = model.score_pairs(validation.resume, validation.job)
    thresholds = {v: choose_thresholds([s[v] for s in val_scores], validation.label.tolist()) for v in ('v1', 'v2')}
    # Refit on ALL supplied training data after validation threshold selection; test remains untouched.
    model = Matcher(training.job.tolist(), training.resume.tolist())
    scores = model.score_pairs(testing.resume, testing.job)
    predicted = {v: [predict(s[v], thresholds[v]['lower'], thresholds[v]['upper']) for s in scores] for v in ('v1', 'v2')}
    truth = testing.label.tolist()
    results = {
        'seed': seed, 'training_rows': len(training), 'validation_rows': len(validation),
        'final_fit_rows': len(training), 'test_rows': len(testing),
        'training_cleaning': train_cleaning, 'test_cleaning': test_cleaning,
        'cross_file_overlap_audit': audit,
        'thresholds_selected_on_validation': thresholds,
        'metrics': {v: metrics(truth, predicted[v]) for v in ('v1','v2')},
        'paired_comparison': mcnemar_exact(truth, predicted['v1'], predicted['v2']),
        'v1_unscoreable': sum(s['v1'] is None for s in scores),
        'limitations': ['Shared resumes/jobs across different pairs may make test examples dependent; overlap counts are reported.',
                        'The test file is never used to fit the vectorizer, keyword vocabulary, or tune thresholds.',
                        'Thresholds are selected on a validation split, then matcher is refitted on training file.',
                        'Educational text-matching baseline; not for automated employment decisions.']}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    pred = testing[['label','pair_id']].copy()
    for v in ('v1','v2'):
        pred[v+'_score'] = [s[v] for s in scores]
        pred[v+'_prediction'] = predicted[v]
    pred.to_csv(out/'predictions.csv', index=False)
    joblib.dump({'matcher': model, 'thresholds': thresholds}, out/'model.joblib')
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', default='results/final')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-limit', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.train,args.test,args.out,args.seed,args.test_limit),indent=2))
