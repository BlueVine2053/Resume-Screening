"""Held-out evaluation: train/validation/test, threshold selection and McNemar test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.data import LABELS, load_dataset, prepare
from src.matching import Matcher

# Thresholds map [0, lower) to No Fit, [lower, upper) to Potential Fit, [upper, 1] to Good Fit.
def predict(score, lower, upper):
    if score is None or pd.isna(score): return 'No Fit'  # Explicit fallback for V1 unscoreable pairs
    return LABELS[0] if score < lower else LABELS[1] if score < upper else LABELS[2]

def choose_thresholds(scores, true_labels):
    # Select ONLY on validation data. Deterministic ties prefer first pair.
    best = (-1, 0.30, 0.65)
    for lower in np.arange(.10, .81, .05):
        for upper in np.arange(lower + .05, .96, .05):
            predictions = [predict(s, lower, upper) for s in scores]
            from sklearn.metrics import f1_score
            f1 = f1_score(true_labels, predictions, labels=LABELS, average='macro', zero_division=0)
            if f1 > best[0] + 1e-12: best = (float(f1), round(float(lower), 4), round(float(upper), 4))
    return {'validation_macro_f1': best[0], 'lower': best[1], 'upper': best[2]}

def split_data(df, seed=42):
    if df.label.value_counts().min() < 4:
        raise ValueError('Need at least 4 examples of each class for stratified train/validation/test splits')
    train, remaining = train_test_split(df, test_size=.40, random_state=seed, stratify=df.label)
    validation, test = train_test_split(remaining, test_size=.50, random_state=seed, stratify=remaining.label)
    for a, b in [(train, validation), (train, test), (validation, test)]:
        assert set(a.pair_id).isdisjoint(set(b.pair_id)), 'Data leakage: duplicate pair across splits'
    return train, validation, test

def metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'per_class': {label: {key: float(report[label][key]) for key in ('precision','recall','f1-score','support')}
                          for label in LABELS},
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}

def mcnemar_exact(y_true, v1_predictions, v2_predictions):
    # Paired correctness; exact two-sided binomial test on discordant predictions.
    b = sum(a == y and c != y for y, a, c in zip(y_true, v1_predictions, v2_predictions))
    c = sum(a != y and d == y for y, a, d in zip(y_true, v1_predictions, v2_predictions))
    p = float(binomtest(min(b, c), n=b+c, p=.5).pvalue) if b+c else 1.0
    return {'v1_only_correct': b, 'v2_only_correct': c, 'discordant': b+c, 'p_value': p,
            'method': 'Exact McNemar (binomial on paired correctness)',
            'significant_at_0_05': p < .05}

def run(df, out_dir='results', seed=42, resume_col=None, job_col=None, label_col=None):
    clean, cleaning = prepare(df, resume_col, job_col, label_col)
    train, validation, test = split_data(clean, seed)
    model = Matcher(train.job.tolist(), train.resume.tolist())
    val_scores = model.score_pairs(validation.resume, validation.job)
    test_scores = model.score_pairs(test.resume, test.job)
    thresholds = {version: choose_thresholds([s[version] for s in val_scores], validation.label.tolist())
                  for version in ('v1', 'v2')}
    predictions = {version: [predict(s[version], thresholds[version]['lower'], thresholds[version]['upper'])
                             for s in test_scores] for version in ('v1', 'v2')}
    truth = test.label.tolist()
    result = {'seed': seed, 'cleaning': cleaning, 'dataset_split': {'train':len(train), 'validation':len(validation), 'test':len(test)},
              'labels': LABELS, 'thresholds_selected_on_validation': thresholds,
              'test_metrics': {v: metrics(truth, predictions[v]) for v in ('v1','v2')},
              'paired_statistical_comparison': mcnemar_exact(truth, predictions['v1'], predictions['v2']),
              'v1_unscoreable_test_pairs': sum(s['v1'] is None for s in test_scores),
              'notes': ['All test metrics use the held-out test split only.',
                        'The V1 fallback for unscoreable pairs is No Fit.',
                        'Thresholds are tuned using validation only.',
                        'Pair-level duplicate leakage is checked; shared resumes or job descriptions across pairs are not fully excluded.',
                        'This is an educational baseline, not an automated hiring system.']}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = test[['resume','job','label']].copy()
    for v in ('v1','v2'):
        rows[f'{v}_score'] = [s[v] for s in test_scores]
        rows[f'{v}_prediction'] = predictions[v]
    rows.to_csv(out/'test_predictions.csv', index=False)
    (out/'keyword_vocabulary.txt').write_text('\n'.join(sorted(model.vocabulary))+'\n', encoding='utf-8')
    return result

def main():
    parser = argparse.ArgumentParser(description='Evaluate V1 and V2 on a held-out test set')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', help='CSV, JSONL, JSON or Parquet path')
    source.add_argument('--hf-dataset', help='Hugging Face dataset repository ID')
    parser.add_argument('--hf-split', default='train'); parser.add_argument('--hf-config')
    parser.add_argument('--resume-col'); parser.add_argument('--job-col'); parser.add_argument('--label-col')
    parser.add_argument('--limit', type=int); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    df = load_dataset(args.data, args.hf_dataset, args.hf_split, args.hf_config, args.limit)
    result = run(df, args.out, args.seed, args.resume_col, args.job_col, args.label_col)
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.out}/evaluation.json and {args.out}/test_predictions.csv")
if __name__ == '__main__': main()
