"""Dataset loading, column validation, normalization and duplicate handling."""
import hashlib
from pathlib import Path
import pandas as pd

LABELS = ['No Fit', 'Potential Fit', 'Good Fit']
ALIASES = {
    'resume': ['resume', 'resume_text', 'resume_content', 'cv', 'candidate_resume', 'resume_str'],
    'job': ['job', 'job_description_text', 'job_description', 'job_text', 'jd', 'job_desc', 'description'],
    'label': ['label', 'fit', 'fit_label', 'match_label', 'category', 'match'],
}

def _pick(columns, requested, role):
    if requested:
        if requested not in columns:
            raise ValueError(f"Column {requested!r} not found. Available: {list(columns)}")
        return requested
    for candidate in ALIASES[role]:
        for col in columns:
            if col.lower() == candidate:
                return col
    raise ValueError(f"Could not find {role} column. Available: {list(columns)}; pass --{role}-col")

def canonical_label(value):
    key = str(value).strip().lower().replace('_', ' ').replace('-', ' ')
    labels = {'good fit':'Good Fit','potential fit':'Potential Fit','no fit':'No Fit',
              'good':'Good Fit','potential':'Potential Fit','none':'No Fit'}
    return labels.get(' '.join(key.split()))

def prepare(df, resume_col=None, job_col=None, label_col=None):
    cols = df.columns
    mapping = {role: _pick(cols, value, role) for role, value in
               [('resume', resume_col), ('job', job_col), ('label', label_col)]}
    out = df[[mapping['resume'], mapping['job'], mapping['label']]].copy()
    out.columns = ['resume', 'job', 'label']
    for field in ['resume', 'job']:
        out[field] = out[field].fillna('').astype(str).str.strip()
    out['label'] = out['label'].map(canonical_label)
    before = len(out)
    out = out[(out.resume != '') & (out.job != '') & out.label.notna()].copy()
    removed_invalid = before - len(out)
    # Normalized pair key prevents the exact same resume/job pair entering multiple splits.
    normalized = (out.resume.str.lower().str.replace(r'\s+', ' ', regex=True) + '\x1f' +
                  out.job.str.lower().str.replace(r'\s+', ' ', regex=True))
    out['pair_id'] = normalized.map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
    conflicts = out.groupby('pair_id')['label'].nunique()
    conflict_ids = set(conflicts[conflicts > 1].index)
    removed_conflicts = int(out.pair_id.isin(conflict_ids).sum())
    out = out[~out.pair_id.isin(conflict_ids)]
    removed_duplicates = int(out.pair_id.duplicated().sum())
    out = out.drop_duplicates('pair_id').reset_index(drop=True)
    if not len(out):
        raise ValueError('No valid, nonconflicting resume/job pairs remain')
    stats = {'original_rows': before, 'invalid_rows_removed': removed_invalid,
             'conflicting_label_rows_removed': removed_conflicts,
             'duplicate_pairs_removed': removed_duplicates, 'clean_rows': len(out)}
    return out, stats

def load_dataset(path=None, hf_dataset=None, hf_split='train', hf_config=None, limit=None):
    if bool(path) == bool(hf_dataset):
        raise ValueError('Choose exactly one of --data or --hf-dataset')
    if path:
        source = Path(path)
        if source.suffix.lower() == '.csv':
            df = pd.read_csv(source)
        elif source.suffix.lower() in ('.jsonl', '.json'):
            df = pd.read_json(source, lines=source.suffix.lower() == '.jsonl')
        elif source.suffix.lower() == '.parquet':
            df = pd.read_parquet(source)
        else:
            raise ValueError('Supported file types: CSV, JSONL, JSON, Parquet')
    else:
        from datasets import load_dataset as hf_load_dataset
        df = hf_load_dataset(hf_dataset, name=hf_config, split=hf_split).to_pandas()
    if limit is not None:
        if limit < 1: raise ValueError('limit must be positive')
        df = df.head(limit)
    return df
