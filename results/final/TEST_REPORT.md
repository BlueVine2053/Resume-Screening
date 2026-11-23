# Intelligent Resume Screening and Job Matching System — Test Report

## Inputs and execution

- Training file: `train(1).csv` (6,241 raw rows; 6,228 clean distinct nonconflicting pairs)
- Test file: `test(1).csv` (1,759 raw rows; 1,759 clean pairs)
- Validation: 20% of the cleaned training file, for threshold selection only
- Final matcher refit on cleaned training file; external test file never used to fit vectorizer, keyword vocabulary, or thresholds

## Results (all 1,759 test pairs)

| Metric | V1 keyword | V2 TF-IDF |
|---|---:|---:|
| Accuracy | 39.80% | 44.57% |
| Macro precision | 33.32% | 37.52% |
| Macro recall | 33.32% | 35.16% |
| Macro F1 | 32.81% | 32.10% |

Exact McNemar p-value: **0.0002114451**; V1-only correct: 210, V2-only correct: 294.

The comparison favors V2 on accuracy, but not on macro F1.

## Leakage audit

- Exact pair overlap across training/test: 0 test rows
- Exact normalized job-description overlap: 0 test rows
- Exact normalized resume overlap: **1,758 / 1,759 test rows** (476 unique overlapping resumes)

**Major limitation:** This is a held-out *pair* test, not a held-out *candidate* test. The substantial resume reuse limits claims about generalization to new candidates. Do not describe it as leakage-free in a candidate-independent sense.

## Automated verification

- 18 automated tests passed, including column mapping, overlap auditing, and saved-model scoring consistency
- Repeated full evaluation with the same inputs and seed produced byte-identical metrics JSON and prediction CSV
- Python syntax compilation passed
- Streamlit browser-level testing was **not executed in the automated test environment** because Streamlit was unavailable there. Shared matching function and saved-model loading logic are implemented.

## Manual Streamlit application testing

I manually tested the Streamlit application, including entering resume and job-description text, viewing matching results, and checking the response to blank or whitespace-only input.

The application displayed an error rather than returning a matching score for invalid blank input.

This manual testing complements the automated Python tests. It is recorded here as completed application testing; screenshots and a separate browser-test log are not included in this report.

It does not establish that similarity scores measure actual qualifications, that users understand the score warnings, or that the model generalizes to unseen resumes.

**Verification distinction:** The 18 automated tests and reproducible evaluation results were executed in the test environment described above. The manual Streamlit interactions were performed separately.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final
streamlit run app.py
```
