# Intelligent Resume Screening and Job Matching System

A Python-based application that evaluates resume-to-job matching using natural language processing and machine learning techniques.

Educational project comparing **V1 keyword overlap** with **V2 TF-IDF cosine similarity** on labeled resume–job pairs. It uses Python/scikit-learn locally and **does not make AI API calls**. Scores are not hiring decisions.

## Repository contents

| Path | Purpose |
|---|---|
| `src/matching.py` | Shared V1 and V2 scoring methods |
| `src/data.py` | Dataset loading, column mapping, validation and pair deduplication |
| `src/evaluate.py` | General split-based evaluation and statistical comparison |
| `src/final_experiment.py` | Final evaluation using **separate** training and test CSV files, overlap audit and model export |
| `app.py` | Optional Streamlit demonstration; loads the final saved matcher if present |
| `tests/` | 18 automated tests |
| `data/train.csv`, `data/test.csv` | **Not included**; download these from Hugging Face before rerunning the experiment |
| `data/sample_pairs.csv` | Small synthetic demo dataset; **not** research evidence |
| `results/final/` | Measured metrics, all 1,759 predictions, saved matcher and test report |

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Reproduce the final experiment

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final
```

The upstream dataset files use the columns `resume_text`, `job_description_text`, and `label` with `Good Fit`, `Potential Fit`, and `No Fit` labels. The command validates the data, removes duplicate pairs, uses a stratified 80/20 split **within the training file** for threshold selection, then refits on the full cleaned training file. The separate test file is never used to fit the keyword list, vectorizer or thresholds. It exports `metrics.json`, `predictions.csv`, and `model.joblib`.

## Dataset source and setup

The original dataset is hosted on Hugging Face: **[cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)**.

**Original training and test CSVs are deliberately excluded from this GitHub repository.** Download the dataset's `train.csv` and `test.csv` from the **Files and versions** tab and place them here:

```text
data/train.csv
data/test.csv
```

If the downloaded files are named `train(1).csv` and `test(1).csv`, rename them to `train.csv` and `test.csv`. Keep the column names `resume_text`, `job_description_text`, and `label`. Then run:

```bash
python -m pytest -q
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final
streamlit run app.py
```

The included `data/sample_pairs.csv` is synthetic and can be used without downloading the Hugging Face dataset. The final evaluation files in `results/final/` are recorded results from a previous run using the upstream training and test CSVs. Dataset revisions can change; to reproduce the exact numbers, use the same source files and software environment. Check the dataset's license and terms before redistributing its contents.

## Recorded results

Results in `results/final/` are from **1,759 separate test pairs**, with 6,228 cleaned training pairs and 1,246 validation pairs (a subset of training).

| Metric | V1 keyword | V2 TF-IDF |
|---|---:|---:|
| Accuracy | 39.80% | 44.57% |
| Macro precision | 33.32% | 37.52% |
| Macro recall | 33.32% | 35.16% |
| Macro F1 | 32.81% | 32.10% |

Paired exact McNemar p-value: **0.0002114451**. V2 is better on overall accuracy, but not macro F1. See `results/final/TEST_REPORT.md` and `metrics.json` for details.

**Critical limitation:** 1,758 of the 1,759 test rows contain normalized resume text also seen in training. There were zero exact duplicate resume–job pairs and zero exact job-description overlaps across files. These are **pair-held-out results**, not evidence of generalization to entirely unseen candidates. A candidate-grouped evaluation would be required to make that claim.

## Run the Streamlit application

```bash
streamlit run app.py
```

When `results/final/model.joblib` exists, Streamlit loads the saved trained matcher and uses the same `Matcher.score()` implementation as the evaluation. Without that file, the app clearly identifies its synthetic demo fallback. The interface shows matching scores rather than hiring recommendations.

## Verification and remaining work

- **18 automated tests passed** in the recorded run; the tests were rerun while preparing this package.
- Full evaluation was previously repeated with byte-identical metrics JSON and prediction CSV using the same data and settings.
- Overlap auditing and saved-model scoring checks are included in automated tests.
- **Browser-level Streamlit testing remains pending**; do not report it as passed.
- A grouped split preventing reuse of resumes across training and testing has **not** been implemented or evaluated.

For configuration management, commit changes to Git, use GitHub Issues for defects, and tag reviewed releases. For strict reproducibility, capture Python/library versions and the upstream dataset revision used for the experiment.

## Other evaluation mode

`python -m src.evaluate --data data/sample_pairs.csv --out results` provides a general split-based educational demonstration. It is **not** the source of the final reported Hugging Face metrics. The final report comes from `src.final_experiment` with the separately downloaded train/test CSVs.
