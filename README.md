# Intelligent Resume Screening and Job Matching System

A Python-based application that evaluates resume-to-job matching using natural language processing and machine learning techniques.

This project compares **V1 keyword overlap** with **V2 TF-IDF cosine similarity** on labeled resume–job pairs. It uses Python/scikit-learn locally and **does not make AI API calls**. Scores are not hiring decisions.

## Repository Contents

| Path | Purpose |
|---|---|
| `src/matching.py` | Shared V1 and V2 scoring methods |
| `src/data.py` | Dataset loading, column mapping, validation, and pair deduplication |
| `src/evaluate.py` | General split-based evaluation and statistical comparison |
| `src/final_experiment.py` | Final evaluation using separate training and test CSV files, overlap auditing, and model export |
| `app.py` | Streamlit application for demonstrating resume-to-job matching |
| `tests/` | 19 automated Python tests |
| `data/train.csv`, `data/test.csv` | Original dataset files downloaded separately from Hugging Face |
| `data/sample_pairs.csv` | Synthetic sample dataset for demonstration |
| `results/final/` | Recorded evaluation metrics and test report; generated predictions and saved model are excluded from GitHub |
| `config.json` | Project metadata and matching-method descriptions |
| `requirements.txt` | Required Python dependencies |

## Installation

Requires Python 3.10 or later.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run automated tests:

```bash
python -m pytest -q
```

## Matching Methods

The application implements two resume-to-job matching approaches.

**V1 — Keyword Overlap Matching**

Compares resume text against job-description keywords and calculates a matching score based on keyword overlap.

**V2 — TF-IDF Cosine Similarity**

Uses Term Frequency–Inverse Document Frequency (TF-IDF) to represent resume and job-description text numerically, then calculates cosine similarity to measure their textual similarity.

Both methods are evaluated against labeled resume–job pairs to compare their performance.

## Dataset Source and Setup

The dataset is available on Hugging Face:

[cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

Download `train.csv` and `test.csv` from the dataset's **Files and versions** tab.

Save the files in the following locations:

```text
data/train.csv
data/test.csv
```

If the downloaded files have names such as `train(1).csv` and `test(1).csv`, rename them to `train.csv` and `test.csv`.

The expected dataset columns are:

- `resume_text`
- `job_description_text`
- `label`

The classification labels are:

- `Good Fit`
- `Potential Fit`
- `No Fit`

The original dataset files are excluded from the GitHub repository. The included `data/sample_pairs.csv` contains synthetic examples for demonstration and is not the source of the final evaluation results.

Dataset revisions can affect reproducibility. The recorded results correspond to the source files and environment used during the evaluation.

## Reproduce the Final Experiment

Run the following command after downloading the dataset:

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final
```

The evaluation process:

1. Loads and validates the training and test datasets.
2. Removes duplicate resume–job pairs.
3. Uses a stratified 80/20 split within the cleaned training dataset for threshold selection.
4. Refits the final matcher using the complete cleaned training dataset.
5. Evaluates V1 and V2 against the separate test dataset.
6. Calculates classification metrics and statistical comparison results.
7. Exports evaluation metrics, predictions, and the saved matcher.

The external test dataset is not used to fit the keyword vocabulary, TF-IDF vectorizer, or classification thresholds.

Generated output files include:

```text
results/final/metrics.json
results/final/predictions.csv
results/final/model.joblib
```

The generated prediction CSV and saved model are excluded from GitHub.

## Recorded Evaluation Results

The final evaluation used:

| Dataset | Number of pairs |
|---|---:|
| Cleaned training dataset | 6,228 |
| Validation subset of training | 1,246 |
| Separate test dataset | 1,759 |

### Performance Comparison

| Metric | V1 Keyword | V2 TF-IDF |
|---|---:|---:|
| Accuracy | 39.80% | 44.57% |
| Macro Precision | 33.32% | 37.52% |
| Macro Recall | 33.32% | 35.16% |
| Macro F1 | 32.81% | 32.10% |

The paired exact McNemar test produced a p-value of **0.0002114451**.

V2 achieved higher overall accuracy than V1. However, V1 achieved a slightly higher macro F1 score.

These results demonstrate differences between keyword-based and TF-IDF-based text matching on the evaluated dataset. They do not establish that the matching scores represent actual candidate qualifications.

### Dataset Overlap Analysis

The evaluation also included an overlap audit between the training and test datasets.

| Overlap category | Test rows |
|---|---:|
| Exact resume–job pair overlap | 0 |
| Exact normalized job-description overlap | 0 |
| Exact normalized resume overlap | 1,758 of 1,759 |

The overlap analysis identified 476 unique normalized resumes shared between training and testing.

**Evaluation limitation:** The reported results measure performance on held-out resume–job pairs, not entirely unseen candidates. The substantial reuse of resume text limits conclusions about generalization to new candidates.

## Run the Streamlit Application

Start the application:

```bash
streamlit run app.py
```

The Streamlit interface provides an interactive demonstration of resume-to-job matching.

Users can enter resume text and a job description to view matching scores.

When `results/final/model.joblib` is available, the application loads the saved trained matcher and uses the same matching implementation as the evaluation.

When the saved matcher is unavailable, the application identifies its synthetic demonstration mode.

The application displays similarity scores rather than hiring recommendations.

## Testing and Verification

- **19 automated Python tests passed** during project verification.
- Automated tests cover data validation, column mapping, overlap auditing, saved-model scoring consistency, and batch input-length validation.
- The full evaluation was repeated using the same inputs and settings, producing byte-identical metrics JSON and prediction CSV files.
- Python syntax compilation passed.
- Manual Streamlit application testing covered resume and job-description text entry, matching results, and blank-input validation.
- The application displayed an error instead of returning a matching score for invalid blank input.
- Detailed evaluation and testing information is documented in `results/final/TEST_REPORT.md`.

## Configuration Management

The project uses Git and GitHub for source-code version control and configuration management.

The repository includes:

- Source-code files and application dependencies.
- Automated testing files.
- Project configuration metadata.
- Dataset setup documentation.
- Recorded evaluation results and test documentation.
- Git commit history for tracking changes.

The `config.json` file identifies the project and its matching methods. It serves as descriptive configuration metadata and does not independently control the application's matching algorithms.

Software release versions are tracked separately from the V1 and V2 matching-method names.

## Additional Evaluation Mode

The project also supports a general split-based evaluation using the included synthetic sample dataset.

Run:

```bash
python -m src.evaluate --data data/sample_pairs.csv --out results
```

This mode provides an educational demonstration of the evaluation workflow.

The final reported Hugging Face evaluation results are produced using `src.final_experiment` with the separately downloaded training and test CSV files.
