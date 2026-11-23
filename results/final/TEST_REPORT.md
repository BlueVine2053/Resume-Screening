# Intelligent Resume Screening and Job Matching System — Test Report

## 1. Test Overview

This report documents the testing and evaluation of the Intelligent Resume Screening and Job Matching System.

The application implements two resume-to-job matching approaches:

- **V1 — Keyword Overlap Matching:** Calculates matching scores based on keyword overlap between resume text and job descriptions.
- **V2 — TF-IDF Cosine Similarity:** Uses TF-IDF vectorization and cosine similarity to measure textual similarity.

Testing included automated Python tests, model evaluation, dataset overlap auditing, reproducibility verification, and manual Streamlit application testing.

## 2. Test Data and Execution

The evaluation used separate training and test datasets obtained from Hugging Face.

**Dataset source:** [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

| Dataset | Number of records |
|---|---:|
| Original training dataset | 6,241 |
| Cleaned training dataset | 6,228 |
| Validation subset | 1,246 |
| Separate test dataset | 1,759 |

The training dataset was cleaned to remove duplicate and conflicting resume–job pairs.

A stratified 80/20 split within the cleaned training dataset was used for threshold selection.

The final matcher was refitted using the complete cleaned training dataset.

The separate test dataset was not used to fit the keyword vocabulary, TF-IDF vectorizer, or classification thresholds.

## 3. Automated Testing

The project includes automated Python tests covering the core application logic.

**Recorded result: 18 automated tests passed.**

Testing covered:

- Dataset loading and column mapping.
- Input validation and data processing.
- Resume–job pair deduplication.
- Keyword-based matching functionality.
- TF-IDF cosine-similarity matching functionality.
- Dataset overlap auditing.
- Saved-model scoring consistency.
- Evaluation functionality.

Python syntax compilation also passed.

## 4. Evaluation Results

Both matching methods were evaluated against the same 1,759 test pairs.

| Metric | V1 Keyword | V2 TF-IDF |
|---|---:|---:|
| Accuracy | 39.80% | 44.57% |
| Macro Precision | 33.32% | 37.52% |
| Macro Recall | 33.32% | 35.16% |
| Macro F1 | 32.81% | 32.10% |

### Statistical Comparison

The paired exact McNemar test produced the following results:

| Measurement | Result |
|---|---:|
| V1-only correct predictions | 210 |
| V2-only correct predictions | 294 |
| Exact McNemar p-value | 0.0002114451 |

V2 achieved higher overall accuracy than V1.

However, V1 achieved a slightly higher macro F1 score.

These results indicate that TF-IDF cosine similarity performed better on overall accuracy for the evaluated dataset, while keyword overlap achieved a slightly higher macro F1 score.

The matching scores are measures of textual similarity and should not be interpreted as hiring decisions or definitive assessments of candidate qualifications.

## 5. Dataset Overlap Audit

An overlap audit was performed to examine whether training and test records shared identical resume text, job-description text, or complete resume–job pairs.

| Overlap category | Result |
|---|---:|
| Exact resume–job pair overlap | 0 test rows |
| Exact normalized job-description overlap | 0 test rows |
| Exact normalized resume overlap | 1,758 of 1,759 test rows |
| Unique normalized resumes overlapping | 476 |

### Evaluation Limitation

The results represent performance on held-out resume–job pairs rather than entirely unseen candidates.

Because 1,758 test rows contain normalized resume text also present in training, the evaluation cannot establish how well the system generalizes to completely new candidates.

This limitation is considered when interpreting the reported accuracy and other evaluation metrics.

## 6. Reproducibility Verification

The full evaluation was repeated using the same input datasets and random seed.

The repeated execution produced:

- Byte-identical evaluation metrics JSON.
- Byte-identical prediction CSV.
- Consistent evaluation results across both matching methods.

These results demonstrate reproducibility under the same recorded inputs and execution settings.

## 7. Manual Streamlit Application Testing

Manual testing was performed on the Streamlit application to verify basic user interactions and input handling.

The testing included:

1. Opening the Streamlit application.
2. Entering resume text.
3. Entering job-description text.
4. Viewing the matching results.
5. Checking application behavior when resume or job-description input was blank or contained only whitespace.

### Observed Results

- The application accepted resume and job-description text.
- Matching results were displayed through the interface.
- Invalid blank or whitespace-only input produced an error rather than a matching score.

These manual tests complement the automated Python tests by checking the application's user-facing functionality.

Screenshots and a separate browser-test log are not included in this report.

## 8. Test Execution Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Automated Tests

```bash
python -m pytest -q
```

### Run Final Evaluation

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

The original dataset files must be downloaded separately and saved as `data/train.csv` and `data/test.csv` before rerunning the final evaluation.

## 9. Test Summary

The recorded testing and evaluation established the following:

- **18 automated Python tests passed.**
- Both V1 keyword overlap and V2 TF-IDF cosine similarity were evaluated.
- V2 achieved higher overall accuracy on the evaluated test pairs.
- V1 achieved a slightly higher macro F1 score.
- Repeated evaluation produced identical recorded outputs using the same inputs and settings.
- Dataset overlap auditing identified substantial resume reuse between training and testing.
- Manual Streamlit testing covered text entry, matching results, and invalid blank-input handling.

The testing provides evidence that the implemented matching methods and evaluation workflow functioned under the recorded conditions.

The evaluation results are specific to the tested dataset and do not establish candidate-independent generalization or actual hiring suitability.
