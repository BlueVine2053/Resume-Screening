# Intelligent Resume Screening and Job Matching System — Test Report

## 1. Test Overview

This report documents the testing and evaluation of the Intelligent Resume Screening and Job Matching System, including the additional verification performed for CR-002.

The application implements two resume-to-job matching approaches:

- **V1 — Keyword Overlap Matching:** Calculates matching scores based on keyword overlap between resume text and job descriptions.
- **V2 — TF-IDF Cosine Similarity:** Uses TF-IDF vectorization and cosine similarity to measure textual similarity.

Testing included automated Python tests, model evaluation, dataset overlap auditing, reproducibility verification, manual Streamlit application testing, automated interface testing, input boundary testing, and backward model compatibility verification.

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

### CR-002 Dataset Compatibility Verification

Before implementing the updated input-validation requirements, the cleaned datasets were examined for maximum text lengths and inputs without recognizable tokens.

| Dataset | Maximum resume length | Maximum job-description length | Tokenless resumes | Tokenless job descriptions |
|---|---:|---:|---:|---:|
| Training | 25,364 | 7,669 | 0 | 0 |
| Validation | 25,364 | 7,669 | 0 | 0 |
| Testing | 25,364 | 8,171 | 0 | 0 |

A maximum input length of 100,000 characters per resume or job description was established.

The observed dataset lengths were below this limit, and no tokenless inputs were identified in the cleaned datasets.

## 3. Automated Testing

The project includes automated Python tests covering the core application logic and Streamlit interface.

**Recorded CR-002 result: 33 automated tests passed in 7.27 seconds.**

### Test Coverage by Version

| Version | Automated tests passed |
|---|---:|
| v0.1.0 | 18 |
| v0.1.1 | 19 |
| CR-002 — target v0.1.2 | 33 |

Existing testing covered:

- Dataset loading and column mapping.
- Input validation and data processing.
- Resume–job pair deduplication.
- Keyword-based matching functionality.
- TF-IDF cosine-similarity matching functionality.
- Dataset overlap auditing.
- Saved-model scoring consistency.
- Evaluation functionality.
- Batch input-length validation for mismatched resume and job-description lists.

Python syntax compilation also passed during previous project verification.

### CR-002 Additional Automated Testing

CR-002 introduced 11 additional matcher validation test cases and 3 automated Streamlit interface tests.

| Test category | Test cases | Result |
|---|---:|---|
| Existing regression tests | 19 | Passed |
| Invalid input types and whitespace-only text | 4 | Passed |
| Tokenless input validation | 4 | Passed |
| Maximum input-length boundary | 1 | Passed |
| Oversized input rejection | 1 | Passed |
| Unknown-word scoring behavior | 1 | Passed |
| Automated Streamlit interface tests | 3 | Passed |
| **Total** | **33** | **Passed** |

The additional matcher tests verify that invalid inputs are rejected, valid inputs at the maximum character limit are accepted, oversized inputs are rejected without exposing the submitted text in the error message, and existing unknown-word scoring behavior is preserved.

### Regression Test Failure and Resolution

The initial CR-002 regression test execution produced 32 passed tests and 1 failed test.

The existing training-only vocabulary test used `secret_test_token` as an unseen input. The updated validation rejected this input because the existing keyword tokenizer did not recognize it as a valid token.

The test input was changed to `secrettesttoken`, preserving the original objective of verifying that scoring unseen text does not modify the trained vocabulary.

No changes to the matching algorithms or tokenizer were required to resolve the failure.

The complete test suite was executed again, resulting in 33 passed tests.

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

CR-002 verification confirmed that the recorded evaluation results remained unchanged from the v0.1.1 baseline.

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

CR-002 did not modify the dataset splitting or overlap-auditing procedures. The overlap results remained unchanged.

## 6. Reproducibility Verification

During previous project verification, the full evaluation was repeated using the same input datasets and random seed.

The repeated execution produced:

- Byte-identical evaluation metrics JSON.
- Byte-identical prediction CSV.
- Consistent evaluation results across both matching methods.

These results demonstrate reproducibility under the same recorded inputs and execution settings.

### CR-002 Baseline Comparison

The final experiment was executed separately using the v0.1.1 baseline and the updated CR-002 source code.

Both executions used the same training and test datasets and a random seed of 42.

The generated output files were stored in separate directories to preserve the original baseline results.

| Verification item | v0.1.1 baseline | CR-002 | Result |
|---|---:|---:|---|
| Training rows | 6,228 | 6,228 | Identical |
| Validation rows | 1,246 | 1,246 | Identical |
| Test rows | 1,759 | 1,759 | Identical |
| V1 accuracy | 39.80% | 39.80% | Identical |
| V2 accuracy | 44.57% | 44.57% | Identical |
| V1 macro F1 | 32.81% | 32.81% | Identical |
| V2 macro F1 | 32.10% | 32.10% | Identical |
| McNemar p-value | 0.0002114451 | 0.0002114451 | Identical |
| V1 unscoreable pairs | 0 | 0 | Identical |

A programmatic comparison was performed using Python JSON parsing and pandas DataFrame equality assertions.

The comparison produced the following results:

```text
PASS: Complete parsed metrics.json is identical
PASS: All prediction rows, scores and labels are identical
Compared prediction rows: 1759
```

The complete parsed metrics JSON comparison included dataset cleaning counts, overlap audit results, selected thresholds, classification metrics, confusion matrices, statistical comparison results, and recorded limitations.

The prediction CSV comparison included all 1,759 rows, pair identifiers, actual labels, matching scores, and predicted classifications.

These checks confirmed that CR-002 did not change the experimental results or individual predictions for the evaluated dataset.

### Backward Model Compatibility

The saved v0.1.1 matcher and the newly trained CR-002 matcher were loaded under the updated source code.

The following valid inputs were used:

- Resume: `python sql data analyst`
- Job description: `python sql data analysis`

| Matching method | v0.1.1 saved matcher | CR-002 matcher |
|---|---:|---:|
| V1 keyword overlap | 0.6666666666666666 | 0.6666666666666666 |
| V2 TF-IDF cosine similarity | 0.8531466398851143 | 0.8531466398851143 |

The compatibility test produced the following results:

```text
PASS: v0.1.1 model loads and scores under CR-002 code
PASS: Baseline and updated model scores are identical
```

This confirms that the previously saved matcher remains usable under the updated code and produces identical scores to the newly trained matcher for the selected test input.

## 7. Streamlit Application Testing

### Previous Manual Testing

Manual testing was performed on the Streamlit application during previous project verification to verify basic user interactions and input handling.

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

### CR-002 Automated Interface Testing

Three automated interface tests were added using Streamlit AppTest.

| Test | Expected behavior | Result |
|---|---|---|
| Persistent disclaimer | Disclaimer is visible before matching scores are calculated | Passed |
| Matching-method labels | V1 keyword matching and V2 TF-IDF cosine similarity are identified | Passed |
| Keyword-matching limitation | Exact-word matching, synonym limitations, and skill interpretation limitations are disclosed | Passed |

The automated interface tests confirm that the required notices are displayed when the application initially loads.

These tests verify the presence and visibility of the notices. They do not establish whether users understand or correctly interpret the matching scores.

## 8. Test Execution Commands

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run Automated Tests

```bash
python -m pytest -q
```

### Run Final Evaluation

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out results/final --seed 42
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

The original dataset files must be downloaded separately and saved as `data/train.csv` and `data/test.csv` before rerunning the final evaluation.

### CR-002 Verification Commands

The baseline experiment was executed from the extracted v0.1.1 source directory:

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out ../baseline-v0.1.1 --seed 42
```

The updated experiment was executed from the separate CR-002 source directory:

```bash
python -m src.final_experiment --train data/train.csv --test data/test.csv --out ../after-CR-002 --seed 42
```

The separate output directories preserved both sets of generated files for comparison.

## 9. Test Summary

The recorded testing and evaluation established the following:

- The v0.1.1 baseline passed 19 automated Python tests.
- CR-002 passed 33 automated Python tests in 7.27 seconds.
- Updated input validation rejects invalid input types, blank text, tokenless text, and oversized inputs.
- Valid inputs at the maximum character limit are accepted.
- Existing batch input-length validation and unknown-word scoring behavior are preserved.
- Automated Streamlit tests confirmed the presence of the persistent disclaimer, matching-method labels, and keyword-matching limitation notice.
- Both V1 keyword overlap and V2 TF-IDF cosine similarity were evaluated.
- V2 achieved higher overall accuracy on the evaluated test pairs.
- V1 achieved a slightly higher macro F1 score.
- Programmatic comparison confirmed identical parsed metrics JSON and all 1,759 prediction rows between the v0.1.1 baseline and CR-002.
- Backward model compatibility was verified for the selected valid input.
- Dataset overlap auditing identified substantial resume reuse between training and testing.
- Previous manual Streamlit testing covered text entry, matching results, and invalid blank-input handling.

The testing provides evidence that the implemented matching methods and evaluation workflow functioned under the recorded conditions.

CR-002 strengthened input validation, expanded automated test coverage, and improved interface transparency without changing the recorded experimental results.

The evaluation results are specific to the tested dataset and do not establish candidate-independent generalization or actual hiring suitability. The existing limitations of keyword matching, classification performance, dataset overlap, and score interpretation remain applicable.
