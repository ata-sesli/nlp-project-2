
# Individual Contribution Report

**Name:** Serhat Karasu
**Student ID:** 2409111050
**GitHub:** Serhat1634
**Project:** NLP Project 2

## 1. Contribution Summary

My main responsibility in this project was dataset preparation, implementation of statistical baseline models, HMM-based sequence labeling, Viterbi decoding, evaluation, and result analysis.

My contribution focused on building a reproducible data preparation and evaluation pipeline and comparing a simple frequency-based baseline with a Hidden Markov Model.

## 2. Dataset Preparation

I implemented the dataset preparation pipeline using the English subset of the `gretelai/synthetic_pii_finance_multilingual` dataset.

My work included:

- Converting character-level entity annotations into token-level BIO labels.
- Preparing the training, validation, and test splits.
- Splitting training documents before chunking to prevent document-level data leakage.
- Preserving the original test source split.
- Removing duplicate training documents that overlap with the test split.
- Dividing long sequences into chunks with a maximum length of 200 tokens.
- Saving the processed datasets in compressed JSONL format.
- Generating dataset statistics and a preparation report.

The resulting dataset contains 30,545 training chunks, 3,389 validation chunks, and 3,903 test chunks.

The dataset includes eight entity categories and 17 BIO labels.

**Main file:** `dataset/prepare_dataset.py`

**Documentation:** `dataset/data/REPORT.md`

## 3. Most Frequent Baseline

I implemented a Most Frequent baseline model.

The model learns the most frequently observed BIO tag for each lowercased token during training. For tokens that are not observed during training, it uses the globally most frequent tag.

This model provides a simple statistical reference point for evaluating the HMM-based approach.

**Main file:** `baselines/most_frequent.py`

## 4. HMM and Viterbi Implementation

I implemented a first-order Hidden Markov Model for sequence labeling.

The model includes:

- Initial-state probabilities.
- Tag transition probabilities.
- Token emission probabilities.
- Laplace smoothing with alpha = 0.1.
- Lowercased token handling.
- An unknown-token mechanism.
- BIO transition constraints.

I also implemented Viterbi decoding to determine the most probable BIO tag sequence for each input sequence.

The Viterbi algorithm uses dynamic programming to find the highest-probability sequence instead of evaluating every possible tag sequence independently.

**Main file:** `baselines/hmm_viterbi.py`

## 5. Evaluation and Comparison

I implemented evaluation and comparison workflows for both models.

The evaluation includes:

- Token-level accuracy.
- Macro precision, recall, and F1.
- Weighted F1.
- Entity-level precision, recall, and F1.
- Per-entity-type exact-span evaluation.

For entity-level evaluation, a prediction is considered correct only when its entity type, start position, and end position match the gold annotation.

The evaluator checks record identifiers and sequence lengths before comparing predictions. Invalid `I-TYPE` tags that do not continue an entity of the same type are interpreted as the beginning of a new entity of that type.

I also generated prediction files and machine-readable result files for validation and test splits.

**Main files:**

- `baselines/entity_evaluation.py`
- `baselines/compare_models.py`

## 6. Main Results

On the test set, the HMM with Viterbi decoding achieved:

- Token-level accuracy: 0.9532
- Token-level macro F1: 0.4688
- Entity-level precision: 0.5469
- Entity-level recall: 0.4971
- Entity-level F1: 0.5208

The Most Frequent baseline achieved an entity-level F1 score of 0.2653.

These results show that the HMM model performs substantially better than the Most Frequent baseline in exact-span entity recognition.

The results and detailed interpretation are available in:

`baselines/ANALYSIS.md`

## 7. Files Included in My Contribution

- `dataset/prepare_dataset.py`
- `dataset/data/REPORT.md`
- `dataset/data/labels.json`
- `dataset/data/train.jsonl.gz`
- `dataset/data/val.jsonl.gz`
- `dataset/data/test.jsonl.gz`
- `baselines/most_frequent.py`
- `baselines/hmm_viterbi.py`
- `baselines/entity_evaluation.py`
- `baselines/compare_models.py`
- `baselines/predictions/`
- `baselines/results/`
- `baselines/ANALYSIS.md`
- This contribution document.

## 8. Summary

My contribution covers the complete workflow from dataset preparation to statistical model evaluation and result analysis.

The implemented pipeline provides reproducible dataset splits, two baseline approaches, HMM-based sequence decoding, token-level and entity-level evaluation, and a detailed comparison of the experimental results.
