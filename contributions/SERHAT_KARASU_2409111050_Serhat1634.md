# Individual Contribution Report

**Name:** Serhat Karasu
**Student ID:** 2409111050
**GitHub:** Serhat1634
**Project:** NLP Project 2

## 1. Contribution Summary

My main responsibility in this project was dataset preparation, implementation of statistical baseline models, HMM-based sequence labeling, Viterbi decoding, evaluation, and result analysis.

My contribution focused on building a reproducible data preparation pipeline and implementing and evaluating statistical sequence-labeling approaches.

## 2. Dataset Preparation

I implemented the dataset preparation pipeline using the English subset of the `gretelai/synthetic_pii_finance_multilingual` dataset.

My work included:

- Converting character-level entity annotations into token-level BIO labels.
- Preparing the training, validation, and test splits.
- Splitting training documents before chunking to prevent document-level data leakage.
- Preserving the original test source split.
- Removing duplicate training documents that overlap with the test split.
- Dividing long sequences into chunks with a target length of 200 tokens.
- Preferring sentence and line boundaries when creating chunks.
- Preserving entity continuity when possible, even if a chunk needs to exceed the target length.
- Saving the processed datasets in compressed JSONL format.
- Generating dataset statistics and a preparation report.
- Documenting the dataset source, filtering rules, tokenization, BIO conversion, and data preparation workflow.

The resulting dataset contains 30,545 training chunks, 3,389 validation chunks, and 3,903 test chunks.

The dataset includes eight entity categories and 17 BIO labels.

**Main file:** `dataset/prepare_dataset.py`

**Documentation:**
- `dataset/README.md`
- `dataset/data/REPORT.md`

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
- Word-shape features for recognizing patterns such as email addresses, phone numbers, numeric strings, URLs, and capitalization patterns.
- An unknown-token mechanism that maps rare or unseen words to word-shape features.
- BIO transition constraints to prevent invalid tag transitions.

I also implemented Viterbi decoding to determine the most probable BIO tag sequence for each input sequence.

The Viterbi algorithm uses dynamic programming to find the highest-probability sequence instead of evaluating every possible tag sequence independently.

**Main file:** `baselines/hmm_viterbi.py`

## 5. Evaluation and Comparison

I contributed to the evaluation and comparison of the statistical baseline models using the project's shared benchmark pipeline.

The evaluation includes:

- Token-level accuracy.
- Entity-level precision, recall, and F1.
- Macro F1 across entity categories.
- Per-entity-type exact-span evaluation.
- Missed PII rate.
- Error analysis, including boundary, entity-type, missed-entity, and invented-entity errors.

For entity-level evaluation, a prediction is considered correct only when its entity type, start position, and end position exactly match the gold annotation.

This strict evaluation approach distinguishes correct entity recognition from predictions that only partially overlap with the annotated spans.

## 6. Main Results

On the shared test benchmark, the HMM with Viterbi decoding achieved:

- Token-level accuracy: 0.960
- Entity-level precision: 0.630
- Entity-level recall: 0.564
- Entity-level F1: 0.595
- Entity-type macro F1: 0.528
- Missed PII rate: 0.436

The Most Frequent baseline achieved an entity-level F1 score of 0.327.

The HMM therefore achieved a higher exact-span entity F1 score than the Most Frequent baseline.

The per-entity results also showed the following F1 scores:

- EMAIL: 0.699
- PHONE: 0.668
- NAME: 0.713
- DATE: 0.628
- ADDRESS: 0.553
- COMPANY: 0.487
- ACCOUNT: 0.236
- ID: 0.236

These results show that the HMM performed better on some entity categories than others, with ACCOUNT and ID recognition remaining particularly challenging.

## 7. Files Included in My Contribution

- `dataset/prepare_dataset.py`
- `dataset/inspect_dataset.py`
- `dataset/README.md`
- `dataset/data/REPORT.md`
- `dataset/data/labels.json`
- `dataset/data/train.jsonl.gz`
- `dataset/data/val.jsonl.gz`
- `dataset/data/test.jsonl.gz`
- `baselines/most_frequent.py`
- `baselines/hmm_viterbi.py`
- `benchmark/results/` (first full benchmark run)
- `requirements.txt`
- This contribution document.

## 8. Summary

My contribution covers the workflow from dataset preparation to statistical sequence-labeling experiments and result analysis.

The implemented work provides reproducible dataset preparation, document-level data splitting, BIO sequence labeling, a Most Frequent baseline, HMM-based sequence modeling, Viterbi decoding, and evaluation through the project's shared benchmark.

This work establishes a statistical baseline and provides a basis for comparing traditional sequence-labeling approaches with more advanced machine learning and deep learning models.