# Elif Su Dayı: contribution

**Role:** Model 3 of the benchmark, the Support Vector Machine (LinearSVC baseline). Everything I made is in the `svm_crf/` folder.

## What I did

1. **Wrote the SVM sequence tagger** (`svm_crf/svm_ner.py`) using `scikit-learn`'s `LinearSVC` with a `DictVectorizer` pipeline. Unlike sequence models, the SVM treats NER as independent multi-class token classification, predicting one label per word without transition probabilities.
2. **Reused the shared feature function** (`svm_crf/features.py`), importing the exact same window-based lexical and morphological representations used by the CRF model (word surface, ±2 sliding context window, 3-character prefixes/suffixes, shape abstraction like `DE89` -> `Xd`, capitalization, digit flags, and `@` symbols). This guaranteed that the only algorithmic difference between SVM and CRF was label transition modeling.
3. **Selected hyperparameters on the validation split.** Tuned the regularization strength $C \in \{0.1, 1.0, 10.0\}$ on a 5,000-chunk training subset. $C = 0.1$ achieved the highest validation entity F1 score (0.6927 vs. 0.6667 for $C=1.0$ and 0.6164 for $C=10.0$). The final model was trained on the full 30,545 training chunks using $C=0.1$, and the test set was evaluated strictly once at the end.
4. **Produced the predictions** in the project's shared JSONL format (`svm_crf/predictions/svm_val.jsonl` and `svm_test.jsonl`), outputting one `{"id", "pred"}` object per chunk aligned with input tokens.
5. **Evaluated the model** across two standard levels, saving all metrics to `svm_crf/results/svm_results.json`:
   - **Token level:** Test macro F1 of 0.7672 (token accuracy 0.9798), outperforming the naive Most Frequent baseline (0.4166) and HMM baseline (0.4688) due to rich window and morphological features.
   - **Entity level (`seqeval`, strict IOB2):** Test micro F1 of 0.7631 and macro F1 of 0.7798, measuring exact span boundary and entity type correctness.
6. **Conducted comparative error analysis** against the CRF model in `svm_crf/README.md`:
   - Documented the impact of independent decision-making: SVM produces illegal BIO transitions (e.g., `O -> I-NAME`) because it lacks state transition dependencies.
   - Analyzed per-entity metrics: strong performance on regex-like patterns (EMAIL at 0.9417 precision, PHONE at 0.8754 precision) but lower recall on ambiguous categories like COMPANY (0.5682 recall, 43.2% missed) and ACCOUNT (0.6350 recall, 36.5% missed).
   - Highlighted the missed-PII rate: SVM left 27.2% of sensitive entities unmasked (vs. CRF's 21.1%), demonstrating why sequence decoding (Viterbi/CRF) is crucial for privacy compliance.

## Where my work is

- `svm_crf/`: `svm_ner.py`, `README.md`, `predictions/svm_val.jsonl`, `predictions/svm_test.jsonl`, `results/svm_results.json`
- This file.