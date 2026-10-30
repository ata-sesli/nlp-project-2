# Elif Su Dayı: contribution

**Role:** Model 3 of the benchmark, the Support Vector Machine (LinearSVC baseline), and Model 4 of the benchmark, the CRF (Conditional Random Field). Everything I made is in the `svm_crf/` folder.

The two models share one feature function (`svm_crf/features.py`), so the only algorithmic difference between them is the CRF's label transitions.

| | Model 3: SVM | Model 4: CRF |
|---|---|---|
| Implementation | `LinearSVC` + `FeatureHasher` | `sklearn-crfsuite` (L-BFGS) |
| Hyperparameters (chosen on validation) | C = 0.1 | c1 = 0.25, c2 = 0.1 |
| Token-level test macro F1 | 0.7672 (accuracy 0.9798) | 0.7551 |
| Entity-level test micro F1 (seqeval, strict IOB2) | 0.7631 | 0.8221 |
| Entity-level test macro F1 | 0.7798 | 0.8252 |
| Missed-PII rate | 27.2% | 21.1% |

Baselines at token level (test macro F1): Most Frequent 0.4166, HMM 0.4688.

---

## Part 1: Model 3, the SVM

1. **Wrote the SVM sequence tagger** (`svm_crf/svm_ner.py`) using `scikit-learn`'s `LinearSVC` with a `FeatureHasher` pipeline. Unlike sequence models, the SVM treats NER as independent multi-class token classification, predicting one label per word without transition probabilities.
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

---

## Part 2: Model 4, the CRF

1. **Wrote the word features** (`svm_crf/features.py`). They follow the three information
   sources from the course's POS/NER slides: the word itself, the two words before and after
   it, and the word's form (first and last 3 letters, word shape such as `DE89` -> `Xd`,
   capital letters, digits, `@`). The SVM model can import the same function, so the only
   difference between SVM and CRF is the CRF's label transitions.
2. **Built the CRF** (`svm_crf/crf_ner.py`) with `sklearn-crfsuite` (L-BFGS, all label
   transitions allowed), trained from scratch on the 30,545 training chunks.
3. **Chose the settings on the validation split.** I searched 9 combinations of c1 (L1) and
   c2 (L2) regularization on a 5,000-chunk training subset and picked the best by validation
   entity F1: c1 = 0.25, c2 = 0.1. The test split was used only once, at the end.
4. **Produced the predictions** in the shared format (`svm_crf/predictions/crf_val.jsonl`,
   `crf_test.jsonl`: one `{"id", "pred"}` line per chunk).
5. **Evaluated the model** at two levels and saved everything to
   `svm_crf/results/crf_results.json`:
   - Token level, with the same metrics as the Most Frequent and HMM baselines: test macro F1
     0.7551 (HMM 0.4688, Most Frequent 0.4166).
   - Entity level (seqeval, strict IOB2), where an entity counts only if its boundaries and
     type are both right: test micro F1 0.8221, macro F1 0.8252.
6. **Analysed the results** in `svm_crf/README.md`: F1 per entity type (EMAIL best at 0.94,
   ACCOUNT hardest at 0.71), the missed-PII rate (21.1%), the label transitions the CRF learned
   (for example `O -> I-NAME` gets a strong negative weight, so it learned the BIO rules by
   itself), run times and limitations.

## Where my work is

- `svm_crf/`: `features.py`, `svm_ner.py`, `crf_ner.py`, `README.md`, `predictions/` (`svm_val.jsonl`, `svm_test.jsonl`, `crf_val.jsonl`, `crf_test.jsonl`), `results/` (`svm_results.json`, `crf_results.json`)
- **Trained models (Hugging Face Hub):** [`elifsudayi/svm-ner`](https://huggingface.co/elifsudayi/svm-ner) and [`elifsudayi/crf-ner`](https://huggingface.co/elifsudayi/crf-ner)
- This file.