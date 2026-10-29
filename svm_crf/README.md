# Support Vector Machine (LinearSVC) & CRF

This directory contains Model 3 (SVM) and Model 4 (CRF) of the benchmark. Both models use the
same hand-written word features (`features.py`), so the difference between them shows the value
of modeling label transitions.

| | SVM (Model 3) | CRF (Model 4) |
|---|---|---|
| Architecture | Independent token classifier (linear SVM) | Linear-chain CRF |
| Specific model | `sklearn.svm.LinearSVC`, `dual="auto"`, `max_iter=2000`, C = 0.1 | `sklearn_crfsuite.CRF(algorithm="lbfgs")`, c1 = 0.25, c2 = 0.1, 200 iterations, all possible transitions |
| Decides | each token on its own | the whole tag sequence at once |
| Training | from scratch on `dataset/data/train.jsonl.gz` (30,545 chunks) | same |
| GPU | not needed | not needed |

## Files

| File | What it does |
|---|---|
| `features.py` | Feature function shared by the SVM and CRF models |
| `svm_ner.py` | Vectorizes features with `FeatureHasher`, picks C on validation, trains the final SVM, writes predictions and results |
| `crf_ner.py` | Picks c1/c2 on validation, trains the final CRF, writes predictions and results |
| `predictions/svm_val.jsonl`, `predictions/svm_test.jsonl` | SVM predictions, one line per chunk: `{"id": ..., "pred": [...]}` |
| `predictions/crf_val.jsonl`, `predictions/crf_test.jsonl` | CRF predictions, same format |
| `results/svm_results.json` | SVM settings, tuning grid, timing, token-level and entity-level reports |
| `results/crf_results.json` | CRF settings, tuning grid, timing, learned transitions, token-level and entity-level reports |
| `models/` | Saved models (`svm.joblib` about 143 MB, `crf.joblib` 11 MB); not committed, rebuilt by the scripts. Checkpoints hosted on Hugging Face: [SVM Checkpoint](https://huggingface.co/elifsudayi/svm-ner) and [CRF Checkpoint](https://huggingface.co/elifsudayi/crf-ner) |

## How to run

From the repository root:

```bash
pip install scikit-learn sklearn-crfsuite seqeval joblib
python svm_crf/svm_ner.py
python svm_crf/crf_ner.py
```

If `seqeval` fails to build, use `pip install --use-pep517 seqeval`.

Options:

- SVM: `--c 0.1` skips the search and trains directly; `--tune-subset 5000` sets the
  training chunks used for the search.
- CRF: `--c1 0.25 --c2 0.1` skips the search; `--tune-subset 5000`, `--max-iterations 200`,
  `--workers 3` (parallel processes for the search).

Run times:

| Step | SVM | CRF |
|---|---|---|
| Hyperparameter search (5,000 chunks) | about 11 min | about 12 min |
| Final training (all chunks) | 26 min (1,603 s) | 43 min |
| Prediction of val + test | 16 s | 26 s |
| Peak RAM | about 5 GB | about 4 GB |

The searches and the CRF were run on a 4-core cloud CPU; the final SVM time is from our own
run of `svm_ner.py`.

## Features

The features follow the three information sources from the course slides (POS/NER intro,
slide 12). Each token gets these features for itself and for the two words before and after it
(window of ±2):

- lowercased word
- first 3 and last 3 letters
- word shape (`Anna` -> `Xx`, `DE89` -> `Xd`, `a@b.com` -> `x@x.x`)
- is title case, is upper case, is a number, contains a digit, contains `@`
- `BOS` / `EOS` markers at the chunk edges

**The key difference:** the SVM treats each token independently and does not learn
label-to-label transitions (like `B-NAME` -> `I-NAME`). The CRF learns a weight for every
transition and decides the whole sequence at once.

To keep memory low, the SVM uses a `FeatureHasher` (2^20 features) and builds the features
chunk by chunk. An earlier version that kept every feature dictionary in memory with
`DictVectorizer` needed more than 22 GB of RAM; the hashed version peaks at about 5 GB.

## Hyperparameter search

All settings were chosen on the validation split. To keep the search fast, each setting was
trained on a 5,000-chunk random training subset (seed 42). The best setting was then trained
on all training chunks. The test split was used only once, after the final models were chosen.

SVM (validation entity micro F1):

| C | Validation entity micro F1 |
|---:|---:|
| 0.1 | **0.6927** |
| 1.0 | 0.6667 |
| 10.0 | 0.6164 |

CRF (validation entity micro F1, 100 iterations):

| c1 \ c2 | 0.01 | 0.05 | 0.1 |
|---|---:|---:|---:|
| 0.05 | 0.7471 | 0.7593 | 0.7615 |
| 0.1 | 0.7575 | 0.7625 | 0.7625 |
| 0.25 | 0.7627 | 0.7674 | **0.7674** |

## Results

### Token level (same metrics as the other baselines)

| Split | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | SVM | 0.9789 | 0.8330 | 0.7420 | 0.7766 | 0.9784 |
| Validation | CRF | 0.9806 | 0.7874 | 0.7989 | 0.7806 | 0.9803 |
| Test | Most Frequent | 0.9493 | 0.6345 | 0.3438 | 0.4166 | 0.9419 |
| Test | HMM + Viterbi | 0.9532 | 0.5260 | 0.5096 | 0.4688 | 0.9550 |
| Test | SVM | 0.9799 | 0.8371 | 0.7302 | 0.7672 | 0.9793 |
| Test | CRF | 0.9813 | 0.8047 | 0.7683 | 0.7551 | 0.9811 |

Most Frequent and HMM numbers are from `baselines/ANALYSIS.md`. Token accuracy is misleading
(about 0.98 for both models) because the vast majority of tokens are `O`.

### Entity level (seqeval, strict IOB2)

An entity counts as correct only if its boundaries and its type are both correct.

| Split | Model | Micro Precision | Micro Recall | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|
| Validation | SVM | 0.7916 | 0.7227 | 0.7556 | 0.7779 |
| Validation | CRF | 0.8531 | 0.7847 | 0.8175 | 0.8272 |
| Test | SVM | 0.8019 | 0.7279 | 0.7631 | 0.7798 |
| Test | CRF | 0.8586 | 0.7887 | 0.8221 | 0.8252 |

Test F1 per entity type:

| Type | SVM Precision | SVM Recall | SVM F1 | CRF Precision | CRF Recall | CRF F1 | Entities |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMAIL | 0.9417 | 0.9378 | 0.9397 | 0.9532 | 0.9295 | **0.9412** | 241 |
| PHONE | 0.8754 | 0.8338 | 0.8541 | 0.9167 | 0.8487 | **0.8814** | 337 |
| ADDRESS | 0.6914 | 0.6735 | 0.6823 | 0.8843 | 0.8443 | **0.8638** | 1,856 |
| DATE | 0.8120 | 0.8134 | 0.8127 | 0.8485 | 0.8444 | **0.8464** | 5,712 |
| NAME | 0.8834 | 0.7450 | 0.8083 | 0.8989 | 0.7734 | **0.8314** | 3,447 |
| ID | 0.8974 | 0.6959 | 0.7839 | 0.8667 | 0.7163 | **0.7844** | 490 |
| COMPANY | 0.7144 | 0.5682 | 0.6330 | 0.7994 | 0.6879 | **0.7395** | 2,932 |
| ACCOUNT | 0.8434 | 0.6350 | **0.7245** | 0.8522 | 0.6141 | 0.7138 | 526 |

Missed PII rate (1 - recall) on test: SVM 27.2%, CRF 21.1% of entities.

## Observations (SVM vs CRF)

- **Token level hides boundary errors.** At token level the SVM is slightly ahead (macro F1
  0.7672 vs 0.7551), but at entity level the CRF is clearly ahead (micro F1 0.8221 vs 0.7631).
  A token can get the right type while the entity's boundaries are wrong, and only the entity
  level counts that as an error.
- **The value of transitions.** Both models use the same features, so the 5.9-point gap in test
  entity micro F1 mostly reflects the value of modeling label transitions. (The models also
  differ in training method and the SVM uses feature hashing, so the gap is not caused by
  transitions alone.)
- **Boundary errors.** The SVM produces broken BIO sequences: on test, about 7% of its `I-` tags
  (1,866 of 25,489) follow a tag that cannot come before them, for example `O` followed by
  `I-NAME`. The CRF does this only 4 times. It learned strong negative weights for these
  transitions without being told the BIO rules: `O -> I-COMPANY` (-12.3), `O -> I-DATE`
  (-12.0), `O -> I-ADDRESS` (-11.6), `O -> I-NAME` (-10.5). The largest gains are on
  multi-word entities: ADDRESS 0.68 -> 0.86 and COMPANY 0.63 -> 0.74.
- **Type ambiguity.** Both models struggle with COMPANY and ACCOUNT. Company names look like
  normal title-case words, and account numbers look like other numbers in the documents; the
  ±2 window is often not enough context to tell them apart. ACCOUNT is the only type where the
  SVM is slightly better than the CRF.
- **Precision over recall.** Both models are usually right when they mark something as PII, but
  miss many entities: the SVM misses 43.2% of companies and 36.5% of account numbers. Leaving 27.2%
  (SVM) or 21.1% (CRF) of PII unmasked would be too much for a real redaction system.

## Limitations

- Features are hand-written; neither model can use word meaning beyond the exact training
  words and the predefined shapes.
- SVM: hashing makes the learned weights hard to inspect, because several features can share
  one hashed column. The saved model is large (about 143 MB: 17 labels x 2^20 weights).
- CRF: c1/c2 were searched on a 5,000-chunk subset and the best c1 (0.25) is at the edge of the
  grid, so a larger c1 might be slightly better. The final model stopped at 200 iterations;
  the loss fell by only 1.2% over the last 50.
- Very rare inside tags (`I-ID`: 9 test tokens, 0 validation tokens; `I-ACCOUNT`: 38 test
  tokens) pull down token-level macro F1, so the entity-level scores are the better summary.
