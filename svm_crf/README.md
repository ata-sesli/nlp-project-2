# CRF (Conditional Random Field)

Model 4 of the benchmark: a linear-chain CRF trained with
`sklearn-crfsuite` (L-BFGS) on hand-written word features.

- Architecture: linear-chain CRF
- Specific model: `sklearn_crfsuite.CRF(algorithm="lbfgs")`, c1 = 0.25, c2 = 0.1,
  200 iterations, all possible label transitions
- Training: from scratch on `dataset/data/train.jsonl.gz` (30,545 chunks)
- No GPU needed

## Files

| File | What it does |
|---|---|
| `features.py` | Feature function shared with the SVM model (same features, so SVM vs CRF measures only the value of label transitions) |
| `crf_ner.py` | Picks c1/c2 on validation, trains the final CRF, writes predictions and results |
| `predictions/crf_val.jsonl`, `predictions/crf_test.jsonl` | One line per chunk: `{"id": ..., "pred": [...]}` |
| `results/crf_results.json` | Settings, tuning grid, timing, learned transitions, token-level and entity-level reports |
| `models/` | Saved model (`crf.joblib`, 11 MB); not committed, rebuilt by the script |

## How to run

From the repository root:

```bash
pip install sklearn-crfsuite seqeval joblib
python svm_crf/crf_ner.py
```

If `seqeval` fails to build, use `pip install --use-pep517 seqeval`.

Options:

- `--c1 0.25 --c2 0.1`: skip the search and train directly
- `--tune-subset 5000`: training chunks used for the search
- `--max-iterations 200`: L-BFGS iterations for the final model
- `--workers 3`: parallel processes for the search

Run time on a 4-core CPU: search 12 min, final training 43 min,
prediction of val + test 26 s.

## Features

The features follow the three information sources from the course slides
(POS/NER intro, slide 12). Each token gets these features for itself and
for the two words before and after it:

- lowercased word
- first 3 and last 3 letters
- word shape (`Anna` -> `Xx`, `DE89` -> `Xd`, `a@b.com` -> `x@x.x`)
- is title case, is upper case, is a number, contains a digit, contains `@`
- `BOS` / `EOS` markers at the chunk edges

The CRF also learns a weight for every label-to-label transition.
Unlike the SVM, it decides the whole tag sequence at once.

## Hyperparameter search

c1 (L1) and c2 (L2) were chosen on the validation split. To keep the search
fast, each setting was trained on 5,000 random training chunks (seed 42,
100 iterations). The best setting was then trained on all training chunks.

| c1 \ c2 | 0.01 | 0.05 | 0.1 |
|---|---:|---:|---:|
| 0.05 | 0.7471 | 0.7593 | 0.7615 |
| 0.1 | 0.7575 | 0.7625 | 0.7625 |
| 0.25 | 0.7627 | 0.7674 | **0.7674** |

Values are validation entity-level micro F1. The test split was used only
once, after the final model was chosen.

## Results

### Token level (same metrics as the other baselines)

| Split | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | CRF | 0.9806 | 0.7874 | 0.7989 | 0.7806 | 0.9803 |
| Test | Most Frequent | 0.9493 | 0.6345 | 0.3438 | 0.4166 | 0.9419 |
| Test | HMM + Viterbi | 0.9532 | 0.5260 | 0.5096 | 0.4688 | 0.9550 |
| Test | CRF | 0.9813 | 0.8047 | 0.7683 | 0.7551 | 0.9811 |

Most Frequent and HMM numbers are from `baselines/ANALYSIS.md`.
On test, CRF raises token macro F1 from 0.4688 (HMM) to 0.7551.

### Entity level (seqeval, strict IOB2)

An entity counts as correct only if its boundaries and its type are both
correct.

| Split | Micro Precision | Micro Recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Validation | 0.8531 | 0.7847 | 0.8175 | 0.8272 |
| Test | 0.8586 | 0.7887 | 0.8221 | 0.8252 |

Test F1 per entity type:

| Type | Precision | Recall | F1 | Entities |
|---|---:|---:|---:|---:|
| EMAIL | 0.9532 | 0.9295 | 0.9412 | 241 |
| PHONE | 0.9167 | 0.8487 | 0.8814 | 337 |
| ADDRESS | 0.8843 | 0.8443 | 0.8638 | 1,856 |
| DATE | 0.8485 | 0.8444 | 0.8464 | 5,712 |
| NAME | 0.8989 | 0.7734 | 0.8314 | 3,447 |
| ID | 0.8667 | 0.7163 | 0.7844 | 490 |
| COMPANY | 0.7994 | 0.6879 | 0.7395 | 2,932 |
| ACCOUNT | 0.8522 | 0.6141 | 0.7138 | 526 |

Missed PII rate (1 - recall) on test: 21.1% of entities.

## Observations

- On test, precision is higher than recall for every type: when the CRF
  marks something as PII it is usually right, but it misses about one in five
  entities.
- COMPANY and ACCOUNT are the hardest types. Company names look like normal
  capitalized words, and account numbers look like other numbers in the
  documents.
- The learned transitions follow the BIO rules without being told:
  `O -> I-COMPANY` (-12.3), `O -> I-DATE` (-12.0), `O -> I-ADDRESS` (-11.6)
  and `O -> I-NAME` (-10.5) get the strongest negative weights.
- Token-level macro F1 is pulled down by very rare inside tags (`I-ID` has
  9 test tokens and 0 validation tokens, `I-ACCOUNT` 38 test tokens), so the
  entity-level scores are the better summary.

## Limitations

- c1/c2 were searched on a 5,000-chunk subset; the best c1 (0.25) is at the
  edge of the grid, so a larger c1 might be slightly better.
- The final model stopped at 200 iterations; the loss fell by only 1.2%
  over the last 50 iterations.
- Features are hand-written; the model cannot use word meaning beyond the
  exact training words.
