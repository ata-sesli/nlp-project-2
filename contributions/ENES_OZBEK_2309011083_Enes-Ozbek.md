# Enes Özbek: contribution

**Role:** model 4 of the benchmark, the CRF (Conditional Random Field). Everything I made is
in the `svm_crf/` folder.

## What I did

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

I used Claude, an AI assistant, to help write the code and run the training.

## Where my work is

- `svm_crf/`: `features.py`, `crf_ner.py`, `README.md`, `predictions/`, `results/`
- This file.
