# Enes Özbek: contribution

**Role:** model 5 of the benchmark, the BiLSTM-CRF (neural sequence model). Everything I made
is in the `bilstm_crf/` folder.

## What I did

1. **Wrote the training script** (`bilstm_crf/train_bilstm_crf.py`). It reads the shared
   dataset (`dataset/data/*.jsonl.gz`, `labels.json`) and builds a BiLSTM-CRF in PyTorch:
   GloVe 6B 100d word vectors, a character CNN, a 2-layer bidirectional LSTM (128 units per
   direction) and a CRF layer (`pytorch-crf`) that picks the whole tag sequence with Viterbi.
2. **Added a character layer to the plan's GloVe model.** GloVe only has lowercase words, so
   without it the model cannot see capital letters, and IBANs, IDs and most numbers are
   unknown words. The character layer reads each word letter by letter, which is the standard
   BiLSTM-CRF design (Lample et al., 2016).
3. **Built the vocabularies from the training split only:** 56,829 words (92.1% of training
   words have a GloVe vector) and 298 characters. Unknown words map to `<unk>`.
4. **Trained the model** on Google Colab (T4 GPU): all 30,545 training chunks, Adam with
   learning rate 0.001, batch size 32, dropout 0.5, seed 42. After every epoch the model was
   scored on the validation split; epoch 9 was the best and training stopped after epoch 12.
   The test split was used once, at the end. Training took 49 minutes.
5. **Produced the predictions** in the shared format (`bilstm_crf/predictions/val.jsonl`,
   `test.jsonl`: one `{"id", "pred"}` line per chunk).
6. **Evaluated the model** at two levels and saved everything to
   `bilstm_crf/results/results.json`:
   - Entity level (seqeval, strict IOB2): test micro F1 0.849, precision 0.864, recall 0.835,
     missed PII rate 16.5%.
   - Token level, with the same metrics as the other baselines: test macro F1 0.816.
7. **Analysed the results** in `bilstm_crf/README.md`: validation F1 per epoch, F1 per entity
   type compared with the CRF (0.822) and BERT (0.868), where the learned features help most
   (ACCOUNT, COMPANY, NAME, ADDRESS) and where the hand-written ones still win (EMAIL), and
   limitations.

I used Claude, an AI assistant, to help write the code and the analysis.

## Where my work is

- `bilstm_crf/`: `train_bilstm_crf.py`, `README.md`, `predictions/`, `results/`, `.gitignore`
- This file.
