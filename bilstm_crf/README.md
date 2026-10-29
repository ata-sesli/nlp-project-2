# BiLSTM-CRF

Model 5 of the benchmark: a neural sequence model trained on our data. It learns its own word
features instead of using hand-written ones, and a CRF layer on top picks the whole tag
sequence at once.

- Architecture: BiLSTM-CRF (neural sequence model)
- Specific model: GloVe 6B 100d word vectors + character CNN + 2-layer BiLSTM (128 units per
  direction) + CRF layer (`pytorch-crf`)
- Training: on `dataset/data/train.jsonl.gz` (30,545 chunks). All layers start from random
  weights; only the word vectors start from GloVe and are updated during training.
- GPU: optional. Trained on a Colab T4 GPU (49 minutes); on a 4-core CPU one epoch takes about
  17 minutes.

**Why a character layer:** GloVe only has lowercase words, so on its own the model cannot see
that "Anna" starts with a capital letter, and IBANs, IDs and most numbers are unknown words.
The character layer reads every word letter by letter (case kept), so it sees capital letters,
digits, `@` and word shapes. This is the standard BiLSTM-CRF design (Lample et al., 2016;
Ma and Hovy, 2016).

## Files

| File | What it does |
|---|---|
| `train_bilstm_crf.py` | Builds the vocabularies, trains the model, picks the best epoch on validation, writes predictions and results |
| `predictions/val.jsonl`, `predictions/test.jsonl` | One line per chunk in the shared format: `{"id": ..., "pred": [one tag per word]}` |
| `results/results.json` | Settings, vocabulary sizes, timing, validation F1 per epoch, token-level and entity-level reports |
| `models/`, `glove/` | Saved model (`bilstm_crf.pt`, 27 MB) and GloVe file; not committed, rebuilt by the script |

## How to run

From the repository root:

```bash
pip install torch pytorch-crf seqeval scikit-learn
python bilstm_crf/train_bilstm_crf.py
```

If `seqeval` fails to build, use `pip install --use-pep517 seqeval`. The script downloads
GloVe (134 MB) to `bilstm_crf/glove/` on the first run and uses a GPU when one is available.
On Colab: turn on a GPU, clone the repository, run the same two commands (PyTorch is already
installed there).

Options:

- `--embeddings random`: start the word vectors at random instead of from GloVe
- `--no-chars`: turn off the character layer
- `--glove PATH`: use a GloVe file you already have (`glove.6B.100d.txt` from the
  [GloVe page](https://nlp.stanford.edu/projects/glove/) also works)
- `--epochs 30`, `--patience 3`, `--batch-size 32`, `--lr 0.001`
- `--train-limit N`: quick test on N random training chunks

## How it works

1. **Words.** Each word is lowercased and looked up in a vocabulary built from the training
   split only: words that have a GloVe vector or appear at least twice (56,829 words). 92.1% of
   training words have a GloVe vector. Other words map to `<unk>`, so the model also learns a
   vector for words it has never seen.
2. **Characters.** The first 30 characters of each word (99.5% of words have 30 or fewer) go through
   a character embedding (30 dimensions) and a CNN (50 filters of width 3); the strongest value
   of each filter becomes the word's character vector.
3. **BiLSTM.** The word vector and the character vector are joined (150 numbers per word) and
   read by a 2-layer bidirectional LSTM, so each word's representation depends on the whole
   chunk, left and right.
4. **CRF.** A linear layer gives a score for each of the 17 tags, and a CRF layer learns a
   score for every tag-to-tag transition. Training maximizes the probability of the correct tag
   sequence; prediction uses Viterbi to pick the best whole sequence.
5. **Training.** Adam (learning rate 0.001), batch size 32, dropout 0.5, gradient clipping 5,
   seed 42. After every epoch the model is scored on the validation split (entity F1); the best
   epoch is kept, and training stops after 3 epochs without improvement. The test split is
   used once, at the end.

The settings follow the project plan (GloVe 100d, 2 layers of 128, Adam 0.001, early stopping)
and the usual values from Lample et al. and Ma and Hovy; no hyperparameter search was run.

## Results

Entity level, seqeval strict IOB2: an entity counts only if its boundaries and its type are both
right.

### Validation F1 per epoch

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Val entity F1 | 0.757 | 0.802 | 0.827 | 0.831 | 0.835 | 0.843 | 0.842 | 0.840 | **0.846** | 0.844 | 0.845 | 0.844 |
| Train loss | 20.78 | 7.18 | 5.20 | 4.25 | 3.72 | 3.35 | 3.09 | 2.86 | 2.70 | 2.55 | 2.40 | 2.28 |

Epoch 9 was kept. The score rises quickly for 6 epochs and then stays flat, while the training
loss keeps falling, so more epochs would mostly fit the training data more closely.

### Overall, compared with the CRF and BERT

| Split | Model | Precision | Recall | F1 | Macro F1 | Missed PII |
|---|---|---:|---:|---:|---:|---:|
| Validation | BiLSTM-CRF | 0.8580 | 0.8338 | 0.8457 | 0.8514 | 16.6% |
| Test | CRF | 0.8586 | 0.7887 | 0.8221 | 0.8252 | 21.1% |
| Test | **BiLSTM-CRF** | 0.8635 | 0.8353 | **0.8492** | 0.8469 | 16.5% |
| Test | BERT (fine-tuned) | 0.8595 | 0.8772 | 0.8683 | 0.8850 | 12.3% |

CRF numbers are from `svm_crf/README.md`, BERT numbers from `bert/README.md`.

### Token level (same metrics as the other baselines)

| Split | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | BiLSTM-CRF | 0.9842 | 0.8041 | 0.8421 | 0.8195 | 0.9842 |
| Test | HMM + Viterbi | 0.9532 | 0.5260 | 0.5096 | 0.4688 | 0.9550 |
| Test | CRF | 0.9813 | 0.8047 | 0.7683 | 0.7551 | 0.9811 |
| Test | BiLSTM-CRF | 0.9848 | 0.8083 | 0.8288 | 0.8163 | 0.9847 |

### Test F1 per entity type

| Type | Precision | Recall | BiLSTM-CRF F1 | CRF F1 | BERT F1 | Entities |
|---|---:|---:|---:|---:|---:|---:|
| EMAIL | 0.8876 | 0.9502 | 0.9178 | **0.9412** | 0.9265 | 241 |
| ADDRESS | 0.8937 | 0.8966 | **0.8951** | 0.8638 | 0.8895 | 1,856 |
| PHONE | 0.8515 | 0.9021 | 0.8761 | 0.8814 | **0.9001** | 337 |
| DATE | 0.8513 | 0.8739 | 0.8625 | 0.8464 | **0.8693** | 5,712 |
| NAME | 0.9172 | 0.8132 | 0.8621 | 0.8314 | **0.8772** | 3,447 |
| ACCOUNT | 0.8643 | 0.7262 | 0.7893 | 0.7138 | **0.8744** | 526 |
| COMPANY | 0.8198 | 0.7602 | 0.7889 | 0.7395 | **0.8252** | 2,932 |
| ID | 0.7958 | 0.7714 | 0.7834 | 0.7844 | **0.9178** | 490 |

## Observations

- **Learned features beat hand-written ones.** Against the CRF, test F1 rises from 0.822 to
  0.849. Precision stays at about 0.86 and recall rises from 0.789 to 0.835, so the missed PII
  rate falls from 21.1% to 16.5%.
- **The gains are on types that need context.** ACCOUNT (+7.6 points), COMPANY (+4.9), ADDRESS
  (+3.1) and NAME (+3.1) gain most: the BiLSTM sees the whole chunk, while the CRF sees only a
  window of two words on each side. EMAIL is the one type where the CRF stays clearly ahead
  (0.941 vs 0.918): its hand-written "contains @" feature almost decides the type on its own.
- **Pretraining adds the rest.** BERT is 1.9 points higher (0.868). The largest gaps are ID
  (+13.4) and ACCOUNT (+8.5). IDs and account numbers look like many other numbers in the
  documents, so their type depends on the meaning of nearby words ("customer number", "IBAN");
  BERT knows these words from pretraining, while the BiLSTM-CRF has to learn them from our
  30,545 chunks (our reading of the results; not tested separately). On ADDRESS the BiLSTM-CRF
  is slightly ahead of BERT (0.895 vs 0.890).
- **No broken BIO sequences.** On test, none of the 28,161 predicted `I-` tags follows a tag
  that cannot come before it (CRF: 4, SVM: 1,866). The CRF layer learned the BIO rules from the
  data.
- **Token accuracy is not informative here:** about 92% of words are `O`, so accuracy is 0.985
  while entity F1 is 0.849.

## Limitations

- One training run (seed 42) on a GPU; variance across seeds was not measured. GPU results are
  not bit-identical to CPU results: a CPU run of the same script reached validation F1 0.7587,
  0.8018 and 0.8250 after epochs 1 to 3, against 0.7574, 0.8019 and 0.8270 on the GPU.
- No hyperparameter search; the settings are the plan's and the usual published values.
- GloVe is lowercase and was trained on 2014 Wikipedia and news text, not on financial
  documents. The character layer sees only the first 30 characters of a word.
- Very rare inside tags (`I-ID`: 9 test tokens, F1 0.0) pull down token-level macro F1, so the
  entity-level scores are the better summary.
