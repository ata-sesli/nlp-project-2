# BERT (main model)

Model 6 of the benchmark: `bert-base-cased` fine-tuned for token classification. It reads a
financial document chunk word by word and gives every word one of 17 BIO tags (8 PII types).

- Architecture: BERT (transformer encoder)
- Specific model: `bert-base-cased` + a linear token classification head (17 outputs)
- Role: the one model we fine-tune; the other five models are the baselines
- Published model: [`akifkulll/finance-pii-bert-base-cased`](https://huggingface.co/akifkulll/finance-pii-bert-base-cased)

**Why cased:** capitalization is one of the information sources taggers use (course slides,
POS/NER intro), and it is a strong hint for names and companies.

## Files

| File | What it does |
|---|---|
| `train_bert_ner.py` | Fine-tunes the model, picks the best epoch on validation, writes predictions and results |
| `bert_colab.ipynb` | Runs the script on Colab (A100), shows the results and uploads the model; kept with its outputs |
| `push_to_hub.py` | Uploads the model to the Hugging Face Hub with a model card written from `results.json` |
| `predictions/val.jsonl`, `predictions/test.jsonl` | One line per chunk in the shared format: `{"id": ..., "pred": [one tag per word]}` |
| `results/results.json` | Settings, timing, validation F1 per epoch, entity scores per split and per type |

The fine-tuned weights (431 MB) are not in the repo; they are on the Hugging Face Hub.

## How to run

From the repository root, on a GPU:

```bash
pip install transformers torch seqeval
python bert/train_bert_ner.py --out runs/bert_full
```

Options: `--model`, `--epochs`, `--lr`, `--batch-size`, `--max-len`, `--train-limit N` (quick test
on N random training chunks), `--seed`.

## How it works

1. **Subword alignment.** BERT splits words into pieces ("Schmidt" -> `Sch`, `##mid`, `##t`).
   Only the first piece of each word gets the word's tag; the other pieces and the special
   tokens get -100, so the loss ignores them. At prediction time each word takes the tag
   predicted for its first piece, so the output has exactly one tag per word, like every other
   model in the benchmark.
2. **Long chunks.** Chunks have at most 200 words, but 4.1% of them are longer than BERT's 512
   subword limit (XML documents, long IDs and keys split into many pieces). These chunks are
   split into halves until each piece fits, and the predictions are joined back. Without this,
   22,891 training words were cut off and test entity F1 was 0.860 instead of 0.868. Two single
   "words" longer than 512 subwords remain cut.
3. **Training.** All 30,545 training chunks, 3 epochs, AdamW, learning rate 3e-5 with 10% linear
   warm-up and linear decay, weight decay 0.01, batch size 16, fp16, seed 42. After each epoch
   the model is scored on the validation split; the best epoch is kept. The test split is used
   once, at the end.

## Results

Entity level, seqeval strict IOB2: an entity counts only if its boundaries and its type are both
right. Hardware: NVIDIA A100 (40 GB), training time 7.6 minutes.

### Validation F1 per epoch

| Epoch | Validation entity F1 | Validation loss |
|---:|---:|---:|
| 1 | 0.8389 | 0.0487 |
| 2 | 0.8575 | 0.0428 |
| 3 | **0.8613** | 0.0418 |

The gain shrinks from +1.9 to +0.4 points and the loss keeps falling, so 3 epochs is close to the
plateau without signs of overfitting.

### Overall

| Split | Precision | Recall | F1 | Missed PII | Token accuracy |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8535 | 0.8690 | 0.8612 | 13.1% | 0.9866 |
| Test | 0.8595 | 0.8772 | **0.8683** | **12.3%** | 0.9872 |

### Test F1 per entity type, compared with the CRF

| Type | Precision | Recall | BERT F1 | CRF F1 | Difference |
|---|---:|---:|---:|---:|---:|
| ACCOUNT | 0.9113 | 0.8403 | 0.8744 | 0.7138 | +16.1 |
| ID | 0.9016 | 0.9347 | 0.9178 | 0.7844 | +13.3 |
| COMPANY | 0.8053 | 0.8462 | 0.8252 | 0.7395 | +8.6 |
| NAME | 0.9040 | 0.8520 | 0.8772 | 0.8314 | +4.6 |
| ADDRESS | 0.8753 | 0.9041 | 0.8895 | 0.8638 | +2.6 |
| DATE | 0.8473 | 0.8925 | 0.8693 | 0.8464 | +2.3 |
| PHONE | 0.8785 | 0.9228 | 0.9001 | 0.8814 | +1.9 |
| EMAIL | 0.9116 | 0.9419 | 0.9265 | 0.9412 | -1.5 |

CRF numbers are from `svm_crf/README.md`.

## Observations

- **BERT finds more PII at the same precision.** Against the CRF, test F1 rises from 0.822 to
  0.868; precision stays at about 0.86 while recall rises from 0.789 to 0.877, so the missed PII
  rate falls from 21.1% to 12.3%.
- **The gain is largest where context decides the type.** Account numbers and IDs look like other
  numbers and company names look like ordinary capitalized words; the CRF only sees a window of
  hand-written features, while BERT's representations are learned from the whole chunk. On types
  with a fixed shape the hand-written features are already strong: PHONE gains only 1.9 points,
  and EMAIL is the one type where the CRF stays ahead (0.941 vs 0.927), because its "contains @"
  feature almost decides the type on its own, while BERT sees an address as several subwords.
- **DATE improves least among the frequent types.** The source labels are inconsistent: words such
  as "monthly" are tagged DATE in some documents and O in others (see the dataset report), which
  caps every model.
- **Token accuracy is not informative here:** 92% of the words are O, so accuracy is 0.987 while
  entity F1 is 0.868.

## Limitations

- One training run (seed 42); variance across seeds was not measured.
- Synthetic, English-only data with many repeated names ("John Doe", "XYZ Corporation"), which may
  make scores higher than on real documents.
- In XML-like documents the word split keeps markup attached to some words (a known dataset
  limitation); all models use the same words, so the comparison stays fair.
