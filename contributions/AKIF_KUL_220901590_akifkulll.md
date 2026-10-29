# Akif Kul: contribution

**Role:** model 6 of the benchmark, the main model: BERT fine-tuned for token classification.
Everything I made is in the `bert/` folder.

## What I did

1. **Wrote the training script** (`bert/train_bert_ner.py`). It reads the shared dataset
   (`dataset/data/*.jsonl.gz`, `labels.json`), fine-tunes `bert-base-cased` with a token
   classification head (17 BIO tags) and writes predictions in the shared format.
2. **Aligned word labels with BERT's subwords.** BERT splits words into pieces ("Schmidt" ->
   Sch, ##mid, ##t). Only the first piece of each word gets the word's tag; the other pieces and
   the special tokens get -100, so the loss ignores them. At prediction time each word takes the
   tag predicted for its first piece.
3. **Fixed lost words at the 512-subword limit.** The first full run showed that about 4% of the
   chunks (XML documents, long IDs and keys) are longer than 512 subwords, so 22,891 training
   words were cut off. I changed the script to split such chunks into pieces and join the
   predictions back, so every word is seen (2 single words longer than the limit remain). Test
   entity F1 rose from 0.860 to 0.868.
4. **Trained the model** on Colab (A100): all 30,545 training chunks, 3 epochs, learning rate
   3e-5 with 10% linear warm-up, batch size 16, fp16, seed 42. The best epoch was chosen by
   validation entity F1; the test split was used once, at the end. Training took 7.6 minutes.
5. **Evaluated it** with seqeval (entity level, strict IOB2): test entity F1 0.868, precision
   0.860, recall 0.877, missed PII rate 12.3%, plus F1 per entity type and per epoch.
   Compared with the CRF (0.822), BERT finds clearly more PII (recall 0.877 vs 0.789) at the
   same precision; the largest gains are on ACCOUNT, ID and COMPANY, which depend on context,
   and the CRF stays slightly ahead only on EMAIL.
6. **Published the model** on the Hugging Face Hub with a model card written from the results
   (`bert/push_to_hub.py`).
7. **Documentation:** `bert/README.md` and the Colab notebook kept with its outputs
   (`bert/bert_colab.ipynb`).

I used Claude, an AI assistant, to help write and check the code. I ran the training, the
evaluation and the upload myself.
