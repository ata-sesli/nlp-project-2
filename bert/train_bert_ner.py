"""Fine-tune BERT for PII detection as token classification (BIO tags).

Reads the shared dataset (dataset/data/*.jsonl.gz + labels.json), fine-tunes a pretrained
model, keeps the checkpoint with the best validation entity F1, and writes:

    <out>/predictions/val.jsonl, test.jsonl   {"id": ..., "pred": [one tag per word]}
    <out>/results.json                        settings, timing, entity scores per split and type
    <out>/model/                              fine-tuned weights + tokenizer

Trial run (small subset, one epoch):
    python bert/train_bert_ner.py --train-limit 5000 --epochs 1 --out /content/drive/MyDrive/NLP-2/bert_trial
Full run:
    python bert/train_bert_ner.py --out /content/drive/MyDrive/NLP-2/bert_full
"""

import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "data"))
    p.add_argument("--model", default="bert-base-cased")
    p.add_argument("--out", required=True, help="output folder (put it on Google Drive)")
    p.add_argument("--train-limit", type=int, default=0, help="use only N random training chunks (0 = all)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512, help="max subword tokens per chunk")
    p.add_argument("--warmup", type=float, default=0.1, help="share of steps used for linear warm-up")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(data_dir, name):
    with gzip.open(Path(data_dir) / f"{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_long(rows, tok, max_len):
    """Split chunks whose subword length exceeds max_len into halves (recursively), so that
    no word is lost to truncation. Each piece remembers its source chunk and word offset."""
    pieces, n_split = [], 0
    for i, r in enumerate(rows):
        stack, parts = [(0, len(r["tokens"]))], []
        while stack:
            a, b = stack.pop()
            n = len(tok(r["tokens"][a:b], is_split_into_words=True)["input_ids"])
            if n <= max_len or b - a <= 1:
                parts.append((a, b))
            else:
                mid = (a + b) // 2
                stack += [(mid, b), (a, mid)]
        n_split += len(parts) > 1
        for a, b in sorted(parts):
            pieces.append({"src": i, "start": a, "tokens": r["tokens"][a:b], "tags": r["tags"][a:b]})
    return pieces, n_split


def encode(rows, tok, label2id, max_len):
    """Tokenize words into subwords and align labels.

    Only the FIRST subword of each word gets the word's label; the other subwords and the
    special tokens ([CLS], [SEP]) get -100, so the loss ignores them.
    """
    out, cut_words = [], 0
    for r in rows:
        enc = tok(r["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        word_ids = enc.word_ids()
        labels, prev = [], None
        for w in word_ids:
            if w is None or w == prev:
                labels.append(-100)          # special token or a later subword of the same word
            else:
                labels.append(label2id[r["tags"][w]])
            prev = w
        seen = {w for w in word_ids if w is not None}
        cut_words += len(r["tokens"]) - len(seen)   # words lost to truncation (should be ~0)
        enc["labels"] = labels
        out.append(dict(enc))
    return out, cut_words


def entity_scores(gold, pred):
    """Entity-level scores (seqeval, strict IOB2): an entity counts only if span and type match."""
    kw = dict(mode="strict", scheme=IOB2)
    return {
        "precision": precision_score(gold, pred, **kw),
        "recall": recall_score(gold, pred, **kw),
        "f1": f1_score(gold, pred, **kw),
        "missed_pii_rate": 1 - recall_score(gold, pred, **kw),
        "token_accuracy": float(np.mean([g == p for gs, ps in zip(gold, pred) for g, p in zip(gs, ps)])),
        "per_type": classification_report(gold, pred, output_dict=True, **kw),
    }


def make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        gold_seqs, pred_seqs = [], []
        for p_row, l_row in zip(preds, labels):
            keep = l_row != -100                 # first subword of every word
            gold_seqs.append([id2label[i] for i in l_row[keep]])
            pred_seqs.append([id2label[i] for i in p_row[keep]])
        s = entity_scores(gold_seqs, pred_seqs)
        return {"f1": s["f1"], "precision": s["precision"], "recall": s["recall"]}
    return compute_metrics


@torch.no_grad()
def predict_words(model, tok, rows, id2label, max_len, batch_size, device):
    """One tag per WORD: the prediction of its first subword. Long chunks are predicted in
    pieces (split_long) and put back together, so every word gets a prediction."""
    model.eval()
    pieces, _ = split_long(rows, tok, max_len)
    results = [["O"] * len(r["tokens"]) for r in rows]
    for i in range(0, len(pieces), batch_size):
        batch = pieces[i:i + batch_size]
        enc = tok([r["tokens"] for r in batch], is_split_into_words=True, truncation=True,
                  max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1).cpu().numpy()
        for b, piece in enumerate(batch):
            tags = results[piece["src"]]
            prev = None
            for pos, w in enumerate(enc.word_ids(b)):
                if w is not None and w != prev:
                    tags[piece["start"] + w] = id2label[int(pred_ids[b, pos])]
                prev = w
    return results


def main():
    a = parse_args()
    set_seed(a.seed)
    out = Path(a.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = json.load(open(Path(a.data_dir) / "labels.json"))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    train, val, test = (load_split(a.data_dir, s) for s in ("train", "val", "test"))
    if a.train_limit:
        train = random.Random(a.seed).sample(train, min(a.train_limit, len(train)))
    print(f"device={device} model={a.model} train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    train_pieces, split_train = split_long(train, tok, a.max_len)
    val_pieces, split_val = split_long(val, tok, a.max_len)
    print(f"chunks longer than {a.max_len} subwords, split in pieces: train {split_train}, val {split_val}")
    train_enc, cut_train = encode(train_pieces, tok, label2id, a.max_len)
    val_enc, cut_val = encode(val_pieces, tok, label2id, a.max_len)
    print(f"words cut by the {a.max_len}-token limit: train {cut_train}, val {cut_val}  (should be 0)")

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    steps = int(np.ceil(len(train_enc) / a.batch_size) * a.epochs)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        warmup_steps=int(a.warmup * steps),     # linear warm-up, then linear decay
        weight_decay=0.01,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        fp16=device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",              # validation entity F1
        logging_steps=50,
        report_to="none",
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_enc, eval_dataset=val_enc,
                      data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=make_compute_metrics(id2label))

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    model = trainer.model.to(device)
    trainer.save_model(str(out / "model"))
    tok.save_pretrained(str(out / "model"))

    results = {"settings": vars(a), "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "train_chunks": len(train), "train_seconds": round(train_seconds, 1),
               "words_cut_by_max_len": {"train": cut_train, "val": cut_val},
               "chunks_split_for_max_len": {"train": split_train, "val": split_val},
               # validation entity F1 after every epoch: shows whether more epochs would help
               "val_history": [{"epoch": h["epoch"], "f1": h["eval_f1"], "precision": h["eval_precision"],
                                "recall": h["eval_recall"], "loss": h["eval_loss"]}
                               for h in trainer.state.log_history if "eval_f1" in h],
               "splits": {}}
    for h in results["val_history"]:
        print(f"epoch {h['epoch']:.0f}: val entity F1 {h['f1']:.4f}  loss {h['loss']:.4f}")
    for name, rows in (("val", val), ("test", test)):
        t0 = time.time()
        pred = predict_words(model, tok, rows, id2label, a.max_len, a.batch_size * 2, device)
        seconds = time.time() - t0
        with open(out / "predictions" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r, p in zip(rows, pred):
                f.write(json.dumps({"id": r["id"], "pred": p}) + "\n")
        s = entity_scores([r["tags"] for r in rows], pred)
        s["predict_seconds"] = round(seconds, 1)
        results["splits"][name] = s
        print(f"{name}: entity F1 {s['f1']:.4f}  P {s['precision']:.4f}  R {s['recall']:.4f}  "
              f"missed PII {s['missed_pii_rate']:.1%}  token acc {s['token_accuracy']:.4f}")

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("done ->", out)


if __name__ == "__main__":
    main()
