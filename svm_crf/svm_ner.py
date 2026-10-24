import argparse
import gzip
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
from seqeval.metrics import classification_report as entity_report
from seqeval.scheme import IOB2
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from features import sentence_features

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
OUTPUT_DIR = ROOT / "svm_crf"
RESULTS_DIR = OUTPUT_DIR / "results"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_PATH = DATA_DIR / "train.jsonl.gz"
VAL_PATH = DATA_DIR / "val.jsonl.gz"
TEST_PATH = DATA_DIR / "test.jsonl.gz"
LABELS_PATH = DATA_DIR / "labels.json"

RESULTS_PATH = RESULTS_DIR / "svm_results.json"
MODEL_PATH = MODEL_DIR / "svm.joblib"

SEED = 42

# Feature hashing keeps memory low: no vocabulary is stored and the
# feature dictionaries are never all kept in memory at the same time.
HASHER = FeatureHasher(
    n_features=2 ** 20,
    input_type="dict",
    alternate_sign=False,
    dtype=np.float32,
)

# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_jsonl_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows

def load_labels(path):
    with open(path, "r", encoding="utf-8") as file:
        label_data = json.load(file)
    if isinstance(label_data, dict):
        return label_data["labels"]
    return label_data

def vectorize(rows):
    """Turn all tokens of the given chunks into one sparse matrix, chunk by chunk."""
    stream = (
        token_features
        for row in rows
        for token_features in sentence_features(row["tokens"])
    )
    return HASHER.transform(stream)

def flat_tags(rows):
    return [tag for row in rows for tag in row["tags"]]

def split_by_chunk(rows, flat_predictions):
    """Cut the flat token predictions back into one list per chunk."""
    sequences = []
    start = 0
    for row in rows:
        length = len(row["tokens"])
        sequences.append([str(tag) for tag in flat_predictions[start:start + length]])
        start += length
    return sequences

def save_predictions(rows, sequences, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row, tags in zip(rows, sequences):
            line = {"id": row["id"], "pred": tags}
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def entity_micro_f1(rows, sequences):
    report = entity_report(
        [row["tags"] for row in rows],
        sequences,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    return float(report["micro avg"]["f1-score"])

def evaluate(rows, sequences, labels, split_name):
    """Token-level report (same as the other baselines) + entity-level report."""
    true_sequences = [row["tags"] for row in rows]
    true_tags = [tag for sequence in true_sequences for tag in sequence]
    predicted_tags = [tag for sequence in sequences for tag in sequence]

    token_report = classification_report(
        true_tags, predicted_tags, labels=labels, output_dict=True, zero_division=0
    )
    span_report = entity_report(
        true_sequences, sequences, mode="strict", scheme=IOB2,
        output_dict=True, zero_division=0,
    )

    print(f"\n{split_name} results (token level)")
    print(classification_report(true_tags, predicted_tags, labels=labels, zero_division=0))
    print(f"{split_name} results (entity level, seqeval strict)")
    print(entity_report(
        true_sequences, sequences, mode="strict", scheme=IOB2,
        digits=4, zero_division=0,
    ))

    return {
        "split": split_name,
        "chunks": len(rows),
        "tokens": len(true_tags),
        "classification_report": token_report,
        "entity_report": json.loads(json.dumps(span_report, default=float)),
    }

def train_svm(X, y, c):
    clf = LinearSVC(C=c, dual="auto", max_iter=2000, random_state=SEED)
    clf.fit(X, y)
    return clf

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SVM NER for PII detection")
    parser.add_argument("--c-values", type=str, default="0.1,1.0,10.0",
                        help="Comma-separated C values to tune")
    parser.add_argument("--tune-subset", type=int, default=5000,
                        help="training chunks used to choose C")
    parser.add_argument("--c", type=float, default=None,
                        help="skip tuning and use this C")
    args = parser.parse_args()

    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_rows = load_jsonl_gz(TRAIN_PATH)
    val_rows = load_jsonl_gz(VAL_PATH)
    test_rows = load_jsonl_gz(TEST_PATH)
    labels = load_labels(LABELS_PATH)

    print(f"Train chunks: {len(train_rows)}")
    print(f"Validation chunks: {len(val_rows)}")
    print(f"Test chunks: {len(test_rows)}")

    X_val = vectorize(val_rows)

    # 1. Choose C on validation, training on a random subset to keep it fast.
    tuning_results = []
    tune_size = 0

    if args.c is not None:
        best_c = args.c
        print(f"\nUsing given C={best_c} (no tuning)")
    else:
        rng = random.Random(SEED)
        tune_rows = rng.sample(train_rows, min(args.tune_subset, len(train_rows)))
        tune_size = len(tune_rows)
        X_tune = vectorize(tune_rows)
        y_tune = flat_tags(tune_rows)

        c_grid = [float(c) for c in args.c_values.split(",")]
        print(f"\nTuning C on {tune_size} training chunks: {c_grid}")
        start = time.time()

        for c in c_grid:
            clf = train_svm(X_tune, y_tune, c)
            sequences = split_by_chunk(val_rows, clf.predict(X_val))
            score = entity_micro_f1(val_rows, sequences)
            print(f"  C={c} -> validation entity F1: {score:.4f}", flush=True)
            tuning_results.append({"C": c, "val_entity_f1": score})

        best_c = max(tuning_results, key=lambda item: item["val_entity_f1"])["C"]
        print(f"Tuning time: {time.time() - start:.0f} s")
        print(f"Best C selected: {best_c}")
        del X_tune, y_tune

    # 2. Train the final model on all training chunks.
    print("\nVectorizing all training chunks...")
    start = time.time()
    X_train = vectorize(train_rows)
    y_train = flat_tags(train_rows)
    print(f"Vectorization time: {time.time() - start:.0f} s")

    print(f"Training the final LinearSVC(C={best_c})...")
    start = time.time()
    model = train_svm(X_train, y_train, best_c)
    train_seconds = time.time() - start
    print(f"Training time: {train_seconds:.0f} s")
    del X_train, y_train

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 3. Predict validation and test (test only once, at the end).
    start = time.time()
    val_sequences = split_by_chunk(val_rows, model.predict(X_val))
    test_sequences = split_by_chunk(test_rows, model.predict(vectorize(test_rows)))
    predict_seconds = time.time() - start

    save_predictions(val_rows, val_sequences, PREDICTIONS_DIR / "svm_val.jsonl")
    save_predictions(test_rows, test_sequences, PREDICTIONS_DIR / "svm_test.jsonl")

    val_results = evaluate(val_rows, val_sequences, labels, "Validation")
    test_results = evaluate(test_rows, test_sequences, labels, "Test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "SVM",
        "library": "scikit-learn (LinearSVC)",
        "prediction_unit": "token",
        "features": "word, prefix/suffix 3, shape, case, digits, @; window of 2",
        "vectorizer": "FeatureHasher (2^20 features)",
        "seed": SEED,
        "best_C": best_c,
        "tuning_subset_chunks": tune_size,
        "tuning_grid": tuning_results,
        "train_chunks": len(train_rows),
        "train_seconds": round(train_seconds, 1),
        "predict_seconds_val_and_test": round(predict_seconds, 1),
        "validation": val_results,
        "test": test_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == '__main__':
    main()
