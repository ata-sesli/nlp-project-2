import argparse
import gzip
import itertools
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import sklearn_crfsuite
from seqeval.metrics import classification_report as entity_report
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report

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

RESULTS_PATH = RESULTS_DIR / "crf_results.json"
MODEL_PATH = MODEL_DIR / "crf.joblib"

SEED = 42

# Small grid for the L1 (c1) and L2 (c2) regularization weights.
C1_GRID = [0.05, 0.1, 0.25]
C2_GRID = [0.01, 0.05, 0.1]


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_jsonl_gz(path):
    """Load records from a gzipped JSONL file."""
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def load_labels(path):
    """Load the label list (a JSON list or {"labels": [...]})."""
    with open(path, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    if isinstance(label_data, dict):
        return label_data["labels"]

    return label_data


def feature_stream(rows):
    """Yield feature sequences one chunk at a time to save memory."""
    for row in rows:
        yield sentence_features(row["tokens"])


def tag_stream(rows):
    """Yield gold tag sequences one chunk at a time."""
    for row in rows:
        yield row["tags"]


# --------------------------------------------------
# CRF training and prediction
# --------------------------------------------------

def train_crf(rows, c1, c2, max_iterations):
    """Train a linear-chain CRF with L-BFGS."""
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )

    model.fit(feature_stream(rows), tag_stream(rows))

    return model


def predict(model, rows):
    """Predict one tag sequence per chunk."""
    predictions = []

    for row in rows:
        tags = model.predict_single(sentence_features(row["tokens"]))

        if len(tags) != len(row["tokens"]):
            raise ValueError(f"Prediction length mismatch: {row['id']}")

        predictions.append(tags)

    return predictions


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(rows, predictions, labels, split_name):
    """Token-level report (same as the other baselines) + entity-level report."""
    true_sequences = [row["tags"] for row in rows]

    true_tags = [tag for sequence in true_sequences for tag in sequence]
    predicted_tags = [tag for sequence in predictions for tag in sequence]

    token_report = classification_report(
        true_tags,
        predicted_tags,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Entity level: an entity is correct only if both its boundaries
    # and its type are correct (seqeval, strict IOB2).
    span_report = entity_report(
        true_sequences,
        predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{split_name} results (token level)")
    print(
        classification_report(
            true_tags,
            predicted_tags,
            labels=labels,
            zero_division=0,
        )
    )

    print(f"{split_name} results (entity level, seqeval strict)")
    print(
        entity_report(
            true_sequences,
            predictions,
            mode="strict",
            scheme=IOB2,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "split": split_name,
        "chunks": len(rows),
        "tokens": len(true_tags),
        "classification_report": token_report,
        "entity_report": json.loads(json.dumps(span_report, default=float)),
    }


def entity_micro_f1(rows, predictions):
    """Entity-level micro F1, used to pick c1 and c2 on validation."""
    report = entity_report(
        [row["tags"] for row in rows],
        predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    return float(report["micro avg"]["f1-score"])


def save_predictions(rows, predictions, path):
    """Write {"id", "pred"} lines in the shared prediction format."""
    with open(path, "w", encoding="utf-8") as file:
        for row, tags in zip(rows, predictions):
            line = {"id": row["id"], "pred": tags}
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


# --------------------------------------------------
# Hyperparameter search on validation
# --------------------------------------------------

def try_setting(args):
    """Train on the tuning subset and score one (c1, c2) pair on validation."""
    tune_rows, val_rows, c1, c2, max_iterations = args

    model = train_crf(tune_rows, c1, c2, max_iterations)
    score = entity_micro_f1(val_rows, predict(model, val_rows))

    print(f"c1={c1:<5} c2={c2:<5} val entity F1={score:.4f}", flush=True)

    return {"c1": c1, "c2": c2, "val_entity_f1": score}


def tune(train_rows, val_rows, subset_size, max_iterations, workers):
    """Pick c1/c2 with a grid search on a random training subset."""
    rng = random.Random(SEED)
    tune_rows = rng.sample(train_rows, min(subset_size, len(train_rows)))

    jobs = [
        (tune_rows, val_rows, c1, c2, max_iterations)
        for c1, c2 in itertools.product(C1_GRID, C2_GRID)
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        grid = list(executor.map(try_setting, jobs))

    best = max(grid, key=lambda item: item["val_entity_f1"])

    return best, grid, len(tune_rows)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRF NER for PII detection")
    parser.add_argument("--tune-subset", type=int, default=5000,
                        help="training chunks used for the c1/c2 search")
    parser.add_argument("--tune-iterations", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="L-BFGS iterations for the final model")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel processes for the grid search")
    parser.add_argument("--c1", type=float, default=None,
                        help="skip tuning and use this c1 (needs --c2)")
    parser.add_argument("--c2", type=float, default=None)
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

    # 1. Choose c1 and c2 on validation.
    if args.c1 is not None and args.c2 is not None:
        best = {"c1": args.c1, "c2": args.c2}
        grid, tune_size = [], 0
        print(f"\nUsing given c1={args.c1}, c2={args.c2} (no tuning)")
    else:
        print(f"\nGrid search on {args.tune_subset} training chunks")
        start = time.time()
        best, grid, tune_size = tune(
            train_rows,
            val_rows,
            args.tune_subset,
            args.tune_iterations,
            args.workers,
        )
        print(f"Grid search time: {time.time() - start:.0f} s")
        print(f"Best: c1={best['c1']}, c2={best['c2']}")

    # 2. Train the final model on the full training split.
    print("\nTraining the final CRF on all training chunks...")
    start = time.time()
    model = train_crf(train_rows, best["c1"], best["c2"], args.max_iterations)
    train_seconds = time.time() - start
    print(f"Training time: {train_seconds:.0f} s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 3. Predict validation and test (test only once, at the end).
    start = time.time()
    val_predictions = predict(model, val_rows)
    test_predictions = predict(model, test_rows)
    predict_seconds = time.time() - start

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(val_rows, val_predictions, PREDICTIONS_DIR / "crf_val.jsonl")
    save_predictions(test_rows, test_predictions, PREDICTIONS_DIR / "crf_test.jsonl")

    val_results = evaluate(val_rows, val_predictions, labels, "Validation")
    test_results = evaluate(test_rows, test_predictions, labels, "Test")

    # 4. Most important learned transitions (for the error analysis).
    transitions = sorted(
        model.transition_features_.items(),
        key=lambda item: item[1],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "CRF",
        "library": "sklearn-crfsuite (L-BFGS)",
        "prediction_unit": "token",
        "features": "word, prefix/suffix 3, shape, case, digits, @; window of 2",
        "seed": SEED,
        "c1": best["c1"],
        "c2": best["c2"],
        "max_iterations": args.max_iterations,
        "tuning_subset_chunks": tune_size,
        "tuning_grid": grid,
        "train_chunks": len(train_rows),
        "train_seconds": round(train_seconds, 1),
        "predict_seconds_val_and_test": round(predict_seconds, 1),
        "num_features": len(model.state_features_),
        "most_unlikely_transitions": [
            {"from": a, "to": b, "weight": w}
            for (a, b), w in transitions[:10]
        ],
        "most_likely_transitions": [
            {"from": a, "to": b, "weight": w}
            for (a, b), w in transitions[-10:][::-1]
        ],
        "validation": val_results,
        "test": test_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
