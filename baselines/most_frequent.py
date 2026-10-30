
"""Most Frequent Token Tag baseline with shared prediction exports."""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
RESULTS_DIR = ROOT / "baselines" / "results"
PREDICTIONS_DIR = ROOT / "baselines" / "predictions"


def load_data(filename):
    rows = []
    with gzip.open(DATA_DIR / filename, "rt", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def train_model(train_rows):
    token_tag_counts = defaultdict(Counter)
    global_tag_counts = Counter()

    for row in train_rows:
        if len(row["tokens"]) != len(row["tags"]):
            raise ValueError(f"Token/tag length mismatch: {row['id']}")

        for token, tag in zip(row["tokens"], row["tags"]):
            token_tag_counts[token.lower()][tag] += 1
            global_tag_counts[tag] += 1

    if not global_tag_counts:
        raise ValueError("Training data contains no tokens.")

    token_to_tag = {
        token: counts.most_common(1)[0][0]
        for token, counts in token_tag_counts.items()
    }

    default_tag = global_tag_counts.most_common(1)[0][0]
    return token_to_tag, default_tag


def predict(rows, token_to_tag, default_tag):
    return [
        [
            token_to_tag.get(token.lower(), default_tag)
            for token in row["tokens"]
        ]
        for row in rows
    ]


def save_predictions(rows, predictions, filename):
    output_path = PREDICTIONS_DIR / filename

    if len(rows) != len(predictions):
        raise ValueError("Prediction sequence count mismatch.")

    with output_path.open("w", encoding="utf-8") as file:
        for row, predicted_tags in zip(rows, predictions):
            if len(row["tokens"]) != len(predicted_tags):
                raise ValueError(f"Prediction length mismatch: {row['id']}")

            record = {
                "id": row["id"],
                "doc_id": row.get("doc_id"),
                "pred": predicted_tags,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Predictions saved to: {output_path}")


def evaluate(rows, predictions, title):
    true_tags = [tag for row in rows for tag in row["tags"]]
    predicted_tags = [tag for sequence in predictions for tag in sequence]

    report = classification_report(
        true_tags,
        predicted_tags,
        zero_division=0,
        digits=4,
        output_dict=True,
    )

    print(f"\n{'=' * 15} {title} {'=' * 15}")
    print(classification_report(
        true_tags,
        predicted_tags,
        zero_division=0,
        digits=4,
    ))

    return report


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_data("train.jsonl.gz")
    val_rows = load_data("val.jsonl.gz")
    test_rows = load_data("test.jsonl.gz")

    token_to_tag, default_tag = train_model(train_rows)

    print(f"Training chunks: {len(train_rows)}")
    print(f"Validation chunks: {len(val_rows)}")
    print(f"Test chunks: {len(test_rows)}")
    print(f"Unique tokens learned: {len(token_to_tag)}")
    print(f"Fallback tag: {default_tag}")

    val_predictions = predict(val_rows, token_to_tag, default_tag)
    val_report = evaluate(val_rows, val_predictions, "VALIDATION RESULTS")

    test_predictions = predict(test_rows, token_to_tag, default_tag)
    test_report = evaluate(test_rows, test_predictions, "TEST RESULTS")

    save_predictions(
        val_rows, val_predictions, "most_frequent_validation.jsonl"
    )
    save_predictions(
        test_rows, test_predictions, "most_frequent_test.jsonl"
    )

    results = {
        "model": "Most Frequent Token Tag Baseline",
        "prediction_unit": "token",
        "prediction_format": "JSONL with id, doc_id and pred fields",
        "fallback_tag": default_tag,
        "training_chunks": len(train_rows),
        "validation_chunks": len(val_rows),
        "test_chunks": len(test_rows),
        "unique_tokens": len(token_to_tag),
        "validation": val_report,
        "test": test_report,
    }

    output_file = RESULTS_DIR / "most_frequent_results.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
