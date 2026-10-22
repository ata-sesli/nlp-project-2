
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import classification_report


DATA_DIR = Path("dataset/data")
RESULTS_DIR = Path("baselines/results")


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
        for token, tag in zip(row["tokens"], row["tags"]):
            token_tag_counts[token.lower()][tag] += 1
            global_tag_counts[tag] += 1

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


def evaluate(rows, predictions, title):
    true_tags = [
        tag
        for row in rows
        for tag in row["tags"]
    ]

    predicted_tags = [
        tag
        for row_predictions in predictions
        for tag in row_predictions
    ]

    report = classification_report(
        true_tags,
        predicted_tags,
        zero_division=0,
        digits=4,
        output_dict=True,
    )

    print(f"\n{'=' * 15} {title} {'=' * 15}")
    print(
        classification_report(
            true_tags,
            predicted_tags,
            zero_division=0,
            digits=4,
        )
    )

    return report


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    results = {
        "model": "Most Frequent Token Tag Baseline",
        "prediction_unit": "token",
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
