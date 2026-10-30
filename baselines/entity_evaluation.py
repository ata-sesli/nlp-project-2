
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
PREDICTIONS_DIR = ROOT / "baselines" / "predictions"
RESULTS_DIR = ROOT / "baselines" / "results"

MODELS = {
    "Most Frequent": {
        "validation": "most_frequent_validation.jsonl",
        "test": "most_frequent_test.jsonl",
    },
    "HMM + Viterbi": {
        "validation": "hmm_viterbi_validation.jsonl",
        "test": "hmm_viterbi_test.jsonl",
    },
}

SPLITS = {
    "validation": DATA_DIR / "val.jsonl.gz",
    "test": DATA_DIR / "test.jsonl.gz",
}


def load_gold_data(path):
    """Load gold-standard records from a compressed JSONL file."""
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def load_predictions(path):
    """Load model predictions from JSONL."""
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def extract_entities(tags):
    """
    Extract entities as (entity_type, start, end) tuples.

    The end position is exclusive.
    Invalid I-tags without a matching preceding entity
    are treated as the beginning of a new entity.
    """
    entities = set()
    current_type = None
    start = None

    for index, tag in enumerate(tags):
        if tag == "O":
            if current_type is not None:
                entities.add((current_type, start, index))
                current_type = None
                start = None
            continue

        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag: {tag}")

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            if current_type is not None:
                entities.add((current_type, start, index))

            current_type = entity_type
            start = index

        elif prefix == "I":
            if current_type == entity_type:
                continue

            # Repair an invalid I-tag by starting a new entity.
            if current_type is not None:
                entities.add((current_type, start, index))

            current_type = entity_type
            start = index

        else:
            raise ValueError(f"Unknown BIO prefix: {prefix}")

    if current_type is not None:
        entities.add((current_type, start, len(tags)))

    return entities


def calculate_entity_metrics(gold_rows, prediction_rows):
    """Calculate exact-span entity-level precision, recall and F1."""
    if len(gold_rows) != len(prediction_rows):
        raise ValueError("Gold/prediction record count mismatch.")

    true_entities = set()
    predicted_entities = set()

    for gold, prediction in zip(gold_rows, prediction_rows):
        if gold["id"] != prediction["id"]:
            raise ValueError(
                f"Record ID mismatch: {gold['id']} != {prediction['id']}"
            )

        gold_tags = gold["tags"]
        predicted_tags = prediction["pred"]

        if len(gold_tags) != len(predicted_tags):
            raise ValueError(
                f"Token/tag length mismatch: {gold['id']}"
            )

        gold_spans = extract_entities(gold_tags)
        predicted_spans = extract_entities(predicted_tags)

        # Include record ID so spans from different records
        # cannot accidentally match.
        true_entities.update(
            (gold["id"], entity_type, start, end)
            for entity_type, start, end in gold_spans
        )

        predicted_entities.update(
            (gold["id"], entity_type, start, end)
            for entity_type, start, end in predicted_spans
        )

    correct = true_entities & predicted_entities

    tp = len(correct)
    fp = len(predicted_entities - true_entities)
    fn = len(true_entities - predicted_entities)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # Per-entity-type metrics.
    entity_types = sorted(
        {entity[1] for entity in true_entities | predicted_entities}
    )

    per_type = {}

    for entity_type in entity_types:
        gold_type = {
            entity for entity in true_entities
            if entity[1] == entity_type
        }
        predicted_type = {
            entity for entity in predicted_entities
            if entity[1] == entity_type
        }

        type_tp = len(gold_type & predicted_type)
        type_fp = len(predicted_type - gold_type)
        type_fn = len(gold_type - predicted_type)

        type_precision = (
            type_tp / (type_tp + type_fp)
            if type_tp + type_fp else 0.0
        )
        type_recall = (
            type_tp / (type_tp + type_fn)
            if type_tp + type_fn else 0.0
        )
        type_f1 = (
            2 * type_precision * type_recall
            / (type_precision + type_recall)
            if type_precision + type_recall else 0.0
        )

        per_type[entity_type] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "support": len(gold_type),
            "predicted": len(predicted_type),
            "true_positive": type_tp,
            "false_positive": type_fp,
            "false_negative": type_fn,
        }

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "gold_entities": len(true_entities),
            "predicted_entities": len(predicted_entities),
        },
        "per_entity_type": per_type,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    csv_rows = []

    for split_name, gold_path in SPLITS.items():
        gold_rows = load_gold_data(gold_path)
        all_results[split_name] = {}

        for model_name, prediction_files in MODELS.items():
            prediction_path = (
                PREDICTIONS_DIR / prediction_files[split_name]
            )

            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Prediction file not found: {prediction_path}"
                )

            prediction_rows = load_predictions(prediction_path)

            metrics = calculate_entity_metrics(
                gold_rows,
                prediction_rows,
            )

            all_results[split_name][model_name] = metrics

            overall = metrics["overall"]

            print(f"\n{split_name.upper()} — {model_name}")
            print(
                f"Entity Precision: {overall['precision']:.4f}"
            )
            print(
                f"Entity Recall:    {overall['recall']:.4f}"
            )
            print(
                f"Entity F1:        {overall['f1']:.4f}"
            )
            print(
                f"TP: {overall['true_positive']} | "
                f"FP: {overall['false_positive']} | "
                f"FN: {overall['false_negative']}"
            )

            csv_rows.append({
                "split": split_name,
                "model": model_name,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "true_positive": overall["true_positive"],
                "false_positive": overall["false_positive"],
                "false_negative": overall["false_negative"],
            })

            print("\nPer-entity-type F1:")
            for entity_type, values in metrics["per_entity_type"].items():
                print(
                    f"{entity_type:10s} "
                    f"P={values['precision']:.4f} "
                    f"R={values['recall']:.4f} "
                    f"F1={values['f1']:.4f} "
                    f"Support={values['support']}"
                )

    json_path = RESULTS_DIR / "entity_level_results.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    csv_path = RESULTS_DIR / "entity_level_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "split",
            "model",
            "precision",
            "recall",
            "f1",
            "true_positive",
            "false_positive",
            "false_negative",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nJSON results saved to: {json_path}")
    print(f"CSV comparison saved to: {csv_path}")


if __name__ == "__main__":
    main()
