"""Shared evaluation for every model in the benchmark.

All models are scored here with the same rules, so their numbers are comparable:

- Entity precision, recall and F1 with seqeval in strict IOB2 mode. An entity counts
  only if its boundaries and its type are both right. An I- tag without a matching
  B- before it is not an entity.
- Token accuracy (reported, not used for ranking).
- PII recall and missed PII rate (1 - recall).
- F1 per entity type.
- Error types: boundary, type, boundary and type, missed, invented.
- Confusion matrix of entity types per word (B- and I- merged, O for non-PII words).

Usage:
    python benchmark/evaluate.py --split test --pred bert/predictions/test.jsonl
"""

import argparse
import gzip
import json
from pathlib import Path

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2, Entities


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
GOLD_FILES = {
    "val": DATA_DIR / "val.jsonl.gz",
    "test": DATA_DIR / "test.jsonl.gz",
}
ERROR_TYPES = ["boundary", "type", "boundary_and_type", "missed", "invented"]


def load_labels():
    with open(DATA_DIR / "labels.json", encoding="utf-8") as file:
        return json.load(file)


def load_gold(split):
    """Return the gold chunks of a split as a list of {"id", "tags"}."""
    rows = []

    with gzip.open(GOLD_FILES[split], "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            rows.append({"id": row["id"], "tags": row["tags"]})

    return rows


def load_predictions(path):
    """Return {chunk id: predicted tags} from a shared-format prediction file."""
    predictions = {}

    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            predictions[row["id"]] = row["pred"]

    return predictions


def align(gold_rows, predictions, labels):
    """Put predictions in gold order and check that they are complete and valid."""
    allowed = set(labels)
    gold_tags = []
    pred_tags = []

    for row in gold_rows:
        if row["id"] not in predictions:
            raise ValueError(f"Missing prediction for chunk {row['id']}")

        pred = predictions[row["id"]]

        if len(pred) != len(row["tags"]):
            raise ValueError(
                f"Chunk {row['id']}: {len(pred)} predicted tags, "
                f"{len(row['tags'])} gold tags"
            )

        unknown = set(pred) - allowed
        if unknown:
            raise ValueError(f"Chunk {row['id']}: unknown tags {sorted(unknown)}")

        gold_tags.append(row["tags"])
        pred_tags.append(pred)

    extra = len(predictions) - len(gold_rows)
    if extra > 0:
        raise ValueError(f"{extra} predictions have no matching gold chunk")

    return gold_tags, pred_tags


def strict_entities(sequences):
    """Entities per chunk as sets of (type, start, end), end exclusive."""
    return [
        {(entity.tag, entity.start, entity.end) for entity in chunk}
        for chunk in Entities(sequences, IOB2).entities
    ]


def overlaps(a, b):
    return a[1] < b[2] and b[1] < a[2]


def count_error_types(gold_tags, pred_tags):
    """Sort every entity that is not exactly right into one error type.

    Each wrong gold entity is one of:
      type:              a predicted entity has the same span but another type
      boundary:          a predicted entity of the same type overlaps it
      boundary_and_type: only predicted entities of other types overlap it
      missed:            no predicted entity overlaps it
    A predicted entity that overlaps no gold entity is "invented".
    """
    counts = dict.fromkeys(ERROR_TYPES, 0)
    counts["correct"] = 0

    gold_chunks = strict_entities(gold_tags)
    pred_chunks = strict_entities(pred_tags)

    for gold, pred in zip(gold_chunks, pred_chunks):
        correct = gold & pred
        counts["correct"] += len(correct)

        for entity in gold - correct:
            overlapping = [p for p in pred if overlaps(entity, p)]

            if any(p[1:] == entity[1:] for p in overlapping):
                counts["type"] += 1
            elif any(p[0] == entity[0] for p in overlapping):
                counts["boundary"] += 1
            elif overlapping:
                counts["boundary_and_type"] += 1
            else:
                counts["missed"] += 1

        for entity in pred - correct:
            if not any(overlaps(entity, g) for g in gold):
                counts["invented"] += 1

    return counts


def entity_type(tag):
    return tag if tag == "O" else tag[2:]


def confusion_counts(gold_tags, pred_tags):
    """Word counts as {gold type: {predicted type: count}}, B- and I- merged."""
    types = list(dict.fromkeys(entity_type(tag) for tag in load_labels()))
    counts = {gold: dict.fromkeys(types, 0) for gold in types}

    for gold, pred in zip(gold_tags, pred_tags):
        for g, p in zip(gold, pred):
            counts[entity_type(g)][entity_type(p)] += 1

    return counts


def evaluate(gold_tags, pred_tags):
    """Compute every benchmark metric for one model on one split."""
    report = classification_report(
        gold_tags,
        pred_tags,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    micro = report["micro avg"]

    tokens = sum(len(tags) for tags in gold_tags)
    correct_tokens = sum(
        g == p
        for gold, pred in zip(gold_tags, pred_tags)
        for g, p in zip(gold, pred)
    )

    per_type = {
        name: {
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1": float(scores["f1-score"]),
            "support": int(scores["support"]),
        }
        for name, scores in sorted(report.items())
        if not name.endswith("avg")
    }

    return {
        "chunks": len(gold_tags),
        "tokens": tokens,
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "f1": float(micro["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "token_accuracy": correct_tokens / tokens,
        "missed_pii_rate": 1 - float(micro["recall"]),
        "per_type": per_type,
        "error_types": count_error_types(gold_tags, pred_tags),
        "confusion": confusion_counts(gold_tags, pred_tags),
    }


def evaluate_file(split, prediction_path):
    gold_tags, pred_tags = align(
        load_gold(split), load_predictions(prediction_path), load_labels()
    )
    return evaluate(gold_tags, pred_tags)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=sorted(GOLD_FILES), required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(evaluate_file(args.split, args.pred), indent=2))


if __name__ == "__main__":
    main()
