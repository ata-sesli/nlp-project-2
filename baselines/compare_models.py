
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "baselines" / "results"

MOST_FREQUENT_PATH = RESULTS_DIR / "most_frequent_results.json"
HMM_PATH = RESULTS_DIR / "hmm_viterbi_results.json"

OUTPUT_PATH = RESULTS_DIR / "model_comparison.csv"
LABEL_OUTPUT_PATH = RESULTS_DIR / "label_f1_comparison.csv"


def load_results(path):
    """Load a model's JSON results."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_report(data, split):
    """Support both direct and nested classification reports."""
    if split not in data:
        raise KeyError(
            f"'{split}' not found. Available keys: {list(data.keys())}"
        )

    result = data[split]

    if not isinstance(result, dict):
        raise ValueError(f"Unexpected format for '{split}' results.")

    # HMM format: {"classification_report": {...}}
    if "classification_report" in result:
        return result["classification_report"]

    # Alternative nested report format.
    if "report" in result:
        return result["report"]

    # Direct sklearn classification_report output.
    if "macro avg" in result and "weighted avg" in result:
        return result

    raise ValueError(
        f"Could not find classification report for '{split}'. "
        f"Available keys: {list(result.keys())}"
    )


def metric(report, section, name, default=0.0):
    """Safely retrieve a metric."""
    value = report.get(section, {})

    if isinstance(value, dict):
        return value.get(name, default)

    return default


def main():
    for path in [MOST_FREQUENT_PATH, HMM_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

    most_frequent = load_results(MOST_FREQUENT_PATH)
    hmm = load_results(HMM_PATH)

    models = [
        ("Most Frequent", most_frequent),
        ("HMM + Viterbi", hmm),
    ]

    # Overall metrics for validation and test.
    rows = []

    for split in ["validation", "test"]:
        for model_name, results in models:
            report = get_report(results, split)

            rows.append({
                "split": split,
                "model": model_name,
                "accuracy": report.get("accuracy", 0.0),
                "macro_precision": metric(
                    report, "macro avg", "precision"
                ),
                "macro_recall": metric(
                    report, "macro avg", "recall"
                ),
                "macro_f1": metric(
                    report, "macro avg", "f1-score"
                ),
                "weighted_f1": metric(
                    report, "weighted avg", "f1-score"
                ),
            })

    comparison = pd.DataFrame(rows)

    print("\nOVERALL MODEL COMPARISON")
    print("=" * 90)
    print(comparison.round(4).to_string(index=False))

    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOverall comparison saved to: {OUTPUT_PATH}")

    # Per-label F1 comparison on the test set.
    mf_report = get_report(most_frequent, "test")
    hmm_report = get_report(hmm, "test")

    labels = sorted(
        label
        for label, values in hmm_report.items()
        if (
            isinstance(values, dict)
            and (
                label.startswith("B-")
                or label.startswith("I-")
            )
        )
    )

    label_rows = []

    for label in labels:
        mf_f1 = metric(mf_report, label, "f1-score")
        hmm_f1 = metric(hmm_report, label, "f1-score")

        label_rows.append({
            "label": label,
            "most_frequent_f1": mf_f1,
            "hmm_f1": hmm_f1,
            "difference": hmm_f1 - mf_f1,
        })

    label_comparison = pd.DataFrame(label_rows)

    print("\nTEST F1 BY PII LABEL")
    print("=" * 90)
    print(label_comparison.round(4).to_string(index=False))

    label_comparison.to_csv(LABEL_OUTPUT_PATH, index=False)
    print(f"\nLabel comparison saved to: {LABEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
