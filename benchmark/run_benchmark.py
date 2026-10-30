"""Score all six models with evaluate.py and write tables and charts.

Usage:
    python benchmark/run_benchmark.py

Writes to benchmark/results/:
    results.json        every metric, per split and model
    summary.csv         main table (entity P/R/F1, macro F1, token accuracy, missed PII)
    per_type_f1.csv     entity F1 per PII type
    error_types.csv     error counts per model
    results.md          the test tables in Markdown, for the README
    f1_by_model.png, per_type_f1.png, error_types.png   test split charts
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import ERROR_TYPES, ROOT, evaluate_file


RESULTS_DIR = ROOT / "benchmark" / "results"

# Prediction files of each model, in the order of the benchmark.
MODELS = {
    "Most Frequent": {
        "val": "baselines/predictions/most_frequent_validation.jsonl",
        "test": "baselines/predictions/most_frequent_test.jsonl",
    },
    "HMM + Viterbi": {
        "val": "baselines/predictions/hmm_viterbi_validation.jsonl",
        "test": "baselines/predictions/hmm_viterbi_test.jsonl",
    },
    "SVM": {
        "val": "svm_crf/predictions/svm_val.jsonl",
        "test": "svm_crf/predictions/svm_test.jsonl",
    },
    "CRF": {
        "val": "svm_crf/predictions/crf_val.jsonl",
        "test": "svm_crf/predictions/crf_test.jsonl",
    },
    "BiLSTM-CRF": {
        "val": "bilstm_crf/predictions/val.jsonl",
        "test": "bilstm_crf/predictions/test.jsonl",
    },
    "BERT": {
        "val": "bert/predictions/val.jsonl",
        "test": "bert/predictions/test.jsonl",
    },
}
SPLITS = ["val", "test"]
SUMMARY_COLUMNS = [
    "precision", "recall", "f1", "macro_f1", "token_accuracy", "missed_pii_rate",
]

# Chart colors (light surface): one hue for single-series bars, a light-to-dark
# ramp of that hue for the heatmap, and fixed categorical slots for error types.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
BAR_COLOR = "#2a78d6"
HEAT_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ERROR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def run():
    results = {split: {} for split in SPLITS}

    for split in SPLITS:
        for model, paths in MODELS.items():
            path = ROOT / paths[split]
            print(f"[{split}] {model}: {paths[split]} ...", flush=True)
            results[split][model] = evaluate_file(split, path)
            print(f"[{split}] {model}: entity F1 {results[split][model]['f1']:.4f}")

    return results


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(results):
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    types = sorted(next(iter(results["test"].values()))["per_type"])

    write_csv(
        RESULTS_DIR / "summary.csv",
        ["split", "model", *SUMMARY_COLUMNS],
        [
            [split, model, *(round(scores[c], 4) for c in SUMMARY_COLUMNS)]
            for split in SPLITS
            for model, scores in results[split].items()
        ],
    )
    write_csv(
        RESULTS_DIR / "per_type_f1.csv",
        ["split", "model", *types],
        [
            [split, model, *(round(scores["per_type"][t]["f1"], 4) for t in types)]
            for split in SPLITS
            for model, scores in results[split].items()
        ],
    )
    write_csv(
        RESULTS_DIR / "error_types.csv",
        ["split", "model", "correct", *ERROR_TYPES],
        [
            [split, model, scores["error_types"]["correct"],
             *(scores["error_types"][e] for e in ERROR_TYPES)]
            for split in SPLITS
            for model, scores in results[split].items()
        ],
    )

    test = results["test"]
    lines = [
        "| Model | Precision | Recall | Entity F1 | Macro F1 | Token acc. | Missed PII |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {model} | " + " | ".join(f"{s[c]:.3f}" for c in SUMMARY_COLUMNS) + " |"
        for model, s in test.items()
    ]
    lines += ["", "| Model | " + " | ".join(types) + " |", "|---|" + "---:|" * len(types)]
    lines += [
        f"| {model} | " + " | ".join(f"{s['per_type'][t]['f1']:.3f}" for t in types) + " |"
        for model, s in test.items()
    ]
    lines += ["", "| Model | Correct | " + " | ".join(ERROR_TYPES) + " |",
              "|---|" + "---:|" * (len(ERROR_TYPES) + 1)]
    lines += [
        f"| {model} | {s['error_types']['correct']} | "
        + " | ".join(str(s["error_types"][e]) for e in ERROR_TYPES) + " |"
        for model, s in test.items()
    ]
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)


def plot_f1(test):
    models = list(test)
    f1 = [test[m]["f1"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    bars = ax.bar(models, f1, color=BAR_COLOR, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in f1], color=TEXT, padding=3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    style(ax, "Test entity F1 (seqeval, strict IOB2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_by_model.png", dpi=150)
    plt.close(fig)


def plot_per_type(test):
    models = list(test)
    types = sorted(test[models[0]]["per_type"])
    grid = [[test[m]["per_type"][t]["f1"] for t in types] for m in models]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", HEAT_RAMP)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)), types)
    ax.set_yticks(range(len(models)), models)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color=TEXT if value < 0.6 else SURFACE, fontsize=9)
    style(ax, "Test entity F1 per PII type")
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "per_type_f1.png", dpi=150)
    plt.close(fig)


def plot_errors(test):
    models = list(test)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    left = [0] * len(models)
    for error, color in zip(ERROR_TYPES, ERROR_COLORS):
        counts = [test[m]["error_types"][error] for m in models]
        ax.barh(models, counts, left=left, color=color, label=error,
                edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, counts)]
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=len(ERROR_TYPES), loc="upper left",
              bbox_to_anchor=(0, -0.12), labelcolor=TEXT_MUTED)
    style(ax, "Test errors by type (number of entities)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_types.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run()
    write_tables(results)
    plot_f1(results["test"])
    plot_per_type(results["test"])
    plot_errors(results["test"])
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
