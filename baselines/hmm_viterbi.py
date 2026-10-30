
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
RESULTS_DIR = ROOT / "baselines" / "results"
PREDICTIONS_DIR = ROOT / "baselines" / "predictions"

TRAIN_PATH = DATA_DIR / "train.jsonl.gz"
VAL_PATH = DATA_DIR / "val.jsonl.gz"
TEST_PATH = DATA_DIR / "test.jsonl.gz"
LABELS_PATH = DATA_DIR / "labels.json"

RESULTS_PATH = RESULTS_DIR / "hmm_viterbi_results.json"

ALPHA = 0.1


# --------------------------------------------------
# Data loading and prediction export
# --------------------------------------------------

def load_jsonl_gz(path):
    """Load records from a gzipped JSONL file."""
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def save_predictions(rows, predictions, filename):
    """Save predictions using the shared JSONL format."""
    output_path = PREDICTIONS_DIR / filename

    if len(rows) != len(predictions):
        raise ValueError("Prediction sequence count mismatch.")

    with output_path.open("w", encoding="utf-8") as file:
        for row, predicted_tags in zip(rows, predictions):
            if len(row["tokens"]) != len(predicted_tags):
                raise ValueError(
                    f"Prediction length mismatch: {row.get('id')}"
                )

            record = {
                "id": row["id"],
                "doc_id": row.get("doc_id"),
                "pred": predicted_tags,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Predictions saved to: {output_path}")


# --------------------------------------------------
# HMM with Viterbi decoding
# --------------------------------------------------

class HMMTagger:
    """First-order HMM with Laplace smoothing and BIO constraints."""

    def __init__(self, labels, alpha=ALPHA):
        self.labels = labels
        self.alpha = alpha

        self.tag_to_id = {
            tag: index for index, tag in enumerate(labels)
        }

        self.start_log_probs = None
        self.transition_log_probs = None
        self.emission_log_probs = None

        self.vocabulary = {}
        self.unknown_token_id = None

    @staticmethod
    def is_valid_bio_transition(previous_tag, current_tag):
        """Check whether a BIO transition is valid."""
        if current_tag.startswith("I-"):
            entity_type = current_tag[2:]
            return previous_tag in {
                f"B-{entity_type}",
                f"I-{entity_type}",
            }

        return True

    def fit(self, sequences):
        """Estimate HMM start, transition and emission probabilities."""
        num_tags = len(self.labels)

        start_counts = np.zeros(num_tags, dtype=np.float64)
        transition_counts = np.zeros(
            (num_tags, num_tags), dtype=np.float64
        )

        emission_counts = defaultdict(Counter)
        tag_counts = np.zeros(num_tags, dtype=np.float64)
        vocabulary = set()

        for sequence in sequences:
            tokens = sequence["tokens"]
            tags = sequence["tags"]

            if not tokens:
                continue

            if len(tokens) != len(tags):
                raise ValueError(
                    f"Token/tag length mismatch: {sequence.get('id')}"
                )

            normalized_tokens = [token.lower() for token in tokens]
            tag_ids = [self.tag_to_id[tag] for tag in tags]

            start_counts[tag_ids[0]] += 1

            for token, tag_id in zip(normalized_tokens, tag_ids):
                emission_counts[tag_id][token] += 1
                tag_counts[tag_id] += 1
                vocabulary.add(token)

            for previous, current in zip(tag_ids, tag_ids[1:]):
                transition_counts[previous, current] += 1

        self.vocabulary = {
            token: index
            for index, token in enumerate(sorted(vocabulary))
        }

        self.unknown_token_id = len(self.vocabulary)
        vocab_size = len(self.vocabulary) + 1

        # Start probabilities with smoothing.
        start_probs = (
            start_counts + self.alpha
        ) / (
            start_counts.sum() + self.alpha * num_tags
        )

        self.start_log_probs = np.log(start_probs)

        # Transition probabilities with smoothing.
        transition_probs = (
            transition_counts + self.alpha
        ) / (
            transition_counts.sum(axis=1, keepdims=True)
            + self.alpha * num_tags
        )

        self.transition_log_probs = np.log(transition_probs)

        # Disallow transitions that violate BIO rules.
        for previous_id, previous_tag in enumerate(self.labels):
            for current_id, current_tag in enumerate(self.labels):
                if not self.is_valid_bio_transition(
                    previous_tag, current_tag
                ):
                    self.transition_log_probs[
                        previous_id, current_id
                    ] = -np.inf

        # Disallow starting a sequence with an inside-entity tag.
        for tag_id, tag in enumerate(self.labels):
            if tag.startswith("I-"):
                self.start_log_probs[tag_id] = -np.inf

        # Emission log probabilities.
        self.emission_log_probs = np.full(
            (num_tags, vocab_size),
            -np.inf,
            dtype=np.float64,
        )

        for tag_id in range(num_tags):
            denominator = (
                tag_counts[tag_id] + self.alpha * vocab_size
            )

            if denominator == 0:
                continue

            for token, count in emission_counts[tag_id].items():
                token_id = self.vocabulary[token]
                probability = (
                    count + self.alpha
                ) / denominator

                self.emission_log_probs[
                    tag_id, token_id
                ] = np.log(probability)

            unknown_probability = self.alpha / denominator

            self.emission_log_probs[
                tag_id, self.unknown_token_id
            ] = np.log(unknown_probability)

        print(f"Training sequences: {len(sequences)}")
        print(f"Number of tags: {num_tags}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print("HMM training completed.")

    def predict(self, tokens):
        """Decode the most likely tag sequence using Viterbi."""
        if not tokens:
            return []

        if self.start_log_probs is None:
            raise RuntimeError("Train the model before prediction.")

        num_tags = len(self.labels)
        num_tokens = len(tokens)

        normalized_tokens = [token.lower() for token in tokens]

        token_ids = [
            self.vocabulary.get(token, self.unknown_token_id)
            for token in normalized_tokens
        ]

        emissions = self.emission_log_probs[:, token_ids].T

        scores = np.full(
            (num_tokens, num_tags),
            -np.inf,
            dtype=np.float64,
        )

        backpointers = np.zeros(
            (num_tokens, num_tags),
            dtype=np.int32,
        )

        scores[0] = self.start_log_probs + emissions[0]

        for t in range(1, num_tokens):
            candidates = (
                scores[t - 1][:, None]
                + self.transition_log_probs
            )

            backpointers[t] = np.argmax(candidates, axis=0)

            scores[t] = (
                np.max(candidates, axis=0) + emissions[t]
            )

        best_path = np.zeros(num_tokens, dtype=np.int32)
        best_path[-1] = int(np.argmax(scores[-1]))

        for t in range(num_tokens - 2, -1, -1):
            best_path[t] = backpointers[
                t + 1, best_path[t + 1]
            ]

        return [self.labels[tag_id] for tag_id in best_path]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, sequences, split_name):
    """Evaluate token-level classification performance."""
    true_tags = []
    predicted_tags = []
    all_predictions = []

    for sequence in sequences:
        tokens = sequence["tokens"]
        tags = sequence["tags"]

        predictions = model.predict(tokens)

        if len(predictions) != len(tags):
            raise ValueError(
                f"Prediction length mismatch: {sequence.get('id')}"
            )

        true_tags.extend(tags)
        predicted_tags.extend(predictions)
        all_predictions.append(predictions)

    report = classification_report(
        true_tags,
        predicted_tags,
        labels=model.labels,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{split_name} results")
    print(
        classification_report(
            true_tags,
            predicted_tags,
            labels=model.labels,
            zero_division=0,
        )
    )

    results = {
        "split": split_name,
        "chunks": len(sequences),
        "tokens": len(true_tags),
        "classification_report": report,
    }

    return results, all_predictions


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_data = load_jsonl_gz(TRAIN_PATH)
    val_data = load_jsonl_gz(VAL_PATH)
    test_data = load_jsonl_gz(TEST_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    # Support either a JSON list or a dictionary containing labels.
    if isinstance(label_data, list):
        labels = label_data
    elif isinstance(label_data, dict) and "labels" in label_data:
        labels = label_data["labels"]
    else:
        raise ValueError(
            "Unexpected labels.json format. "
            "Expected a list or a dictionary with a 'labels' field."
        )

    print(f"Train chunks: {len(train_data)}")
    print(f"Validation chunks: {len(val_data)}")
    print(f"Test chunks: {len(test_data)}")

    model = HMMTagger(labels=labels)
    model.fit(train_data)

    val_results, val_predictions = evaluate(
        model, val_data, "Validation"
    )

    test_results, test_predictions = evaluate(
        model, test_data, "Test"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    save_predictions(
        val_data,
        val_predictions,
        "hmm_viterbi_validation.jsonl",
    )

    save_predictions(
        test_data,
        test_predictions,
        "hmm_viterbi_test.jsonl",
    )

    output = {
        "model": "HMM + Viterbi",
        "prediction_unit": "token",
        "prediction_format": "JSONL with id, doc_id and pred fields",
        "smoothing_alpha": ALPHA,
        "bio_transition_constraints": True,
        "train_chunks": len(train_data),
        "validation": val_results,
        "test": test_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
