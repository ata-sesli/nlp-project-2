
import gzip
import json
import re
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
MIN_WORD_FREQUENCY = 2

SHAPE_PREFIX = "__SHAPE_"
UNKNOWN_SHAPE = "__SHAPE_UNKNOWN__"


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
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

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
# Word-shape classification
# --------------------------------------------------

def get_word_shape(token):
    """
    Map a token to a generalized word-shape class.

    Examples:
        Serhat                  -> TITLECASE
        ISTANBUL                -> UPPERCASE
        istanbul                -> LOWERCASE
        123456                  -> NUMBER
        serhat@example.com      -> EMAIL
        TR120006200000000000001 -> IBAN
        +905321234567           -> PHONE
    """
    if not token:
        return UNKNOWN_SHAPE

    stripped = token.strip()

    if not stripped:
        return UNKNOWN_SHAPE

    # Email addresses.
    if re.fullmatch(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
        stripped,
    ):
        return SHAPE_PREFIX + "EMAIL__"

    # IBAN-like strings.
    compact = re.sub(r"\s+", "", stripped).upper()

    if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return SHAPE_PREFIX + "IBAN__"

    # Phone-like strings.
    digits = re.sub(r"\D", "", stripped)

    if (
        10 <= len(digits) <= 15
        and re.fullmatch(r"[+()\-\s./\d]+", stripped)
    ):
        return SHAPE_PREFIX + "PHONE__"

    # Numeric values, including decimals and signed numbers.
    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", stripped):
        return SHAPE_PREFIX + "NUMBER__"

    # URLs.
    if re.fullmatch(
        r"(?:https?://)?(?:www\.)?[A-Za-z0-9-]+"
        r"(?:\.[A-Za-z0-9-]+)+(?:/[^\s]*)?",
        stripped,
        flags=re.IGNORECASE,
    ):
        return SHAPE_PREFIX + "URL__"

    # Uppercase tokens.
    if stripped.isupper() and any(char.isalpha() for char in stripped):
        return SHAPE_PREFIX + "UPPERCASE__"

    # Lowercase tokens.
    if stripped.islower() and any(char.isalpha() for char in stripped):
        return SHAPE_PREFIX + "LOWERCASE__"

    # Titlecase tokens.
    if (
        len(stripped) > 1
        and stripped[0].isupper()
        and stripped[1:].islower()
    ):
        return SHAPE_PREFIX + "TITLECASE__"

    # Character-level shape for mixed tokens.
    shape = []

    for char in stripped:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append(char)

    # Limit the number of distinct classes for long tokens.
    if len(shape) > 12:
        shape = shape[:6] + ["*"] + shape[-3:]

    return SHAPE_PREFIX + "".join(shape) + "__"


# --------------------------------------------------
# HMM with Viterbi decoding
# --------------------------------------------------

class HMMTagger:
    """First-order HMM with add-k smoothing, word shapes and BIO constraints."""

    def __init__(
        self,
        labels,
        alpha=ALPHA,
        min_word_frequency=MIN_WORD_FREQUENCY,
    ):
        self.labels = labels
        self.alpha = alpha
        self.min_word_frequency = min_word_frequency

        self.tag_to_id = {
            tag: index for index, tag in enumerate(labels)
        }

        self.start_log_probs = None
        self.transition_log_probs = None
        self.emission_log_probs = None

        self.vocabulary = {}
        self.unknown_token_id = None

        self.word_frequencies = Counter()
        self.shape_classes = set()

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

    def get_feature(self, token):
        """
        Keep frequent words as lexical features.
        Map rare and unseen words to their word-shape classes.
        """
        normalized = token.lower()

        if (
            self.word_frequencies.get(normalized, 0)
            >= self.min_word_frequency
        ):
            return normalized

        return get_word_shape(token)

    def fit(self, sequences):
        """Estimate HMM start, transition and emission probabilities."""
        num_tags = len(self.labels)

        start_counts = np.zeros(
            num_tags,
            dtype=np.float64,
        )

        transition_counts = np.zeros(
            (num_tags, num_tags),
            dtype=np.float64,
        )

        emission_counts = defaultdict(Counter)

        tag_counts = np.zeros(
            num_tags,
            dtype=np.float64,
        )

        # First pass: calculate word frequencies.
        for sequence in sequences:
            tokens = sequence["tokens"]
            tags = sequence["tags"]

            if len(tokens) != len(tags):
                raise ValueError(
                    f"Token/tag length mismatch: {sequence.get('id')}"
                )

            for token in tokens:
                self.word_frequencies[token.lower()] += 1

        # Second pass: collect HMM statistics.
        for sequence in sequences:
            tokens = sequence["tokens"]
            tags = sequence["tags"]

            if not tokens:
                continue

            tag_ids = []

            for tag in tags:
                if tag not in self.tag_to_id:
                    raise ValueError(f"Unknown label: {tag}")

                tag_ids.append(self.tag_to_id[tag])

            start_counts[tag_ids[0]] += 1

            for token, tag_id in zip(tokens, tag_ids):
                feature = self.get_feature(token)

                emission_counts[tag_id][feature] += 1
                tag_counts[tag_id] += 1

                if feature.startswith(SHAPE_PREFIX):
                    self.shape_classes.add(feature)

            for previous, current in zip(tag_ids, tag_ids[1:]):
                transition_counts[previous, current] += 1

        # Build the feature vocabulary.
        vocabulary = set()

        for tag_counter in emission_counts.values():
            vocabulary.update(tag_counter.keys())

        vocabulary.add(UNKNOWN_SHAPE)

        self.vocabulary = {
            feature: index
            for index, feature in enumerate(sorted(vocabulary))
        }

        self.unknown_token_id = self.vocabulary[UNKNOWN_SHAPE]
        vocab_size = len(self.vocabulary)

        # --------------------------------------------------
        # Start probabilities
        # --------------------------------------------------

        # BIO sequences cannot start with an I- tag.
        valid_start_mask = np.array(
            [
                not tag.startswith("I-")
                for tag in self.labels
            ],
            dtype=bool,
        )

        start_counts[~valid_start_mask] = 0.0

        start_denominator = (
            start_counts.sum()
            + self.alpha * valid_start_mask.sum()
        )

        start_probs = np.zeros(
            num_tags,
            dtype=np.float64,
        )

        start_probs[valid_start_mask] = (
            start_counts[valid_start_mask] + self.alpha
        ) / start_denominator

        self.start_log_probs = np.full(
            num_tags,
            -np.inf,
            dtype=np.float64,
        )

        self.start_log_probs[valid_start_mask] = np.log(
            start_probs[valid_start_mask]
        )

        # --------------------------------------------------
        # Transition probabilities
        # --------------------------------------------------

        self.transition_log_probs = np.full(
            (num_tags, num_tags),
            -np.inf,
            dtype=np.float64,
        )

        for previous_id, previous_tag in enumerate(self.labels):
            valid_current_ids = [
                current_id
                for current_id, current_tag in enumerate(self.labels)
                if self.is_valid_bio_transition(
                    previous_tag,
                    current_tag,
                )
            ]

            denominator = (
                transition_counts[
                    previous_id,
                    valid_current_ids,
                ].sum()
                + self.alpha * len(valid_current_ids)
            )

            probabilities = (
                transition_counts[
                    previous_id,
                    valid_current_ids,
                ]
                + self.alpha
            ) / denominator

            self.transition_log_probs[
                previous_id,
                valid_current_ids,
            ] = np.log(probabilities)

        # --------------------------------------------------
        # Emission probabilities
        # --------------------------------------------------

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

            # Add-k smoothing:
            # Every tag-feature pair receives a non-zero probability.
            default_probability = self.alpha / denominator

            self.emission_log_probs[tag_id, :] = np.log(
                default_probability
            )

            # Replace smoothed defaults with observed emission counts.
            for feature, count in emission_counts[tag_id].items():
                feature_id = self.vocabulary[feature]

                probability = (
                    count + self.alpha
                ) / denominator

                self.emission_log_probs[
                    tag_id,
                    feature_id,
                ] = np.log(probability)

        print(f"Training sequences: {len(sequences)}")
        print(f"Number of tags: {num_tags}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Word-shape classes: {len(self.shape_classes)}")
        print(f"Minimum word frequency: {self.min_word_frequency}")
        print("HMM training completed.")

    def predict(self, tokens):
        """Decode the most likely tag sequence using Viterbi."""
        if not tokens:
            return []

        if self.start_log_probs is None:
            raise RuntimeError("Train the model before prediction.")

        num_tags = len(self.labels)
        num_tokens = len(tokens)

        token_ids = []

        for token in tokens:
            feature = self.get_feature(token)

            token_id = self.vocabulary.get(
                feature,
                self.unknown_token_id,
            )

            token_ids.append(token_id)

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

        # Initialization.
        scores[0] = self.start_log_probs + emissions[0]

        if np.all(np.isneginf(scores[0])):
            raise RuntimeError(
                "Viterbi decoding failed at the first token."
            )

        # Recursion.
        for t in range(1, num_tokens):
            candidates = (
                scores[t - 1][:, None]
                + self.transition_log_probs
            )

            backpointers[t] = np.argmax(
                candidates,
                axis=0,
            )

            scores[t] = (
                np.max(candidates, axis=0)
                + emissions[t]
            )

        if np.all(np.isneginf(scores[-1])):
            raise RuntimeError(
                "Viterbi decoding failed: no valid tag sequence."
            )

        # Termination and backtracking.
        best_path = np.zeros(
            num_tokens,
            dtype=np.int32,
        )

        best_path[-1] = int(np.argmax(scores[-1]))

        for t in range(num_tokens - 2, -1, -1):
            best_path[t] = backpointers[
                t + 1,
                best_path[t + 1],
            ]

        return [
            self.labels[tag_id]
            for tag_id in best_path
        ]


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
    for path in [
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        LABELS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train_data = load_jsonl_gz(TRAIN_PATH)
    val_data = load_jsonl_gz(VAL_PATH)
    test_data = load_jsonl_gz(TEST_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        label_data = json.load(file)

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = HMMTagger(labels=labels)
    model.fit(train_data)

    val_results, val_predictions = evaluate(
        model,
        val_data,
        "Validation",
    )

    test_results, test_predictions = evaluate(
        model,
        test_data,
        "Test",
    )

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
        "minimum_word_frequency": MIN_WORD_FREQUENCY,
        "word_shape_features": True,
        "word_shape_classes": sorted(model.shape_classes),
        "bio_transition_constraints": True,
        "train_chunks": len(train_data),
        "validation": val_results,
        "test": test_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
