import argparse
import copy
import gzip
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from seqeval.metrics import classification_report as entity_report
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "data"
OUTPUT_DIR = ROOT / "bilstm_crf"
RESULTS_DIR = OUTPUT_DIR / "results"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODEL_DIR = OUTPUT_DIR / "models"
GLOVE_DIR = OUTPUT_DIR / "glove"

TRAIN_PATH = DATA_DIR / "train.jsonl.gz"
VAL_PATH = DATA_DIR / "val.jsonl.gz"
TEST_PATH = DATA_DIR / "test.jsonl.gz"
LABELS_PATH = DATA_DIR / "labels.json"

RESULTS_PATH = RESULTS_DIR / "results.json"
MODEL_PATH = MODEL_DIR / "bilstm_crf.pt"

# GloVe 6B, 100 dimensions (Wikipedia 2014 + Gigaword 5, 400k lowercase words).
# The official file is glove.6B.100d.txt inside glove.6B.zip (822 MB) from
# https://nlp.stanford.edu/projects/glove/. The default download below holds
# the same vectors in one 134 MB gzip file; --glove also accepts the .txt file.
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)
GLOVE_PATH = GLOVE_DIR / "glove-wiki-gigaword-100.gz"

SEED = 42
PAD = "<pad>"
UNK = "<unk>"

WORD_DIM = 100
CHAR_DIM = 30
CHAR_FILTERS = 50
MAX_WORD_CHARS = 30
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.5
CLIP = 5.0


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


def load_glove(path, wanted):
    """Read GloVe vectors, keeping only the words in `wanted`."""
    if not path.exists():
        print(f"Downloading GloVe to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GLOVE_URL, path)

    vectors = {}
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip().split(" ")

            # The gzip version starts with a header line: "400000 100".
            if len(parts) != WORD_DIM + 1:
                continue

            if parts[0] in wanted:
                vectors[parts[0]] = [float(value) for value in parts[1:]]

    return vectors


# --------------------------------------------------
# Vocabularies and encoding
# --------------------------------------------------

def build_vocabularies(train_rows, embeddings, glove_path):
    """Word vocabulary (lowercased, like GloVe) and character vocabulary
    (case kept, so the character layer sees capital letters)."""
    word_counts = Counter(
        token.lower() for row in train_rows for token in row["tokens"]
    )
    glove = load_glove(glove_path, set(word_counts)) if embeddings == "glove" else {}

    # Keep a word if it has a GloVe vector or appears at least twice.
    # Other words become <unk>, so the model also learns an <unk> vector
    # for words it has never seen.
    words = sorted(
        word for word, count in word_counts.items() if count >= 2 or word in glove
    )
    word_to_id = {word: index for index, word in enumerate([PAD, UNK] + words)}

    # Random start for words without a GloVe vector (Ma and Hovy, 2016).
    scale = (3.0 / WORD_DIM) ** 0.5
    vectors = torch.empty(len(word_to_id), WORD_DIM).uniform_(-scale, scale)
    vectors[0] = 0.0

    for word, index in word_to_id.items():
        if word in glove:
            vectors[index] = torch.tensor(glove[word])

    char_counts = Counter(
        char for row in train_rows for token in row["tokens"] for char in token
    )
    chars = sorted(char_counts)
    char_to_id = {char: index for index, char in enumerate([PAD, UNK] + chars)}

    stats = {
        "train_words_lowercased": len(word_counts),
        "word_vocabulary": len(word_to_id),
        "words_with_glove_vector": sum(1 for word in word_to_id if word in glove),
        "train_tokens_with_glove_vector": sum(
            count for word, count in word_counts.items() if word in glove
        ),
        "train_tokens": sum(word_counts.values()),
        "char_vocabulary": len(char_to_id),
    }

    return word_to_id, char_to_id, vectors, stats


def encode(rows, word_to_id, char_to_id, tag_to_id):
    """Turn every chunk into word ids, character ids and tag ids."""
    unk_word = word_to_id[UNK]
    unk_char = char_to_id[UNK]
    encoded = []

    for row in rows:
        words = [word_to_id.get(token.lower(), unk_word) for token in row["tokens"]]
        chars = [
            [char_to_id.get(char, unk_char) for char in token[:MAX_WORD_CHARS]]
            for token in row["tokens"]
        ]
        tags = [tag_to_id[tag] for tag in row["tags"]]
        encoded.append((words, chars, tags))

    return encoded


def make_batches(encoded, batch_size, shuffle, rng):
    """Group chunks of similar length into batches (less padding, faster)."""
    order = list(range(len(encoded)))

    if shuffle:
        rng.shuffle(order)

    pool_size = batch_size * 50
    batches = []

    for start in range(0, len(order), pool_size):
        pool = sorted(order[start:start + pool_size], key=lambda i: len(encoded[i][0]))
        batches.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))

    if shuffle:
        rng.shuffle(batches)

    return batches


def collate(encoded, indices, device):
    """Pad one batch into tensors."""
    items = [encoded[i] for i in indices]
    max_len = max(len(words) for words, _, _ in items)
    max_chars = max(len(chars) for _, word_chars, _ in items for chars in word_chars)

    words = torch.zeros(len(items), max_len, dtype=torch.long)
    chars = torch.zeros(len(items), max_len, max_chars, dtype=torch.long)
    tags = torch.zeros(len(items), max_len, dtype=torch.long)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)

    for b, (word_ids, char_ids, tag_ids) in enumerate(items):
        length = len(word_ids)
        words[b, :length] = torch.tensor(word_ids)
        tags[b, :length] = torch.tensor(tag_ids)
        mask[b, :length] = True

        for t, token_chars in enumerate(char_ids):
            chars[b, t, :len(token_chars)] = torch.tensor(token_chars)

    return words.to(device), chars.to(device), tags.to(device), mask.to(device)


# --------------------------------------------------
# Model
# --------------------------------------------------

class BiLSTMCRF(nn.Module):
    """Word embedding + character CNN -> 2-layer BiLSTM -> CRF."""

    def __init__(self, word_vectors, num_chars, num_tags, use_chars):
        super().__init__()
        self.use_chars = use_chars
        self.word_embedding = nn.Embedding.from_pretrained(
            word_vectors, freeze=False, padding_idx=0
        )
        input_dim = WORD_DIM

        if use_chars:
            # Reads each word letter by letter: capital letters, digits,
            # "@" and shapes such as IBANs that GloVe (lowercase) cannot see.
            self.char_embedding = nn.Embedding(num_chars, CHAR_DIM, padding_idx=0)
            self.char_cnn = nn.Conv1d(CHAR_DIM, CHAR_FILTERS, kernel_size=3, padding=1)
            input_dim += CHAR_FILTERS

        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(
            input_dim,
            HIDDEN,
            num_layers=LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.hidden_to_tag = nn.Linear(2 * HIDDEN, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def emissions(self, words, chars, mask):
        """Score of every tag for every word, before the CRF."""
        inputs = self.word_embedding(words)

        if self.use_chars:
            batch, length, max_chars = chars.shape
            flat = chars.view(batch * length, max_chars)
            char_vectors = self.char_cnn(self.char_embedding(flat).transpose(1, 2))

            # Max over the real letters of each word (padding is ignored).
            char_vectors = char_vectors.masked_fill((flat == 0).unsqueeze(1), -1e4)
            char_vectors = char_vectors.max(dim=2).values
            char_vectors = char_vectors * mask.view(-1, 1)
            inputs = torch.cat([inputs, char_vectors.view(batch, length, -1)], dim=2)

        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            self.dropout(inputs), lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=words.size(1))

        return self.hidden_to_tag(self.dropout(output))

    def loss(self, words, chars, tags, mask):
        """Negative log-likelihood of the gold tag sequences."""
        return -self.crf(self.emissions(words, chars, mask), tags, mask=mask, reduction="mean")

    def decode(self, words, chars, mask):
        """Best tag sequence for every chunk (Viterbi)."""
        return self.crf.decode(self.emissions(words, chars, mask), mask=mask)


# --------------------------------------------------
# Training and prediction
# --------------------------------------------------

def predict(model, encoded, id_to_tag, batch_size, device):
    """Predict tag sequences in the original chunk order."""
    model.eval()
    predictions = [None] * len(encoded)

    with torch.no_grad():
        for indices in make_batches(encoded, batch_size, shuffle=False, rng=None):
            words, chars, _, mask = collate(encoded, indices, device)

            for index, sequence in zip(indices, model.decode(words, chars, mask)):
                predictions[index] = [id_to_tag[tag] for tag in sequence]

    return predictions


def entity_scores(rows, predictions):
    """Entity-level micro precision, recall and F1 (seqeval, strict IOB2)."""
    report = entity_report(
        [row["tags"] for row in rows],
        predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    return {key: float(value) for key, value in report["micro avg"].items()}


def train(model, train_encoded, val_rows, val_encoded, id_to_tag, args, device):
    """Train with Adam; keep the epoch with the best validation entity F1
    and stop when it has not improved for `patience` epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = random.Random(SEED)

    history = []
    best_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        batches = make_batches(train_encoded, args.batch_size, shuffle=True, rng=rng)

        for indices in batches:
            words, chars, tags, mask = collate(train_encoded, indices, device)
            optimizer.zero_grad()
            loss = model.loss(words, chars, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()

        train_seconds = time.time() - start
        val_predictions = predict(model, val_encoded, id_to_tag, args.batch_size, device)
        scores = entity_scores(val_rows, val_predictions)

        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / len(batches), 4),
            "val_precision": scores["precision"],
            "val_recall": scores["recall"],
            "val_f1": scores["f1-score"],
            "seconds": round(time.time() - start, 1),
        })
        print(
            f"Epoch {epoch}: loss {total_loss / len(batches):.4f}, "
            f"val entity F1 {scores['f1-score']:.4f} "
            f"(train {train_seconds:.0f} s, total {time.time() - start:.0f} s)",
            flush=True,
        )

        if scores["f1-score"] > best_f1:
            best_f1 = scores["f1-score"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    model.load_state_dict(best_state)

    return history, best_epoch


def save_predictions(rows, predictions, path):
    """Write predictions in the shared format: {"id": ..., "pred": [...]}."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for row, tags in zip(rows, predictions):
            line = {"id": row["id"], "pred": tags}
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


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
    print(classification_report(true_tags, predicted_tags, labels=labels, zero_division=0))

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BiLSTM-CRF NER for PII detection")
    parser.add_argument("--embeddings", choices=["glove", "random"], default="glove",
                        help="start word vectors from GloVe or at random")
    parser.add_argument("--no-chars", action="store_true",
                        help="turn off the character layer")
    parser.add_argument("--glove", type=Path, default=GLOVE_PATH,
                        help="GloVe 100d file (.txt or .gz); downloaded if missing")
    parser.add_argument("--epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many epochs without a better val F1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="quick test on N random training chunks (0 = all)")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_chars = not args.no_chars

    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_rows = load_jsonl_gz(TRAIN_PATH)
    val_rows = load_jsonl_gz(VAL_PATH)
    test_rows = load_jsonl_gz(TEST_PATH)
    labels = load_labels(LABELS_PATH)

    if args.train_limit:
        train_rows = random.Random(SEED).sample(train_rows, args.train_limit)

    print(f"Train chunks: {len(train_rows)}")
    print(f"Validation chunks: {len(val_rows)}")
    print(f"Test chunks: {len(test_rows)}")
    print(f"Device: {device}, embeddings: {args.embeddings}, character layer: {use_chars}")

    # 1. Vocabularies and word vectors (from the training split only).
    word_to_id, char_to_id, vectors, vocab_stats = build_vocabularies(
        train_rows, args.embeddings, args.glove
    )
    print(json.dumps(vocab_stats, indent=2))

    tag_to_id = {tag: index for index, tag in enumerate(labels)}
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}

    train_encoded = encode(train_rows, word_to_id, char_to_id, tag_to_id)
    val_encoded = encode(val_rows, word_to_id, char_to_id, tag_to_id)
    test_encoded = encode(test_rows, word_to_id, char_to_id, tag_to_id)

    # 2. Train; the best epoch is chosen on the validation split.
    model = BiLSTMCRF(vectors, len(char_to_id), len(labels), use_chars).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameters:,}")

    start = time.time()
    history, best_epoch = train(
        model, train_encoded, val_rows, val_encoded, id_to_tag, args, device
    )
    train_seconds = time.time() - start
    print(f"Training time: {train_seconds:.0f} s, best epoch: {best_epoch}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "word_to_id": word_to_id,
            "char_to_id": char_to_id,
            "labels": labels,
            "embeddings": args.embeddings,
            "use_chars": use_chars,
        },
        MODEL_PATH,
    )

    # 3. Predict validation and test (test only once, at the end).
    start = time.time()
    val_predictions = predict(model, val_encoded, id_to_tag, args.batch_size, device)
    test_predictions = predict(model, test_encoded, id_to_tag, args.batch_size, device)
    predict_seconds = time.time() - start

    save_predictions(val_rows, val_predictions, PREDICTIONS_DIR / "val.jsonl")
    save_predictions(test_rows, test_predictions, PREDICTIONS_DIR / "test.jsonl")

    val_results = evaluate(val_rows, val_predictions, labels, "Validation")
    test_results = evaluate(test_rows, test_predictions, labels, "Test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "BiLSTM-CRF",
        "library": "PyTorch + pytorch-crf",
        "prediction_unit": "token",
        "settings": {
            "embeddings": args.embeddings,
            "character_layer": use_chars,
            "word_dim": WORD_DIM,
            "char_dim": CHAR_DIM,
            "char_filters": CHAR_FILTERS if use_chars else 0,
            "max_word_chars": MAX_WORD_CHARS,
            "lstm_hidden": HIDDEN,
            "lstm_layers": LAYERS,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "gradient_clip": CLIP,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "seed": SEED,
        },
        "device": str(device),
        "parameters": parameters,
        "vocabulary": vocab_stats,
        "train_chunks": len(train_rows),
        "train_seconds": round(train_seconds, 1),
        "predict_seconds_val_and_test": round(predict_seconds, 1),
        "best_epoch": best_epoch,
        "val_history": history,
        "validation": val_results,
        "test": test_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
