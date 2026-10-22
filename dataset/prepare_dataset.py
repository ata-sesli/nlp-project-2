
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "gretelai/synthetic_pii_finance_multilingual"
OUTPUT_DIR = Path(__file__).parent / "data"
SEED = 42
MAX_TOKENS = 200

LABEL_MAP = {
    "name": "NAME",
    "first_name": "NAME",
    "last_name": "NAME",
    "user_name": "NAME",
    "company": "COMPANY",
    "street_address": "ADDRESS",
    "local_latlng": "ADDRESS",
    "date": "DATE",
    "date_of_birth": "DATE",
    "time": "DATE",
    "date_time": "DATE",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "iban": "ACCOUNT",
    "bban": "ACCOUNT",
    "swift_bic_code": "ACCOUNT",
    "credit_card_number": "ACCOUNT",
    "credit_card_security_code": "ACCOUNT",
    "account_pin": "ACCOUNT",
    "bank_routing_number": "ACCOUNT",
    "customer_id": "ID",
    "employee_id": "ID",
    "ssn": "ID",
    "driver_license_number": "ID",
    "passport_number": "ID",
    "ipv4": "ID",
    "ipv6": "ID",
    "api_key": "ID",
    "password": "ID",
}

ENTITY_TYPES = [
    "NAME", "COMPANY", "ADDRESS", "DATE",
    "EMAIL", "PHONE", "ACCOUNT", "ID",
]
LABELS = ["O"] + [
    prefix + "-" + entity
    for entity in ENTITY_TYPES
    for prefix in ("B", "I")
]


def parse_spans(raw_spans, text):
    """Parse and validate character-level PII spans."""
    try:
        spans = json.loads(raw_spans) if isinstance(raw_spans, str) else raw_spans
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(spans, list):
        return None

    cleaned = []
    for span in spans:
        if not isinstance(span, dict):
            return None

        start = span.get("start")
        end = span.get("end")
        label = span.get("label")

        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(label, str)
            or label not in LABEL_MAP
            or start < 0
            or end <= start
            or end > len(text)
        ):
            return None

        cleaned.append({
            "start": start,
            "end": end,
            "label": LABEL_MAP[label],
        })

    cleaned.sort(key=lambda x: (x["start"], x["end"]))

    for previous, current in zip(cleaned, cleaned[1:]):
        if current["start"] < previous["end"]:
            return None

    return cleaned


def tokenize_with_offsets(text):
    """Whitespace tokenization plus splitting boundary punctuation."""
    tokens = []

    for match in re.finditer(r"\S+", text):
        word = match.group()
        start = match.start()
        end = match.end()

        # Split opening brackets/parentheses from the beginning.
        while len(word) > 1 and word[0] in "([":
            tokens.append({
                "token": word[0],
                "start": start,
                "end": start + 1,
            })
            word = word[1:]
            start += 1

        # Split trailing punctuation without changing token internals.
        trailing = ""
        while len(word) > 1 and word[-1] in ",.;:)]":
            trailing = word[-1] + trailing
            word = word[:-1]
            end -= 1

        if word:
            tokens.append({
                "token": word,
                "start": start,
                "end": end,
            })

        for char in trailing:
            tokens.append({
                "token": char,
                "start": end,
                "end": end + 1,
            })
            end += 1

    return tokens


def make_bio_tags(tokens, spans):
    tags = ["O"] * len(tokens)
    unaligned = 0

    for span in spans:
        hits = [
            i for i, token in enumerate(tokens)
            if token["start"] < span["end"]
            and token["end"] > span["start"]
        ]

        if not hits:
            unaligned += 1
            continue

        first = tokens[hits[0]]
        last = tokens[hits[-1]]

        if first["start"] != span["start"] or last["end"] != span["end"]:
            unaligned += 1

        entity = span["label"]

        for position, token_index in enumerate(hits):
            if tags[token_index] != "O":
                return None, unaligned

            prefix = "B" if position == 0 else "I"
            tags[token_index] = f"{prefix}-{entity}"

    return tags, unaligned


def prepare_split(dataset, split_name, report):
    records = []
    seen_texts = set()

    report[split_name] = Counter()
    stats = report[split_name]
    stats["raw_documents"] = len(dataset)

    for row_index, row in enumerate(dataset):
        stats["checked"] += 1

        if str(row.get("language", "")).strip().lower() != "english":
            stats["non_english"] += 1
            continue

        text = row.get("generated_text")
        if not isinstance(text, str) or not text.strip():
            stats["empty_text"] += 1
            continue

        spans = parse_spans(row.get("pii_spans"), text)
        if spans is None:
            stats["invalid_spans"] += 1
            continue

        normalized_text = text.strip()
        if normalized_text in seen_texts:
            stats["duplicate_documents"] += 1
            continue

        if len(text.split()) < 20:
            stats["short_documents"] += 1
            continue

        if not spans:
            stats["documents_without_pii"] += 1
            continue

        tokens = tokenize_with_offsets(text)
        tags, unaligned = make_bio_tags(tokens, spans)

        if tags is None:
            stats["token_tag_conflicts"] += 1
            continue

        if not tokens:
            stats["empty_tokens"] += 1
            continue

        seen_texts.add(normalized_text)
        stats["kept_documents"] += 1
        stats["unaligned_spans"] += unaligned

        source_index = row.get("index", row_index)
        doc_id = f"{split_name}_{source_index}"

        records.append({
            "doc_id": doc_id,
            "document_type": row.get("document_type", "Unknown"),
            "text": text,
            "tokens": [token["token"] for token in tokens],
            "token_offsets": [
                (token["start"], token["end"]) for token in tokens
            ],
            "tags": tags,
        })

    return records, seen_texts


def chunk_document(record):
    tokens = record["tokens"]
    tags = record["tags"]
    offsets = record["token_offsets"]
    text = record["text"]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))

        if end < len(tokens):
            # Prefer a sentence or line boundary, without splitting entities.
            candidates = []
            for boundary in range(start + 1, end + 1):
                if boundary < len(tokens) and tags[boundary].startswith("I-"):
                    continue

                if boundary == len(tokens):
                    continue

                previous_end = offsets[boundary - 1][1]
                next_start = offsets[boundary][0]
                gap = text[previous_end:next_start]

                if "\n" in gap or tokens[boundary - 1].endswith("."):
                    candidates.append(boundary)

            if candidates:
                end = candidates[-1]
            else:
                while end > start and end < len(tags) and tags[end].startswith("I-"):
                    end -= 1

                # A single entity longer than MAX_TOKENS must remain intact.
                if end == start:
                    end = min(start + MAX_TOKENS, len(tokens))
                    while end < len(tags) and tags[end].startswith("I-"):
                        end += 1

        if end <= start:
            end = min(start + MAX_TOKENS, len(tokens))

        chunk_number = len(chunks) + 1
        chunks.append({
            "id": f"{record['doc_id']}_c{chunk_number}",
            "doc_id": record["doc_id"],
            "document_type": record["document_type"],
            "tokens": tokens[start:end],
            "tags": tags[start:end],
        })
        start = end

    return chunks


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Hugging Face veri seti yükleniyor...")
    dataset = load_dataset(DATASET_NAME)

    report = {}
    all_seen_texts = set()

    # Process test first so duplicate training documents are excluded.
    test_records, test_seen = prepare_split(
        dataset["test"], "test", report
    )
    all_seen_texts.update(test_seen)

    train_records, _ = prepare_split(
        dataset["train"], "train", report
    )

    # Enforce cross-split deduplication: test has priority.
    test_texts = {record["text"].strip() for record in test_records}
    before = len(train_records)
    train_records = [
        record for record in train_records
        if record["text"].strip() not in test_texts
    ]
    report["train"]["duplicates_of_test"] = before - len(train_records)

    # Split by document before chunking, preventing document leakage.
    rng = random.Random(SEED)
    rng.shuffle(train_records)
    val_count = max(1, round(len(train_records) * 0.10))
    val_records = train_records[:val_count]
    train_records = train_records[val_count:]

    split_records = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    split_chunks = {}
    for split_name, records in split_records.items():
        chunks = []
        for record in records:
            chunks.extend(chunk_document(record))
        split_chunks[split_name] = chunks

        write_jsonl_gz(
            OUTPUT_DIR / f"{split_name}.jsonl.gz",
            chunks,
        )
        print(
            f"{split_name}: {len(records)} belge, {len(chunks)} parça"
        )

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as file:
        json.dump(LABELS, file, ensure_ascii=False, indent=2)

    # Build a reproducible text report.
    lines = [
        "# Dataset Preparation Report",
        "",
        f"- Source: `{DATASET_NAME}`",
        "- Language: English",
        f"- Random seed: {SEED}",
        f"- Maximum chunk length: {MAX_TOKENS} tokens",
        "- Tagging scheme: BIO",
        "- Test source split preserved; duplicate training documents removed.",
        "",
        "## Filtering statistics",
        "",
    ]

    for split_name, stats in report.items():
        lines.extend([f"### {split_name}", ""])
        for key, value in stats.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(["## Output sizes", ""])
    for split_name, chunks in split_chunks.items():
        lines.append(f"- {split_name}: {len(chunks)} chunks")
    lines.extend(["", "## Entity counts by BIO label", ""])

    entity_counts = Counter()
    for chunks in split_chunks.values():
        for chunk in chunks:
            entity_counts.update(
                tag for tag in chunk["tags"] if tag.startswith("B-")
            )

    for entity in ENTITY_TYPES:
        lines.append(f"- {entity}: {entity_counts[f'B-{entity}']}")

    lines.extend(["", "## Document type distribution", ""])
    for split_name, records in split_records.items():
        lines.append(f"### {split_name}")
        type_counts = Counter(
            record["document_type"] for record in records
        )
        for document_type, count in sorted(type_counts.items()):
            lines.append(f"- {document_type}: {count}")
        lines.append("")

    with open(OUTPUT_DIR / "REPORT.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    # Print 20 random chunks for manual inspection.
    all_chunks = [
        chunk
        for chunks in split_chunks.values()
        for chunk in chunks
    ]
    sample_size = min(20, len(all_chunks))
    sample = random.Random(SEED).sample(all_chunks, sample_size)

    print("\n--- BIO inspection samples ---")
    for chunk in sample:
        print(f"\n{chunk['id']}")
        print("TOKENS:", chunk["tokens"])
        print("TAGS:  ", chunk["tags"])

    print("\nTamamlandı. Çıktı klasörü:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
