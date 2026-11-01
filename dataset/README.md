
cat > dataset/README.md <<'EOF'
# Dataset Preparation

This directory contains the dataset preparation pipeline for the Personally Identifiable Information (PII) detection task.

The preparation pipeline loads the source dataset, filters and validates documents, converts character-level PII annotations into BIO token labels, creates document-level train/validation/test splits, and divides documents into smaller chunks.

## 1. Source Dataset

The project uses the following source dataset:

- **Dataset:** `gretelai/synthetic_pii_finance_multilingual`
- **Platform:** Hugging Face Datasets
- **Language:** English
- **Content:** Synthetic text containing financial and personal information
- **License:** Apache License 2.0 (Apache-2.0)

The dataset is downloaded through the Hugging Face `datasets` library. The preparation script loads it automatically.

For more information, see the [Hugging Face dataset card](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual).

## 2. Data Filtering and Validation

Raw documents are not used directly for model training. They first pass through several filtering and validation steps.

The pipeline applies the following checks:

1. **Language filtering:** Only English documents are accepted.
2. **Text validation:** Empty or invalid text fields are rejected.
3. **Span validation:** PII annotation boundaries and entity labels are checked.
4. **Overlap validation:** Documents containing overlapping annotation spans are rejected.
5. **Duplicate detection:** Documents with duplicate text are removed.
6. **Minimum length:** Documents containing fewer than 20 words are rejected.
7. **PII validation:** Documents without PII annotations are rejected.
8. **Token-label consistency:** Documents are rejected when conflicting labels would be assigned to the same token.

A span is considered invalid if its start position is negative, its end position is not greater than its start position, its boundaries exceed the text length, or its entity label is unsupported.

Filtering statistics, including the number of documents rejected for each reason, are recorded in `dataset/data/REPORT.md`.

## 3. Data Filtering Statistics

The following table shows the source document counts and the number of documents accepted during filtering:

| Source split | Raw documents | Accepted documents |
|---|---:|---:|
| Train | 50,346 | 19,746 |
| Test | 5,594 | 2,258 |

The accepted training count is reported before the subsequent removal of documents duplicated in the test split. One training document matching the test split is removed during that additional check.

Documents may be rejected because they are not in English, are shorter than the minimum length, contain invalid or overlapping spans, have no PII annotations, contain duplicate text, or produce conflicting token labels.

For the complete filtering statistics, see `dataset/data/REPORT.md`.

## 4. Tokenization and BIO Labeling

The pipeline initially identifies tokens using whitespace-based tokenization (`\S+`). Opening parentheses and square brackets, along with selected trailing punctuation marks, are separated into individual tokens.

Character offsets are preserved during tokenization so that the original PII spans can be aligned with the resulting tokens.

Character-level PII annotations are converted into the BIO (Beginning, Inside, Outside) tagging scheme by comparing annotation spans with token character offsets.

The BIO scheme uses three types of labels:

- **B (Beginning):** Marks the first token of an entity.
- **I (Inside):** Marks subsequent tokens belonging to the same entity.
- **O (Outside):** Marks tokens that do not belong to a PII entity.

Example:

| Token | BIO label |
|---|---|
| John | B-NAME |
| Smith | I-NAME |
| works | O |
| at | O |
| Acme | B-COMPANY |

### Span-to-token alignment

Some character-level PII spans do not align exactly with token boundaries.

These spans are counted as `unaligned_spans`, but they are not automatically discarded. Instead, every token whose character range overlaps the annotation span receives the corresponding BIO label.

This approach helps preserve PII information when annotation boundaries do not match token boundaries exactly.

If assigning a span would produce conflicting labels for the same token, the document is rejected.

## 5. Entity Categories and Labels

The dataset uses eight PII entity categories:

| Entity category | Source annotation labels |
|---|---|
| NAME | `name`, `first_name`, `last_name`, `user_name` |
| COMPANY | `company` |
| ADDRESS | `street_address`, `local_latlng` |
| DATE | `date`, `date_of_birth`, `time`, `date_time` |
| EMAIL | `email` |
| PHONE | `phone_number` |
| ACCOUNT | `iban`, `bban`, `swift_bic_code`, `credit_card_number`, `credit_card_security_code`, `account_pin`, `bank_routing_number` |
| ID | `customer_id`, `employee_id`, `ssn`, `driver_license_number`, `passport_number`, `ipv4`, `ipv6`, `api_key`, `password` |

Each entity category has two BIO labels:

- One `B-` label for the beginning of an entity.
- One `I-` label for the continuation of an entity.

The `O` label represents tokens outside PII entities.

The resulting label set contains **17 labels**:

- 1 outside label (`O`)
- 8 beginning labels (`B-`)
- 8 inside labels (`I-`)

The complete label list is stored in `dataset/data/labels.json`.

## 6. Train, Validation, and Test Splits

Dataset splitting is performed at the document level before chunking. This prevents chunks originating from the same document from being distributed across different splits.

The splitting configuration is:

- **Random seed:** 42
- **Validation ratio:** Approximately 10% of the remaining training documents
- **Test split:** Preserved from the source dataset's test split
- **Duplicate prevention:** Training documents matching test documents are removed

The test split is processed first. Training documents with text identical to a test document are then removed.

The remaining training documents are shuffled using random seed 42. Approximately 10% are assigned to validation, while the rest are used for training.

Splitting before chunking helps prevent document-level data leakage between the training, validation, and test sets.

## 7. Document Chunking

Documents are divided into smaller chunks to control model input length.

The chunking configuration and behavior are:

- **Target chunk length:** 200 tokens
- Sentence and line boundaries are preferred when selecting chunk boundaries.
- Continuing `I-` labels are kept together whenever possible to avoid splitting an entity across chunks.
- A chunk may exceed 200 tokens when necessary to preserve the integrity of a long entity.

Therefore, 200 tokens is a target length rather than a strict maximum.

The generated dataset contains the following number of chunks:

| Split | Number of chunks |
|---|---:|
| Train | 30,545 |
| Validation | 3,389 |
| Test | 3,903 |
| **Total** | **37,837** |

## 8. Generating the Dataset

Run the following commands from the project root directory.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the dataset preparation script:

```bash
python dataset/prepare_dataset.py
```

The script downloads the source dataset, applies the filtering and annotation pipeline, creates the data splits, and writes the generated files.

## 9. Generated Files

The preparation script creates the following files in `dataset/data/`:

| File | Description |
|---|---|
| `train.jsonl.gz` | Compressed training chunks |
| `val.jsonl.gz` | Compressed validation chunks |
| `test.jsonl.gz` | Compressed test chunks |
| `labels.json` | Complete BIO label list |
| `REPORT.md` | Filtering statistics and dataset preparation report |

The JSONL files contain tokenized text and their corresponding BIO labels. They are compressed using gzip to reduce storage requirements.

Each generated record contains the information required by the downstream sequence-labeling models, including its identifier, tokens, and tags.

---

For detailed filtering counts and dataset statistics, see [`data/REPORT.md`](data/REPORT.md).
EOF

To inspect the generated dataset, review split statistics, and examine sample records, run:

```bash
python dataset/inspect_dataset.py
```
