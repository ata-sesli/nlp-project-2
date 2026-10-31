
import json
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual")

for split_name, split_data in dataset.items():
    print(f"\n{'=' * 50}")
    print(f"SPLIT: {split_name}")
    print(f"Toplam belge: {len(split_data)}")

    languages = Counter()
    labels = Counter()
    malformed_json = 0
    invalid_spans = 0
    short_docs = 0
    no_pii_docs = 0

    for row in split_data:
        text = row.get("generated_text") or ""
        languages[row.get("language", "MISSING")] += 1

        if len(text.split()) < 20:
            short_docs += 1

        try:
            spans = json.loads(row.get("pii_spans") or "[]")
            if not isinstance(spans, list):
                raise ValueError("pii_spans is not a list")
        except (json.JSONDecodeError, TypeError, ValueError):
            malformed_json += 1
            continue

        if not spans:
            no_pii_docs += 1

        for span in spans:
            label = span.get("label", "MISSING")
            labels[label] += 1

            start = span.get("start")
            end = span.get("end")

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(text)
            ):
                invalid_spans += 1

    print("\nDil dağılımı:")
    for language, count in languages.most_common():
        print(f"  {language}: {count}")

    print("\nPII etiketleri (span sayısı):")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")

    print(f"\nBozuk JSON belgesi: {malformed_json}")
    print(f"Geçersiz aralık sayısı: {invalid_spans}")
    print(f"20 kelimeden kısa belge: {short_docs}")
    print(f"PII içermeyen belge: {no_pii_docs}")
