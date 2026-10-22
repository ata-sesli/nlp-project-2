from datasets import load_dataset

DATASET_NAME = "gretelai/synthetic_pii_finance_multilingual"

print("Veri seti indiriliyor...")

dataset = load_dataset(DATASET_NAME)

print("\nİndirme tamamlandı.")

for split_name, split_data in dataset.items():
    print(f"\nBölüm: {split_name}")
    print(f"Belge sayısı: {len(split_data)}")
    print(f"Sütunlar: {split_data.column_names}")

    if len(split_data) > 0:
        print("\nİlk örneğin:")
        print(split_data[0])
