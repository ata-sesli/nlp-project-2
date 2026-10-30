from huggingface_hub import hf_hub_download
import joblib

# 1. Download and Load SVM Model
print("Downloading SVM model...")
svm_path = hf_hub_download(repo_id="elifsudayi/svm-ner", filename="svm.joblib")
svm_model = joblib.load(svm_path)
print("SVM model loaded successfully!")

# 2. Download and Load CRF Model
print("Downloading CRF model...")
crf_path = hf_hub_download(repo_id="elifsudayi/crf-ner", filename="crf.joblib") 
crf_model = joblib.load(crf_path)
print("CRF model loaded successfully!")

print("Both models were successfully prepared from Hugging Face!")