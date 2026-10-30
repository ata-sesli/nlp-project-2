import os
from huggingface_hub import HfApi, create_repo

# Hugging Face username and repository names
repo_id_svm = "elifsudayi/svm-ner"
repo_id_crf = "elifsudayi/crf-crf" 

api = HfApi()

def push_models():
    print("Uploading models to Hugging Face Hub...")
    
    # Upload SVM model file (e.g., svm.joblib or target directory)
    if os.path.exists("svm_crf/models/svm.joblib"):
        api.upload_file(
            path_or_fileobj="svm_crf/models/svm.joblib",
            path_in_repo="svm.joblib",
            repo_id=repo_id_svm,
            repo_type="model",
        )
        print("SVM model uploaded successfully!")

if __name__ == "__main__":
    push_models()