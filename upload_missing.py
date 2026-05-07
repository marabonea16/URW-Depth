from huggingface_hub import HfApi
import os, time

api = HfApi()
repo_id = "mara-bonea-16/tinydepth-experiments"
models_base = "/home/ubuntu/TinyDepth/models"

missing_models = [
    "Tiny-Depth-Basic-Uncertainty-Head-2",
    "Tiny-Depth-Basic-Uncertainty-Head-Smoothness",
    "Tiny-Depth-Uncertainty-Guided-Automasking",
    "Tiny-Depth-Weather-Robust-Feature-Supression",
]

for i, model_name in enumerate(missing_models):
    model_path = os.path.join(models_base, model_name, "models")
    print(f"\n[{i+1}/{len(missing_models)}] Uploading {model_name}...")
    api.upload_folder(
        folder_path=model_path,
        path_in_repo=f"{model_name}/models",
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["*.pth", "*.json"],
        ignore_patterns=["weights_latest"],
        commit_message=f"Add {model_name} weights",
    )
    print(f"  Done: {model_name}")
    if i < len(missing_models) - 1:
        print("  Waiting 30s before next upload...")
        time.sleep(30)

print("\nAll done.")
