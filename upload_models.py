from huggingface_hub import HfApi

api = HfApi()

api.create_repo("tinydepth-experiments", repo_type="model", private=True, exist_ok=True)

api.upload_folder(
    folder_path="/home/ubuntu/TinyDepth/models",
    repo_id="mara-bonea-16/tinydepth-experiments",
    repo_type="model",
    commit_message="Upload all model checkpoints"
)

print("Done!")
