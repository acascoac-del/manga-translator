"""
Download models from Hugging Face.
Run this before starting the API to pre-cache models.
"""

import os
import sys
import traceback
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN", None)
MODELS_DIR = os.path.join("app", "models")

models_to_download = [
    {
        "repo_id": "acsaco/MangaLingo",
        "filename": "mini-services/manga-api/models/detection/comictextdetector.pt",
        "subfolder": "detection",
        "local_name": "comictextdetector.pt",
    },
    {
        "repo_id": "acsaco/MangaLingo",
        "filename": "mini-services/manga-api/models/detection/comictextdetector.pt.onnx",
        "subfolder": "detection",
        "local_name": "comictextdetector.pt.onnx",
    },
    {
        "repo_id": "acsaco/MangaLingo",
        "filename": "mini-services/manga-api/models/inpainting/lama_large_512px.ckpt",
        "subfolder": "inpainting",
        "local_name": "lama_large_512px.ckpt",
    },
    {
        "repo_id": "acsaco/MangaLingo",
        "filename": "mini-services/manga-api/models/inpainting/inpainting_lama_mpe.ckpt",
        "subfolder": "inpainting",
        "local_name": "inpainting_lama_mpe.ckpt",
    },
]

print("Downloading models from Hugging Face...", flush=True)

for model in models_to_download:
    target_dir = os.path.join(MODELS_DIR, model["subfolder"])
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model["local_name"])

    if not os.path.exists(target_path):
        print(f"Downloading {model['local_name']}...", flush=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                repo_type="dataset",
                token=HF_TOKEN,
            )
            os.replace(downloaded_path, target_path)
            print(f"  -> {model['local_name']} saved.", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed: {model['local_name']}: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"{model['local_name']} already exists.", flush=True)

print("All models downloaded!", flush=True)
