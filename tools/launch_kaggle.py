import sys
import os
# Ensure the root directory is in the path so av_pipeline imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import nbformat as nbf
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    # 1. Setup export directory
    export_dir = "kaggle_server"
    os.makedirs(export_dir, exist_ok=True)

    # 2. Create the Notebook
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_code_cell(
            "!git clone https://github.com/yuviiitm26/av-voice-pipeline.git\n"
            "%cd av-voice-pipeline\n"
            "!pip install -r requirements.txt\n"
            "!pip install easyocr ultralytics soundfile librosa openai-whisper\n"
            "!python server.py"
        )
    ]
    nb["cells"] = cells

    nb_path = os.path.join(export_dir, "jev-brain-server.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    # 3. Create metadata
    metadata = {
      "id": "yuvrajgosainiitm/jev-brain-server",
      "title": "jev-brain-server",
      "code_file": "jev-brain-server.ipynb",
      "language": "python",
      "kernel_type": "notebook",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [],
      "competition_sources": [],
      "kernel_sources": []
    }
    
    meta_path = os.path.join(export_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 4. Push via Kaggle API
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    print("Pushing kernel to Kaggle...")
    api.kernels_push(export_dir)
    print("Successfully pushed! You can view it at: https://kaggle.com/yuvrajgosainiitm/jev-brain-server")

if __name__ == "__main__":
    main()
