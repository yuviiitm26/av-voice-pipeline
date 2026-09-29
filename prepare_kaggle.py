import os
import json
import nbformat as nbf

# 1. Setup the export directory
export_dir = "kaggle_export"
os.makedirs(export_dir, exist_ok=True)

# 2. Read the python file and split by '# %%' cells
with open("av_voice_pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

# Our python file uses '# %%' to separate cells
raw_cells = content.split("# %%")

# 3. Create a Jupyter Notebook
nb = nbf.v4.new_notebook()
cells = []

# Add an initial installation cell for Kaggle
install_cell = "!pip install -q soundfile librosa gradio pydantic"
cells.append(nbf.v4.new_code_cell(install_cell))

# Process the rest of the cells
for c in raw_cells:
    c = c.strip()
    if not c:
        continue
    # Convert Gradio launch to use share=True so it creates a public link from Kaggle
    c = c.replace("_app.launch(share=False, show_error=True, inbrowser=True)", "_app.launch(share=True, show_error=True)")
    cells.append(nbf.v4.new_code_cell(c))

nb["cells"] = cells

# Write the notebook to the export directory
notebook_path = os.path.join(export_dir, "av-voice-pipeline.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook: {notebook_path}")

# 4. Create the Kaggle kernel-metadata.json
metadata = {
  "id": "yuvrajgosainiitm/av-voice-pipeline",
  "title": "AV Voice Automation Pipeline",
  "code_file": "av-voice-pipeline.ipynb",
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

print(f"Created metadata: {meta_path}")
