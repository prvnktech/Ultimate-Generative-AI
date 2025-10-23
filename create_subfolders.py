# create_subfolders.py
import os

# List of subfolders to create inside each chapter
subfolders = ["notebooks", "scripts", "outputs"]

# Base path (repo root)
base_path = "."

# Loop through all folders in the repo root
for item in os.listdir(base_path):
    chapter_path = os.path.join(base_path, item)
    if os.path.isdir(chapter_path) and item.startswith("Chapter_"):
        for sub in subfolders:
            subfolder_path = os.path.join(chapter_path, sub)
            os.makedirs(subfolder_path, exist_ok=True)
        print(f"Created subfolders in {chapter_path}")

print("\n✅ Subfolders created successfully for all existing chapters!")
