import subprocess
from pathlib import Path
import zipfile
import shutil
#!pip install kaggle

DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "nih-chest-xrays/data",
    "-p", str(DATA_DIR)
], check=True)

IMAGE_DIR = DATA_DIR / "nih_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

zip_path = Path(DATA_DIR) / "data.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    image_files = [
        f for f in zip_ref.namelist() if f.startswith("images") and f.endswith(".png")
    ]
    for i, image_file in enumerate(image_files, start=1):
        with zip_ref.open(image_file) as source:
            # Preserve original extension
            extension = Path(image_file).suffix
            # Give each image a unique filename
            destination = IMAGE_DIR / f"image_{i:05d}{extension}"
            with open(destination, "wb") as target:
                shutil.copyfileobj(source, target)
    with zip_ref.open("Data_Entry_2017.csv") as source:
        destination = DATA_DIR / "nih_labels.csv"
        with open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
# remove the zip file after extraction
zip_path.unlink()