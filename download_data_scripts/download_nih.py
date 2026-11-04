import subprocess
from pathlib import Path
import zipfile
import shutil
import pandas as pd
import cv2
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

labels = pd.read_csv(DATA_DIR / "nih_labels.csv")
labels["label"] = labels["Finding Labels"].apply(lambda x: x.split("|"))
labels["label"] = labels["label"].apply(lambda x: "Effusion" in x)

patients = labels["Patient ID"].unique()
testpatients = list(pd.Series(list(patients)).sample(n=1000,random_state=42))
testimages = labels[labels["Patient ID"].isin(testpatients)][["Image Index","Patient ID","label"]]
trainimages = labels[~labels["Patient ID"].isin(testpatients)][["Image Index","Patient ID","label"]]
testimages.to_csv(DATA_DIR / "nih_labels_TEST.csv", index=False)
trainimages.to_csv(DATA_DIR / "nih_labels_TRAIN.csv", index=False)


def process_xray(image_path, target_size=(224, 224)):
  # 1. Read the original image
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    print(f'Error: Could not read image at {image_path}')
    return

  # 2. Perform your resizing/preprocessing
  # (Using direct resize here for simplicity; replace with your padding logic if preferred)
  processed_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

  # 3. Save directly back to the original path
  # Note: For PNG, default compression is fine, but you can explicitly set it.
  success = cv2.imwrite(image_path, processed_img)

count = 0
for file_path in IMAGE_DIR.iterdir():
    count += 1
    if file_path.is_file():
        process_xray(str(file_path))
    if count % 1000 == 0:
       print(f'Processed {count/len(list(IMAGE_DIR.iterdir()))*100:.2f}% of images')
        