## Download CXR images from VinBigData Kaggle - only a 1GB subset

# import dependencies
import subprocess
from pathlib import Path
import zipfile
import pandas as pd
import cv2
#subprocess.run(["pip", "install", "kaggle"], check=True)

# set up
DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = DATA_DIR / "vinbigdata_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
zip_path = Path(DATA_DIR) / "vinbigdata-chest-xray-original-png.zip"

# download png version of the dataset from kaggle
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "corochann/vinbigdata-chest-xray-original-png",
    "-p", str(DATA_DIR)
], check=True)

# Extract ONLY the train folder
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    for file in zip_ref.namelist():
        if file.startswith("train"):
            zip_ref.extract(file, str(DATA_DIR))
# remove the zip file after extraction
zip_path.unlink()

# download the train.csv file from the original dataset
subprocess.run([
    "kaggle", "competitions", "download",
    "-c", "vinbigdata-chest-xray-abnormalities-detection",
    "-f", "train.csv",
    "-p", str(DATA_DIR), "--unzip"
], check=True)

(DATA_DIR/"train").rename(DATA_DIR/"vinbigdata_images")
(DATA_DIR/"train.csv").rename(DATA_DIR/"vinbigdata_labels.csv")
(DATA_DIR/"train_meta.csv").rename(DATA_DIR/"vinbigdata_meta.csv")


labels = pd.read_csv(DATA_DIR / "vinbigdata_labels.csv", index_col=0)
labels["label"] = labels["class_name"].str.contains("Pleural effusion", na=False)
labels["label"].mean()

tests = labels.index.to_series().sample(n=1000,random_state=42)
testimages = labels.loc[labels.index.isin(tests)]["label"]
trainimages = labels.loc[~labels.index.isin(tests)]["label"]
testimages.to_csv(DATA_DIR / "vinbigdata_labels_TEST.csv")
trainimages.to_csv(DATA_DIR / "vinbigdata_labels_TRAIN.csv")

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