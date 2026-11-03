## Download CXR images from VinBigData Kaggle - only a 1GB subset

# import dependencies
import subprocess
from pathlib import Path
import zipfile
#subprocess.run(["pip", "install", "kaggle"], check=True)

# set up
DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
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


