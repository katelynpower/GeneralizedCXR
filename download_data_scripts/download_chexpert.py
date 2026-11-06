## Download CheXpert data from Redivis

from pathlib import Path
import redivis

DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

organization = redivis.organization("AIMI")
dataset = organization.dataset("chexpert_plus")

df = dataset.table("PNG_train").to_pandas_dataframe()

patients = set()
for f in df["file_name"].dropna():
    pathname = str(f).replace("/", "_")
    if pathname.split("_")[3] == "frontal.png" and pathname.split("_")[0] not in patients:
        patients.add(pathname.split("_")[0])
        redivis.table("aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj").file(f).download(str(DATA_DIR /Path("chexpert_images") / pathname))
    if len(patients) >= 10728:
            break

table = redivis.table("aimi.chexpert_plus:5yyj:v1_0.chexpert_labels:pmec")
file = table.file("report_fixed.json")
file.download(DATA_DIR / "chexpert_report_labels.json")
file = table.file("findings_fixed.json")
file.download(DATA_DIR / "chexpert_findings_labels.json")
file = table.file("impression_fixed.json")
file.download(DATA_DIR / "chexpert_impression_labels.json")

# per-image metadata (AP/PA view, age, sex)
meta = redivis.table("aimi.chexpert_plus:5yyj:v1_0.df_chexpert_plus_240401:bavj").to_pandas_dataframe(
    variables=["path_to_image", "frontal_lateral", "ap_pa", "age", "sex"])
meta.to_csv(DATA_DIR / "chexpert_plus_meta.csv", index=False)

(DATA_DIR / "png_train.csv").rename(DATA_DIR / "chexpert_png_train.csv")


## CLEAN UP

import json
import pandas as pd

# --- Config ---
IMAGE_DIR = DATA_DIR / "chexpert_images"          # folder with your images
OUTPUT_CSV = DATA_DIR / "labels_filtered.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

def load_labels(json_path):
    FILENAME_KEY = "path_to_image"

    text = json_path.read_text(encoding="utf-8-sig").strip()
    data = [json.loads(line) for line in text.splitlines() if line.strip()]
    df = pd.DataFrame(data).rename(
            columns={FILENAME_KEY: "filename"}
        )

    existing = {
    p.name for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }

    # Normalize to basename in case the JSON stores paths
    df["filename"] = df["filename"].apply(lambda x: Path(x.replace("/", "_").split("_", 1)[-1]).with_suffix(".png").name)
    df_filtered = df[df["filename"].isin(existing)].reset_index(drop=True)

    return df_filtered

# label from the full-report CheXbert output only (it covers both findings and impression)
report = load_labels(DATA_DIR / "chexpert_report_labels.json")

print("report:")
print(report["Pleural Effusion"].value_counts(dropna=False))

# 1 = positive, 0 / not mentioned = negative, -1 = uncertain -> dropped from the labels
# (uncertain images stay on disk but are not in the TRAIN/TEST CSVs)
pe = report[["filename", "Pleural Effusion"]].drop_duplicates("filename")
uncertain = pe["Pleural Effusion"] == -1
print(f"{uncertain.sum()} uncertain images dropped")
pe = pe[~uncertain]
pe["label"] = pe["Pleural Effusion"] == 1

# keep PA films only (AP films stay on disk but are not in the TRAIN/TEST CSVs)
meta = pd.read_csv(DATA_DIR / "chexpert_plus_meta.csv")
meta["filename"] = meta["path_to_image"].apply(lambda x: Path(x.replace("/", "_").split("_", 1)[-1]).with_suffix(".png").name)
pe = pe[pe["filename"].isin(meta.loc[meta["ap_pa"] == "PA", "filename"])]
print(f"{len(pe)} PA images kept")

pe = pe[["filename", "label"]]

# no CheXpert test split: every experiment is tested on VinBigData, so all images are training data
print("Peural Effusion rate trainimages:")
print(pe["label"].mean())

pe.to_csv(DATA_DIR / "chexpert_labels_TRAIN.csv", index=False)


import cv2


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

for file_path in IMAGE_DIR.iterdir():
    if file_path.is_file():
        process_xray(str(file_path))