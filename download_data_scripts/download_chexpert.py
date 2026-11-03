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

(DATA_DIR / "png_train.csv").rename(DATA_DIR / "chexpert_png_train.csv")