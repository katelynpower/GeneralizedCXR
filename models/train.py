"""Train the DenseNet-121 pleural effusion model (VinBigData only) from the command line.

Script version of initial.ipynb for batch jobs, e.g. on an Open OnDemand / Slurm cluster:

    python train.py --data-dir /path/to/data --out-dir runs/vinbig_seed0 --seed 0

Writes to --out-dir: config.json, best_model.keras, history.csv, val/test predictions,
and val/test metrics tables (threshold chosen on validation, applied to test).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with vinbigdata_images/ and the label CSVs")
    p.add_argument("--out-dir", required=True, help="folder for this run's outputs (created if missing)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation steps; effective batch = batch-size x accum-steps")
    p.add_argument("--epochs", type=int, default=30, help="upper limit; early stopping usually ends sooner")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience on val AUROC")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=10,
                   help="seed for the train/val split; keep fixed across experiments")
    p.add_argument("--seed", type=int, default=0, help="seed for weight init, shuffling and augmentation")
    p.add_argument("--xla", action="store_true", help="compile with XLA (faster on large GPUs, needs more memory)")
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates for metric CIs (0 = skip)")
    p.add_argument("--subset", type=int, default=0,
                   help="quick test: use only the first N train images (and N/4 val/test); 0 = all")
    return p.parse_args()


# ---------------------------------------------------------------- data

def load_images(image_dir, image_ids):
    """Read the listed PNGs as 224x224x3 uint8 (grayscale copied to 3 channels)."""
    imgs = np.empty((len(image_ids), 224, 224, 3), dtype=np.uint8)
    for i, image_id in enumerate(image_ids):
        img = cv2.imread(str(image_dir / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_dir / f"{image_id}.png")
        imgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return imgs


def load_vinbigdata(data_dir):
    train = pd.read_csv(data_dir / "vinbigdata_labels_TRAIN.csv")
    test = pd.read_csv(data_dir / "vinbigdata_labels_TEST.csv")
    image_dir = data_dir / "vinbigdata_images"
    return (train["image_id"].values, load_images(image_dir, train["image_id"]), train["label"].astype(int).values,
            test["image_id"].values, load_images(image_dir, test["image_id"]), test["label"].astype(int).values)


# ---------------------------------------------------------------- model

def build_model(prevalence):
    pretrained = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return Sequential([
        pretrained,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        # float32 output keeps the probabilities numerically stable under mixed precision
        Dense(1, activation="sigmoid", dtype="float32",
              bias_initializer=Constant(np.log(prevalence / (1 - prevalence)))),
    ])


# ---------------------------------------------------------------- metrics

def choose_threshold(y_true, y_prob):
    """Youden's J threshold, chosen on the validation set only."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax(tpr - fpr)]


def calibration_intercept_slope(y_true, y_prob):
    """Calibration-in-the-large (intercept with slope fixed at 1) and calibration slope.
    Ideal: intercept 0, slope 1."""
    lp = logit(np.clip(y_prob, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    slope = LogisticRegression(C=np.inf).fit(lp, y_true).coef_[0, 0]  # unpenalized
    # intercept a in logit(P(y=1)) = a + lp, i.e. lp used as an offset
    nll = lambda a: -np.mean(y_true * np.log(expit(a + lp[:, 0]))
                             + (1 - y_true) * np.log(1 - expit(a + lp[:, 0])))
    intercept = minimize_scalar(nll).x
    return intercept, slope


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with equal-count (quantile) bins, which suits low prevalence."""
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
    return ece


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "F1": 2 * ppv * sens / (ppv + sens) if ppv + sens else np.nan,
        "Balanced accuracy": (sens + spec) / 2,
        "Brier score": brier_score_loss(y_true, y_prob),
        "Calibration intercept": intercept,
        "Calibration slope": slope,
        "ECE": expected_calibration_error(y_true, y_prob),
        "Mean predicted prob": y_prob.mean(),
        "Observed prevalence": y_true.mean(),
    }


def metrics_table(y_true, y_prob, threshold, n_boot=1000, seed=42):
    """Point estimates plus stratified-bootstrap 95% CIs (set n_boot=0 to skip)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    point = compute_metrics(y_true, y_prob, threshold)
    table = pd.DataFrame({"Estimate": point})
    if n_boot:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
        boots = []
        for _ in range(n_boot):
            # resample positives and negatives separately so every replicate keeps the same prevalence
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            boots.append(compute_metrics(y_true[idx], y_prob[idx], threshold))
        boots = pd.DataFrame(boots)
        table["95% CI low"] = boots.quantile(0.025)
        table["95% CI high"] = boots.quantile(0.975)
    return table.round(3)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    tf.keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus and not args.no_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("GPUs:", gpus, flush=True)

    t0 = time.time()
    train_ids, train_imgs, train_labels, test_ids, test_imgs, test_labels = load_vinbigdata(data_dir)
    train_ids, val_ids, train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_ids, train_imgs, train_labels, test_size=args.val_frac, stratify=train_labels,
        random_state=args.split_seed)
    if args.subset:
        n, m = args.subset, max(args.subset // 4, 50)
        train_ids, train_imgs, train_labels = train_ids[:n], train_imgs[:n], train_labels[:n]
        val_ids, val_imgs, val_labels = val_ids[:m], val_imgs[:m], val_labels[:m]
        test_ids, test_imgs, test_labels = test_ids[:m], test_imgs[:m], test_labels[:m]
    print(f"Loaded images in {time.time() - t0:.0f}s | train {len(train_labels)} ({train_labels.mean():.3f}) "
          f"| val {len(val_labels)} ({val_labels.mean():.3f}) | test {len(test_labels)} ({test_labels.mean():.3f})",
          flush=True)

    # same DenseNet preprocessing for train, val and test (augmentation on training batches only)
    datagen = ImageDataGenerator(brightness_range=[0.8, 1.2], preprocessing_function=preprocess_input)
    # val/test are preprocessed batch by batch, in order, so no float32 copy of the whole array is kept in memory
    evalgen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def eval_flow(imgs, labels=None):
        return evalgen.flow(imgs, labels, batch_size=args.batch_size, shuffle=False)

    model = build_model(train_labels.mean())
    accum = args.accum_steps if args.accum_steps > 1 else None  # Keras only accepts None or >= 2
    model.compile(optimizer=Adam(learning_rate=args.lr, gradient_accumulation_steps=accum),
                  loss="binary_crossentropy",
                  metrics=[metrics.AUC(name="auroc"), metrics.AUC(curve="PR", name="auprc")],
                  jit_compile=args.xla)
    model.fit(
        datagen.flow(train_imgs, train_labels, batch_size=args.batch_size, seed=args.seed),
        epochs=args.epochs, verbose=2, validation_data=eval_flow(val_imgs, val_labels),
        callbacks=[EarlyStopping(monitor="val_auroc", mode="max", patience=args.patience, restore_best_weights=True),
                   ModelCheckpoint(str(out_dir / "best_model.keras"), monitor="val_auroc", mode="max",
                                   save_best_only=True),
                   CSVLogger(str(out_dir / "history.csv"))])
    print(f"Training finished after {(time.time() - t0) / 60:.1f} min", flush=True)

    # threshold from validation, applied unchanged to test
    val_prob = model.predict(eval_flow(val_imgs), verbose=0)[:, 0]
    test_prob = model.predict(eval_flow(test_imgs), verbose=0)[:, 0]
    threshold = float(choose_threshold(val_labels, val_prob))
    pd.DataFrame({"image_id": val_ids, "label": val_labels, "prob": val_prob}).to_csv(
        out_dir / "predictions_val.csv", index=False)
    pd.DataFrame({"image_id": test_ids, "label": test_labels, "prob": test_prob}).to_csv(
        out_dir / "predictions_test.csv", index=False)
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))

    for name, y, prob in [("val", val_labels, val_prob), ("test", test_labels, test_prob)]:
        table = metrics_table(y, prob, threshold, n_boot=args.n_boot)
        table.to_csv(out_dir / f"metrics_{name}.csv")
        print(f"\n{name} metrics (threshold {threshold:.3f}):\n{table.to_string()}", flush=True)


if __name__ == "__main__":
    main()
