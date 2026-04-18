"""
step2_feature_extraction.py  ─  Sub-step 2: Transfer Learning (Feature Extraction)
────────────────────────────────────────────────────────────────────────────────────
Simulates freezing a pre-trained ResNet backbone (128-d embeddings already extracted
and stored in medical_imaging_meta.csv) and training a logistic regression head.

This mirrors exactly what happens when you:
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 5)   # new head
    # freeze backbone — only head is trainable

The 128-d feat_* columns represent ResNet features after global average pooling.

Run:
    python step2_feature_extraction.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.metrics        import (classification_report,
                                    confusion_matrix,
                                    ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ──────────────────────────────────────────────────────────
DATA_FILE        = "medical_imaging_meta.csv"
OUTPUT_DIR       = Path("outputs")
MODEL_CACHE_DIR  = Path("model_cache")
RANDOM_SEED      = 42
MAX_ITER         = 1000
CONDITIONS       = ["Normal", "Pneumonia", "COVID-19", "Pleural Effusion", "Cardiomegaly"]
FEATURE_PREFIX   = "feat_"


# ── Data helpers ───────────────────────────────────────────────────────

def load_labeled_splits(path: str):
    """Return train and test feature matrices and label arrays."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] {path} not found. Run generate_dataset.py first.")

    labeled = df[df["split"] != "unlabeled"].copy()
    feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]

    X_train = labeled[labeled["split"] == "train"][feat_cols].values
    y_train = labeled[labeled["split"] == "train"]["condition"].values
    X_test  = labeled[labeled["split"] == "test"][feat_cols].values
    y_test  = labeled[labeled["split"] == "test"]["condition"].values

    print(f"  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")
    print(f"  Features   : {X_train.shape[1]}")
    return X_train, y_train, X_test, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit scaler on train, apply to both."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# ── Model helpers ──────────────────────────────────────────────────────

def compute_balanced_weights(y_train: np.ndarray) -> dict:
    """Compute per-class weights to penalise majority classes."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def train_feature_extraction_head(X_train, y_train, class_weight_dict):
    """Train logistic regression classification head (frozen backbone proxy)."""
    clf = LogisticRegression(
        max_iter      = MAX_ITER,
        random_state  = RANDOM_SEED,
        class_weight  = class_weight_dict,
        solver        = "lbfgs",
        C             = 1.0,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Evaluation helpers ─────────────────────────────────────────────────

def evaluate_model(clf, X_test, y_test, model_name: str) -> dict:
    """Return per-class metrics and print classification report."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=clf.classes_,
        output_dict=True,
        zero_division=0,
    )
    print(f"\n── {model_name} — Classification Report ───────────────────")
    print(classification_report(y_test, y_pred, target_names=clf.classes_, zero_division=0))
    return {"report": report, "y_pred": y_pred, "y_prob": y_prob}


def plot_confusion_matrix(y_test, y_pred, classes, title: str, path: Path):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def print_per_class_clinical_analysis(report: dict, model_name: str):
    """Highlight which classes the model struggles with and clinical impact."""
    print(f"\n── {model_name} — Clinical Risk Analysis ──────────────────")
    risky_classes = []
    for cond in CONDITIONS:
        if cond not in report:
            continue
        recall = report[cond]["recall"]
        flag   = ""
        if recall < 0.60:
            flag = "  🔴 HIGH RISK — too many false negatives"
            risky_classes.append(cond)
        elif recall < 0.75:
            flag = "  🟡 MODERATE RISK"
        print(f"  {cond:<22s}  Recall: {recall:.2f}  F1: {report[cond]['f1-score']:.2f}{flag}")

    if risky_classes:
        print(f"""
  Dr. Rao's deployment concern:
  Low recall on {risky_classes} means patients with these conditions
  are being missed. In a clinical screening context, false negatives
  carry life-threatening consequences — these classes need augmentation,
  oversampling (SMOTE), or a lower decision threshold before deployment.
""")
    else:
        print("\n  All classes above 60% recall threshold — acceptable for pilot.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 2] Feature Extraction — Frozen Backbone + Logistic Head")
    print("  (Equivalent to: resnet50(pretrained=True) + frozen backbone)")

    X_train, y_train, X_test, y_test = load_labeled_splits(DATA_FILE)

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    class_weights = compute_balanced_weights(y_train)
    print(f"\n  Class weights: { {k: round(v, 2) for k, v in class_weights.items()} }")

    clf = train_feature_extraction_head(X_train_sc, y_train, class_weights)
    print(f"\n  Model: LogisticRegression  (C=1.0, balanced class weights)")
    print(f"  Backbone: FROZEN (only classification head trained)")

    results = evaluate_model(clf, X_test_sc, y_test, "Feature Extraction")
    print_per_class_clinical_analysis(results["report"], "Feature Extraction")

    plot_confusion_matrix(
        y_test, results["y_pred"],
        classes=clf.classes_,
        title="Confusion Matrix — Feature Extraction (Frozen Backbone)",
        path=OUTPUT_DIR / "fig3_cm_feature_extraction.png",
    )

    # Save model artifacts for downstream steps
    import pickle
    with open(MODEL_CACHE_DIR / "fe_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_CACHE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\n[Saved] Model artifacts to model_cache/")
    print("\n[Step 2 Complete] ✓")


if __name__ == "__main__":
    main()
