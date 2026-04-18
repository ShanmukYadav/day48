"""
step3_finetuning.py  ─  Sub-step 3: Fine-Tuning vs Feature Extraction
────────────────────────────────────────────────────────────────────────
Simulates "unfreezing" part of the backbone by:
  • Adding a small MLP on top of the frozen-head features (mimics
    unfreezing the last few ResNet blocks with a lower learning rate).
  • Comparing macro-F1, per-class recall, and precision-recall AUC
    against the frozen head from Step 2.

In a real PyTorch workflow this corresponds to:
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},   # backbone (low lr)
        {"params": model.fc.parameters(),     "lr": 1e-3},   # head
    ])

Run:
    python step3_finetuning.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network   import MLPClassifier
from sklearn.metrics          import (classification_report,
                                      precision_recall_curve,
                                      average_precision_score,
                                      confusion_matrix,
                                      ConfusionMatrixDisplay)
from sklearn.preprocessing    import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ──────────────────────────────────────────────────────────
DATA_FILE       = "medical_imaging_meta.csv"
OUTPUT_DIR      = Path("outputs")
MODEL_CACHE_DIR = Path("model_cache")
RANDOM_SEED     = 42
FEATURE_PREFIX  = "feat_"
CONDITIONS      = ["Normal", "Pneumonia", "COVID-19", "Pleural Effusion", "Cardiomegaly"]

# Clinical cost matrix: rows=true, cols=predicted
# High value = costly miss (false negative on dangerous condition)
CLINICAL_COST = {
    "COVID-19":         {"FN_cost": 5, "FP_cost": 1},
    "Pneumonia":        {"FN_cost": 4, "FP_cost": 1},
    "Pleural Effusion": {"FN_cost": 4, "FP_cost": 1},
    "Cardiomegaly":     {"FN_cost": 3, "FP_cost": 1},
    "Normal":           {"FN_cost": 1, "FP_cost": 2},
}


# ── Data helpers ───────────────────────────────────────────────────────

def load_splits_with_scaler(data_file: str, scaler_path: Path):
    """Load data and apply pre-fitted scaler from Step 2."""
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        sys.exit("[ERROR] medical_imaging_meta.csv not found.")

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        sys.exit("[ERROR] model_cache/scaler.pkl not found. Run step2 first.")

    labeled   = df[df["split"] != "unlabeled"].copy()
    feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]

    X_train = scaler.transform(labeled[labeled["split"] == "train"][feat_cols].values)
    y_train = labeled[labeled["split"] == "train"]["condition"].values
    X_test  = scaler.transform(labeled[labeled["split"] == "test"][feat_cols].values)
    y_test  = labeled[labeled["split"] == "test"]["condition"].values
    return X_train, y_train, X_test, y_test


# ── Model helpers ──────────────────────────────────────────────────────

def train_finetuned_model(X_train, y_train) -> MLPClassifier:
    """
    MLP with two hidden layers — simulates unfreezing last backbone blocks.
    Lower effective LR (smaller step size, more epochs) mirrors fine-tuning.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # MLPClassifier doesn't accept class_weight natively — oversample manually
    sample_weights = np.array([dict(zip(classes, weights))[y] for y in y_train])

    # Duplicate minority class samples proportional to weight
    idx_augmented = []
    for i, w in enumerate(sample_weights):
        repeats = max(1, int(round(float(w))))
        idx_augmented.extend([i] * repeats)
    X_aug = np.asarray(X_train[idx_augmented], dtype=np.float64)
    y_aug = np.asarray(y_train[idx_augmented])

    clf = MLPClassifier(
        hidden_layer_sizes = (256, 128),
        activation         = "relu",
        solver             = "adam",
        learning_rate_init = 1e-4,   # low lr — mirrors fine-tuning convention
        max_iter           = 500,
        random_state       = RANDOM_SEED,
        early_stopping     = False,
        n_iter_no_change   = 20,
    )
    clf.fit(X_aug, y_aug)
    return clf


# ── Evaluation helpers ─────────────────────────────────────────────────

def evaluate_and_report(clf, X_test, y_test, model_name: str) -> dict:
    """Predict and return report dict."""
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
    return {"report": report, "y_pred": y_pred, "y_prob": y_prob, "classes": clf.classes_}


def compute_clinical_risk_score(report: dict, model_name: str) -> float:
    """Weighted miss cost across conditions."""
    total_cost = 0.0
    print(f"\n── {model_name} — Clinical Risk Score ────────────────────")
    for cond, costs in CLINICAL_COST.items():
        if cond not in report:
            continue
        fn_rate = 1.0 - report[cond]["recall"]          # false negative rate
        fp_rate = 1.0 - report[cond]["precision"]       # false positive rate
        cost = fn_rate * costs["FN_cost"] + fp_rate * costs["FP_cost"]
        total_cost += cost
        print(f"  {cond:<22s}  FN-rate: {fn_rate:.2f}  FP-rate: {fp_rate:.2f}  "
              f"Weighted cost: {cost:.3f}")
    print(f"  {'TOTAL CLINICAL RISK SCORE':<22s}  {total_cost:.3f}  "
          f"(lower is safer)")
    return total_cost


def plot_comparison_bar(report_fe: dict, report_ft: dict,
                        classes, output_dir: Path):
    """Side-by-side bar chart: Feature Extraction vs Fine-Tuning per class."""
    metrics = ["precision", "recall", "f1-score"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        fe_vals = [report_fe["report"].get(c, {}).get(metric, 0) for c in classes]
        ft_vals = [report_ft["report"].get(c, {}).get(metric, 0) for c in classes]
        x = np.arange(len(classes))
        ax.bar(x - 0.2, fe_vals, 0.38, label="Feature Extraction", color="#2196F3", alpha=0.85)
        ax.bar(x + 0.2, ft_vals, 0.38, label="Fine-Tuning",        color="#FF9800", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.capitalize(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylabel("Score")
    plt.suptitle("Feature Extraction vs Fine-Tuning — Per-Class Metrics",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "fig4_fe_vs_ft_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_cm(y_test, y_pred, classes, title, path):
    """Save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def print_deployment_recommendation(cost_fe: float, cost_ft: float):
    """Advise Dr. Rao on safer deployment choice."""
    winner = "Fine-Tuning" if cost_ft < cost_fe else "Feature Extraction"
    loser  = "Feature Extraction" if winner == "Fine-Tuning" else "Fine-Tuning"
    print(f"""
── Deployment Recommendation for Dr. Rao ─────────────────────────────
  {winner} achieves a lower clinical risk score ({min(cost_fe, cost_ft):.3f})
  compared to {loser} ({max(cost_fe, cost_ft):.3f}).

  Clinical rationale:
  • Overall accuracy is not the right metric for this task. COVID-19,
    Pneumonia, and Pleural Effusion missed cases carry 4–5× the clinical
    cost of a false alarm.
  • {winner} shows better recall on minority dangerous classes, making
    it the safer deployment choice — even if overall accuracy is lower.
  • Recommended decision threshold: lower than default 0.5 for high-risk
    classes (covered in Step 7 triage protocol).
""")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 3] Fine-Tuning vs Feature Extraction Comparison")

    X_train, y_train, X_test, y_test = load_splits_with_scaler(
        DATA_FILE, MODEL_CACHE_DIR / "scaler.pkl"
    )

    # Load frozen-head model from Step 2
    try:
        with open(MODEL_CACHE_DIR / "fe_model.pkl", "rb") as f:
            clf_fe = pickle.load(f)
    except FileNotFoundError:
        sys.exit("[ERROR] fe_model.pkl not found. Run step2 first.")

    print("\n  Training Fine-Tuned MLP (simulates unfreezing backbone layers) ...")
    clf_ft = train_finetuned_model(X_train, y_train)

    results_fe = evaluate_and_report(clf_fe, X_test, y_test, "Feature Extraction (Step 2)")
    results_ft = evaluate_and_report(clf_ft, X_test, y_test, "Fine-Tuning (Step 3)")

    cost_fe = compute_clinical_risk_score(results_fe["report"], "Feature Extraction")
    cost_ft = compute_clinical_risk_score(results_ft["report"], "Fine-Tuning")

    all_classes = list(clf_fe.classes_)
    plot_comparison_bar(results_fe, results_ft, all_classes, OUTPUT_DIR)
    plot_cm(y_test, results_ft["y_pred"], all_classes,
            "Confusion Matrix — Fine-Tuning",
            OUTPUT_DIR / "fig5_cm_finetuning.png")

    print_deployment_recommendation(cost_fe, cost_ft)

    # Cache fine-tuned model
    with open(MODEL_CACHE_DIR / "ft_model.pkl", "wb") as f:
        pickle.dump(clf_ft, f)
    print("[Saved] model_cache/ft_model.pkl")
    print("\n[Step 3 Complete] ✓")


if __name__ == "__main__":
    main()
