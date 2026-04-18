"""
step5_me1_prep_and_unlabeled.py  ─  Sub-step 5: ME1 Prep + Unlabeled Classification
──────────────────────────────────────────────────────────────────────────────────────
Part A: Personal synthesis — Batch Normalisation (selected as weakest area)
Part B: Classify the 30 unlabeled images using the best model from Steps 2 & 3

Run:
    python step5_me1_prep_and_unlabeled.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE       = "medical_imaging_meta.csv"
OUTPUT_DIR      = Path("outputs")
MODEL_CACHE_DIR = Path("model_cache")
FEATURE_PREFIX  = "feat_"


# ── Part A: ME1 Personal Synthesis ────────────────────────────────────

ME1_SYNTHESIS = """
╔══════════════════════════════════════════════════════════════════════╗
║             ME1 PREPARATION — Personal Synthesis (Sub-step 5)       ║
╠══════════════════════════════════════════════════════════════════════╣
║  TOPIC: Batch Normalisation in Deep Neural Networks                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  200-WORD EXPLANATION (as if teaching a classmate):                  ║
║                                                                      ║
║  Imagine training a deep network where each layer receives inputs    ║
║  from the previous layer. As weights update during backprop, the     ║
║  distribution of those inputs keeps shifting — a problem called      ║
║  "internal covariate shift." Batch Normalisation (BN) fixes this     ║
║  by normalising the output of each layer to have zero mean and       ║
║  unit variance, computed across the current mini-batch:              ║
║                                                                      ║
║      x_hat = (x − μ_B) / √(σ²_B + ε)                               ║
║      y = γ·x_hat + β    (γ, β are learned scale and shift)          ║
║                                                                      ║
║  The learnable γ and β allow the network to undo normalisation if    ║
║  that's what the task needs. At inference time, BN uses running      ║
║  statistics (exponential moving average of μ and σ) instead of       ║
║  mini-batch stats, so behaviour is consistent across batch sizes.    ║
║                                                                      ║
║  Why does this help?                                                 ║
║  1. Higher learning rates become safe — gradients stay well-scaled.  ║
║  2. Acts as mild regularisation (noise from batch statistics).       ║
║  3. Networks converge faster and are less sensitive to weight init.  ║
║                                                                      ║
║  Common trap: BN behaves differently in train() vs eval() mode in    ║
║  PyTorch. Forgetting model.eval() before inference is a classic bug. ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  INTERVIEW QUESTION 1:                                               ║
║  "What is internal covariate shift and how does Batch Normalisation  ║
║   address it? What are the learnable parameters and what do they do?"║
║                                                                      ║
║  MODEL ANSWER:                                                       ║
║  Internal covariate shift is the change in the distribution of each  ║
║  layer's inputs as model parameters update. BN normalises activations ║
║  per mini-batch (zero mean, unit variance) then applies learnable     ║
║  scale (γ) and shift (β). γ and β allow the model to represent any   ║
║  mean/variance if needed, recovering the identity transform. This     ║
║  stabilises and accelerates training by keeping gradient magnitudes   ║
║  consistent across layers.                                           ║
║                                                                      ║
║  INTERVIEW QUESTION 2:                                               ║
║  "How does Batch Normalisation behave differently during training     ║
║   vs inference, and why does this matter in production?"             ║
║                                                                      ║
║  MODEL ANSWER:                                                       ║
║  During training, BN normalises using the current mini-batch mean    ║
║  and variance. During inference, it uses a running mean/variance     ║
║  accumulated during training (via exponential moving average). This   ║
║  is critical in production: if the model is in train() mode during   ║
║  inference, batch stats from test data leak into normalisation,       ║
║  causing inconsistent predictions — especially dangerous with         ║
║  single-sample inference (batch size=1, variance=undefined).         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ── Part B: Classify unlabeled images ──────────────────────────────────

def load_unlabeled_features(data_file: str, scaler) -> tuple:
    """Extract 30 unlabeled rows and scale features."""
    df = pd.read_csv(data_file)
    unlabeled = df[df["split"] == "unlabeled"].copy()
    feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    X_unlabeled = scaler.transform(unlabeled[feat_cols].values)
    image_ids   = unlabeled["image_id"].values
    return X_unlabeled, image_ids, unlabeled


def classify_unlabeled(clf, X_unlabeled, image_ids,
                       unlabeled_df: pd.DataFrame) -> pd.DataFrame:
    """Predict class + confidence for each unlabeled image."""
    y_pred  = clf.predict(X_unlabeled)
    y_proba = clf.predict_proba(X_unlabeled)
    confidence = y_proba.max(axis=1)

    results = pd.DataFrame({
        "image_id":           image_ids,
        "predicted_condition": y_pred,
        "confidence":         np.round(confidence, 4),
        "hospital_site":      unlabeled_df["hospital_site"].values,
        "equipment":          unlabeled_df["equipment"].values,
        "image_quality":      unlabeled_df["image_quality"].values,
    })
    return results


def plot_unlabeled_predictions(results: pd.DataFrame, output_dir: Path):
    """Bar chart of predicted class distribution for unlabeled set."""
    counts = results["predicted_condition"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Predicted class distribution
    axes[0].bar(counts.index, counts.values, color="#673AB7", alpha=0.85)
    axes[0].set_title("Predicted Conditions — 30 Unlabeled Images",
                      fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Condition")
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")

    # Confidence distribution
    axes[1].hist(results["confidence"], bins=10, color="#009688", alpha=0.85,
                 edgecolor="white")
    axes[1].axvline(0.70, color="red",    linestyle="--", label="Auto-classify (0.70)")
    axes[1].axvline(0.50, color="orange", linestyle="--", label="Review threshold (0.50)")
    axes[1].set_title("Prediction Confidence Distribution",
                      fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Max Softmax Confidence")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = output_dir / "fig8_unlabeled_predictions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_predictions(results: pd.DataFrame, path: Path):
    """Save predictions CSV."""
    results.to_csv(path, index=False)
    print(f"[Saved] {path}")
    print(f"\n── Unlabeled Prediction Sample (first 10) ─────────────────")
    print(results[["image_id", "predicted_condition", "confidence"]].head(10).to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part A ──────────────────────────────────────────────────────────
    print(ME1_SYNTHESIS)

    # ── Part B ──────────────────────────────────────────────────────────
    print("\n[Step 5B] Classifying 30 Unlabeled Images ...")

    try:
        with open(MODEL_CACHE_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_CACHE_DIR / "ft_model.pkl", "rb") as f:
            clf_ft = pickle.load(f)
        with open(MODEL_CACHE_DIR / "fe_model.pkl", "rb") as f:
            clf_fe = pickle.load(f)
    except FileNotFoundError:
        sys.exit("[ERROR] Model cache not found. Run step2 and step3 first.")

    X_unlabeled, image_ids, unlabeled_df = load_unlabeled_features(DATA_FILE, scaler)

    # Use best model (fine-tuned from step 3)
    results_ft = classify_unlabeled(clf_ft, X_unlabeled, image_ids, unlabeled_df)
    results_fe = classify_unlabeled(clf_fe, X_unlabeled, image_ids, unlabeled_df)

    # Ensemble: average probabilities
    proba_ft = clf_ft.predict_proba(X_unlabeled)
    proba_fe = clf_fe.predict_proba(X_unlabeled)

    # Align classes
    all_classes = list(clf_ft.classes_)
    fe_classes  = list(clf_fe.classes_)
    proba_fe_aligned = np.zeros((len(X_unlabeled), len(all_classes)))
    for i, cls in enumerate(all_classes):
        if cls in fe_classes:
            proba_fe_aligned[:, i] = proba_fe[:, fe_classes.index(cls)]

    ensemble_proba     = 0.6 * proba_ft + 0.4 * proba_fe_aligned
    ensemble_pred      = np.array(all_classes)[ensemble_proba.argmax(axis=1)]
    ensemble_confidence= ensemble_proba.max(axis=1)

    final_results = pd.DataFrame({
        "image_id":            image_ids,
        "predicted_condition": ensemble_pred,
        "confidence":          np.round(ensemble_confidence, 4),
        "ft_prediction":       results_ft["predicted_condition"].values,
        "fe_prediction":       results_fe["predicted_condition"].values,
        "hospital_site":       unlabeled_df["hospital_site"].values,
        "equipment":           unlabeled_df["equipment"].values,
        "image_quality":       unlabeled_df["image_quality"].values,
    })

    print(f"\n  Ensemble: 60% Fine-Tuned + 40% Feature Extraction")
    print(f"  Total unlabeled classified: {len(final_results)}")

    plot_unlabeled_predictions(final_results, OUTPUT_DIR)
    save_predictions(final_results, OUTPUT_DIR / "unlabeled_predictions.csv")

    print("\n[Step 5 Complete] ✓")


if __name__ == "__main__":
    main()
