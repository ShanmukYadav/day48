"""
step7_triage_protocol.py  ─  Sub-step 7 (Hard): Triage Protocol for 30 Unlabeled Images
──────────────────────────────────────────────────────────────────────────────────────────
Designs and applies a three-tier triage protocol based on model confidence:

  Tier 1 — AUTO-CLASSIFY  : confidence ≥ 0.75  (send directly to report)
  Tier 2 — RADIOLOGIST REVIEW : 0.50 ≤ confidence < 0.75  (flag for expert)
  Tier 3 — REJECT / RESCAN : confidence < 0.50  (image quality or OOD)

Thresholds justified by clinical cost structure:
  • False negative on COVID-19/Pneumonia = cost 5 (life-threatening miss)
  • False positive on Normal = cost 2 (unnecessary follow-up, not fatal)
  → Conservative threshold: only auto-classify when confidence is high.

Expected false-negative rate is computed under two calibration assumptions.

Run:
    python step7_triage_protocol.py
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

# ── Triage thresholds (justified below) ────────────────────────────────
T_AUTO_CLASSIFY = 0.75   # High confidence: auto-classify → report
T_REVIEW        = 0.50   # Medium: flag for radiologist
# Below T_REVIEW = Tier 3: reject / rescan

# Clinical cost per error type
FN_COST_DANGEROUS = 5.0    # missing COVID-19, Pneumonia, Effusion
FP_COST_NORMAL    = 2.0    # false alarm for normal patient

DANGEROUS_CONDITIONS = {"COVID-19", "Pneumonia", "Pleural Effusion"}


# ── Triage helpers ──────────────────────────────────────────────────────

def assign_tier(confidence: float) -> str:
    if confidence >= T_AUTO_CLASSIFY:
        return "Tier1_AutoClassify"
    elif confidence >= T_REVIEW:
        return "Tier2_RadiologyReview"
    else:
        return "Tier3_RejectRescan"


def compute_expected_fnr(results_tier1: pd.DataFrame,
                         calibrated: bool = True) -> float:
    """
    Estimate expected False-Negative Rate (FNR) at auto-classify threshold.

    If calibrated (well-calibrated model):
        confidence ≥ 0.75 → expect ≤ 25% error rate on positive cases
        FNR ≈ fraction of dangerous conditions auto-classified at low recall

    If NOT calibrated (overconfident model):
        softmax probabilities are higher than actual accuracy;
        effective FNR is scaled up by 1/(calibration factor).
    """
    n_dangerous = results_tier1[
        results_tier1["predicted_condition"].isin(DANGEROUS_CONDITIONS)
    ].shape[0]
    n_total = len(results_tier1)

    if n_total == 0:
        return 0.0

    mean_conf = results_tier1["confidence"].mean()

    if calibrated:
        # Expected error rate at mean confidence
        base_error = 1.0 - mean_conf
        fnr = base_error * (n_dangerous / max(n_total, 1))
    else:
        # Uncalibrated: typical softmax overconfidence factor ~1.3–1.5
        calibration_factor = 1.4
        effective_error    = (1.0 - mean_conf) * calibration_factor
        fnr = min(effective_error * (n_dangerous / max(n_total, 1)), 1.0)

    return round(fnr, 4)


def print_triage_report(triaged: pd.DataFrame):
    """Print tier summary and clinical interpretation."""
    tier_counts = triaged["tier"].value_counts()
    print(f"\n── Triage Protocol Results ─────────────────────────────────")
    print(f"\n  Thresholds:")
    print(f"    Tier 1 (Auto-classify)    : confidence ≥ {T_AUTO_CLASSIFY}")
    print(f"    Tier 2 (Radiologist review): {T_REVIEW} ≤ confidence < {T_AUTO_CLASSIFY}")
    print(f"    Tier 3 (Reject / Rescan)  : confidence < {T_REVIEW}\n")

    for tier in ["Tier1_AutoClassify", "Tier2_RadiologyReview", "Tier3_RejectRescan"]:
        count = tier_counts.get(tier, 0)
        subset = triaged[triaged["tier"] == tier]
        pct    = count / len(triaged) * 100
        mean_c = subset["confidence"].mean() if count > 0 else 0
        print(f"  {tier:<28s}  {count:3d} images  ({pct:5.1f}%)  "
              f"mean conf: {mean_c:.3f}")

    print(f"\n  Dangerous conditions in Tier 1 (auto-classified):")
    tier1 = triaged[triaged["tier"] == "Tier1_AutoClassify"]
    for cond in DANGEROUS_CONDITIONS:
        n = (tier1["predicted_condition"] == cond).sum()
        if n > 0:
            print(f"    {cond:<22s} {n} images — will bypass radiologist")

    print(f"\n  Threshold Justification:")
    print(f"    T=0.75 for auto-classify: at this confidence level,")
    print(f"    the expected per-sample error is ≤25%. For dangerous")
    print(f"    conditions (FN cost={FN_COST_DANGEROUS}), even a 25% miss rate")
    print(f"    is acceptable ONLY for Tier 1 when combined with periodic")
    print(f"    audit sampling (5% of auto-classified sent for review).")
    print(f"    T=0.50 for review: images below 50% confidence show the")
    print(f"    model is uncertain between 2+ classes — radiologist needed.")
    print(f"    Below 50%: image may be OOD or poor quality — rescan first.")


def plot_triage_dashboard(triaged: pd.DataFrame, output_dir: Path):
    """Visual triage dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    tier_colors = {
        "Tier1_AutoClassify":    "#4CAF50",
        "Tier2_RadiologyReview": "#FF9800",
        "Tier3_RejectRescan":    "#F44336",
    }

    # 1. Tier distribution pie
    tier_counts = triaged["tier"].value_counts()
    axes[0, 0].pie(
        tier_counts.values,
        labels=[t.replace("_", "\n") for t in tier_counts.index],
        colors=[tier_colors[t] for t in tier_counts.index],
        autopct="%1.0f%%", startangle=90
    )
    axes[0, 0].set_title("Triage Tier Distribution", fontweight="bold")

    # 2. Confidence scatter coloured by tier
    x = np.arange(len(triaged))
    for tier, grp in triaged.groupby("tier"):
        axes[0, 1].scatter(grp.index, grp["confidence"],
                           c=tier_colors[tier], label=tier.replace("_", " "),
                           alpha=0.8, s=40)
    axes[0, 1].axhline(T_AUTO_CLASSIFY, color="green",  linestyle="--", lw=1.2, label=f"T1={T_AUTO_CLASSIFY}")
    axes[0, 1].axhline(T_REVIEW,        color="orange", linestyle="--", lw=1.2, label=f"T2={T_REVIEW}")
    axes[0, 1].set_title("Confidence per Image", fontweight="bold")
    axes[0, 1].set_xlabel("Image Index"); axes[0, 1].set_ylabel("Confidence")
    axes[0, 1].legend(fontsize=7)

    # 3. Tier 1 condition breakdown
    tier1 = triaged[triaged["tier"] == "Tier1_AutoClassify"]
    if len(tier1) > 0:
        cond_counts = tier1["predicted_condition"].value_counts()
        axes[1, 0].bar(cond_counts.index, cond_counts.values, color="#4CAF50", alpha=0.85)
        axes[1, 0].set_title("Tier 1 — Auto-Classified by Condition", fontweight="bold")
        axes[1, 0].set_ylabel("Count")
        plt.setp(axes[1, 0].get_xticklabels(), rotation=20, ha="right")
    else:
        axes[1, 0].text(0.5, 0.5, "No Tier 1 images", ha="center", va="center")
        axes[1, 0].set_title("Tier 1 — Auto-Classified by Condition", fontweight="bold")

    # 4. Expected FNR comparison
    tier1_data   = triaged[triaged["tier"] == "Tier1_AutoClassify"]
    fnr_cal      = compute_expected_fnr(tier1_data, calibrated=True)
    fnr_uncal    = compute_expected_fnr(tier1_data, calibrated=False)
    bars = axes[1, 1].bar(
        ["Well-Calibrated\nModel", "Overconfident\n(Uncalibrated)"],
        [fnr_cal * 100, fnr_uncal * 100],
        color=["#2196F3", "#E53935"], alpha=0.85
    )
    for bar, val in zip(bars, [fnr_cal, fnr_uncal]):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.3f}", ha="center", fontsize=9)
    axes[1, 1].set_ylabel("Expected FNR (%)")
    axes[1, 1].set_title("Expected False-Negative Rate\n(Tier 1 Auto-Classify)",
                         fontweight="bold")
    axes[1, 1].set_ylim(0, max(fnr_uncal * 100 * 1.4, 5))

    plt.suptitle("Triage Protocol Dashboard — 30 Unlabeled Images",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "fig10_triage_dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 7] Triage Protocol (Hard Sub-step)")

    # Load unlabeled predictions from Step 5
    preds_path = OUTPUT_DIR / "unlabeled_predictions.csv"
    try:
        predictions = pd.read_csv(preds_path)
    except FileNotFoundError:
        sys.exit("[ERROR] unlabeled_predictions.csv not found. Run step5 first.")

    # Assign tiers
    predictions["tier"] = predictions["confidence"].apply(assign_tier)

    print_triage_report(predictions)

    # Compute FNR
    tier1 = predictions[predictions["tier"] == "Tier1_AutoClassify"]
    fnr_calibrated   = compute_expected_fnr(tier1, calibrated=True)
    fnr_uncalibrated = compute_expected_fnr(tier1, calibrated=False)

    print(f"\n── Expected False-Negative Rate Analysis ───────────────────")
    print(f"  Tier 1 images : {len(tier1)}")
    print(f"  FNR (well-calibrated)  : {fnr_calibrated:.4f}  ({fnr_calibrated*100:.2f}%)")
    print(f"  FNR (overconfident)    : {fnr_uncalibrated:.4f} ({fnr_uncalibrated*100:.2f}%)")
    print(f"""
  Interpretation:
  If the model is well-calibrated, auto-classifying Tier 1 images
  carries a {fnr_calibrated*100:.1f}% expected FNR — acceptable for screening
  where true positives will be caught in subsequent clinical review.

  If the model is overconfident (common for small datasets with softmax),
  the true FNR rises to ~{fnr_uncalibrated*100:.1f}%. At this level, for dangerous
  conditions (COVID-19, Pneumonia), we recommend:
    (a) Applying temperature scaling to calibrate softmax outputs,
    (b) Lowering T_AUTO_CLASSIFY from 0.75 → 0.85,
    (c) Adding 5% random audit of auto-classified images.
""")

    plot_triage_dashboard(predictions, OUTPUT_DIR)

    # Save tiered report
    triage_path = OUTPUT_DIR / "triage_report.csv"
    predictions.to_csv(triage_path, index=False)
    print(f"[Saved] {triage_path}")
    print("\n[Step 7 Complete] ✓")


if __name__ == "__main__":
    main()
