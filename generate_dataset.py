"""
generate_dataset.py
────────────────────────────────────────────────────────────────────────
Generates medical_imaging_meta.csv (520 rows) mimicking a real chest
X-ray metadata file with 5 conditions + 30 unlabeled rows.

Run once before the main notebook/scripts:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

# ── Constants ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_LABELED   = 490
N_UNLABELED = 30
OUTPUT_FILE = "medical_imaging_meta.csv"

CONDITIONS = ["Normal", "Pneumonia", "COVID-19", "Pleural Effusion", "Cardiomegaly"]

# Realistic class distribution (imbalanced — clinically authentic)
CLASS_WEIGHTS = [0.45, 0.25, 0.12, 0.10, 0.08]

HOSPITAL_SITES = ["AIIMS_Delhi", "PGIMER_Chandigarh", "NIMHANS", "KEM_Mumbai"]
EQUIPMENT      = ["GE_DR", "Siemens_CR", "Philips_DR", "Canon_CR"]

np.random.seed(RANDOM_SEED)


def generate_labeled_rows(n: int) -> pd.DataFrame:
    """Generate labeled metadata rows with realistic imaging features."""
    labels = np.random.choice(CONDITIONS, size=n, p=CLASS_WEIGHTS)

    # Image quality correlated weakly with equipment type
    equipment = np.random.choice(EQUIPMENT, size=n)
    quality_map = {"GE_DR": 0.85, "Siemens_CR": 0.80, "Philips_DR": 0.82, "Canon_CR": 0.75}
    base_quality = np.array([quality_map[e] for e in equipment])
    image_quality = np.clip(base_quality + np.random.normal(0, 0.05, n), 0.5, 1.0)

    # Simulated embedding dims (128-d PCA of ResNet features — representative)
    embed_base = {cond: np.random.randn(128) for cond in CONDITIONS}
    embeddings = np.vstack([
        embed_base[lbl] + np.random.randn(128) * 0.6 for lbl in labels
    ])
    embed_cols = {f"feat_{i:03d}": embeddings[:, i] for i in range(128)}

    df = pd.DataFrame({
        "image_id":       [f"IMG_{i:05d}" for i in range(n)],
        "condition":      labels,
        "hospital_site":  np.random.choice(HOSPITAL_SITES, size=n),
        "equipment":      equipment,
        "patient_age":    np.random.randint(18, 80, size=n),
        "patient_sex":    np.random.choice(["M", "F"], size=n),
        "image_quality":  np.round(image_quality, 3),
        "split":          np.where(np.random.rand(n) < 0.8, "train", "test"),
        "label":          labels,
    })
    for col, vals in embed_cols.items():
        df[col] = np.round(vals, 4)
    return df


def generate_unlabeled_rows(n: int, start_idx: int) -> pd.DataFrame:
    """Generate unlabeled rows (label = NaN)."""
    equipment = np.random.choice(EQUIPMENT, size=n)
    quality_map = {"GE_DR": 0.85, "Siemens_CR": 0.80, "Philips_DR": 0.82, "Canon_CR": 0.75}
    base_quality = np.array([quality_map[e] for e in equipment])
    image_quality = np.clip(base_quality + np.random.normal(0, 0.05, n), 0.5, 1.0)

    embeddings = np.random.randn(n, 128)
    embed_cols = {f"feat_{i:03d}": embeddings[:, i] for i in range(128)}

    df = pd.DataFrame({
        "image_id":      [f"IMG_{start_idx + i:05d}" for i in range(n)],
        "condition":     "Unlabeled",
        "hospital_site": np.random.choice(HOSPITAL_SITES, size=n),
        "equipment":     equipment,
        "patient_age":   np.random.randint(18, 80, size=n),
        "patient_sex":   np.random.choice(["M", "F"], size=n),
        "image_quality": np.round(image_quality, 3),
        "split":         "unlabeled",
        "label":         np.nan,
    })
    for col, vals in embed_cols.items():
        df[col] = np.round(vals, 4)
    return df


def main():
    labeled   = generate_labeled_rows(N_LABELED)
    unlabeled = generate_unlabeled_rows(N_UNLABELED, start_idx=N_LABELED)
    full      = pd.concat([labeled, unlabeled], ignore_index=True)
    full.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {OUTPUT_FILE}  ({len(full)} rows, "
          f"{N_LABELED} labeled + {N_UNLABELED} unlabeled)")
    print("\nClass distribution (labeled):")
    print(labeled["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
