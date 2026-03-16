"""
Data augmentation for landmark CSVs.

Applies landmark-space augmentations (NOT image augmentations) to multiply
the dataset by ~5x while preserving class balance.

Augmentations:
  - Gaussian noise     : slight jitter on all coords (sigma 0.01-0.03)
  - 2D rotation        : ±15° around the centroid of the hand
  - Random scaling     : 0.9x – 1.1x around the centroid
  - Horizontal mirror  : flip x-axis (simulates left-hand view of same gesture)

Usage:
    python data/augment_landmarks.py \
        --input  data/raw/landmarks.csv \
        --output data/raw/landmarks_augmented.csv \
        --factor 5
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
NOISE_SIGMA_RANGE = (0.01, 0.03)
ROTATION_MAX_DEG = 15.0
SCALE_MIN, SCALE_MAX = 0.9, 1.1

# MediaPipe landmark index groups (used for any future skeleton-aware ops)
LANDMARK_GROUPS = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation primitives  (all operate on (21, 3) numpy arrays)
# ──────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(pts: np.ndarray) -> np.ndarray:
    """Add small Gaussian noise to every coordinate."""
    sigma = random.uniform(*NOISE_SIGMA_RANGE)
    noise = np.random.normal(0, sigma, pts.shape)
    return pts + noise


def rotate_2d(pts: np.ndarray) -> np.ndarray:
    """Rotate the x-y plane around the hand centroid by a random angle."""
    angle_deg = random.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    centroid = pts.mean(axis=0)           # (3,)
    centered = pts - centroid             # translate to centroid

    rotated = centered.copy()
    rotated[:, 0] = cos_a * centered[:, 0] - sin_a * centered[:, 1]
    rotated[:, 1] = sin_a * centered[:, 0] + cos_a * centered[:, 1]
    # z is left unchanged (2-D rotation in the image plane)

    return rotated + centroid             # translate back


def random_scale(pts: np.ndarray) -> np.ndarray:
    """Scale the hand uniformly around its centroid."""
    factor = random.uniform(SCALE_MIN, SCALE_MAX)
    centroid = pts.mean(axis=0)
    return (pts - centroid) * factor + centroid


def horizontal_mirror(pts: np.ndarray) -> np.ndarray:
    """Mirror the hand along the x-axis (simulates the opposite hand).

    When normalised landmarks are mirrored we simply negate the x-axis.
    No landmark re-indexing is needed: MediaPipe assigns the same indices
    regardless of handedness (wrist=0, thumb tip=4, …), and the model
    learns gestures from relative finger positions, not absolute chirality.
    """
    mirrored = pts.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


# ──────────────────────────────────────────────────────────────────────────────
# Augment one row
# ──────────────────────────────────────────────────────────────────────────────

AUGMENTATIONS = [add_gaussian_noise, rotate_2d, random_scale, horizontal_mirror]


def augment_row(features: np.ndarray) -> list[np.ndarray]:
    """Apply all individual augmentations + one combined variant.

    Returns a list of augmented feature vectors (each shape (63,)).
    """
    pts = features.reshape(NUM_LANDMARKS, 3)
    results: list[np.ndarray] = []

    for aug_fn in AUGMENTATIONS:
        results.append(aug_fn(pts).flatten())

    # Combined: noise + rotation + scale
    combined = add_gaussian_noise(rotate_2d(random_scale(pts)))
    results.append(combined.flatten())

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment landmark CSV")
    parser.add_argument(
        "--input",
        default="data/raw/landmarks.csv",
        help="Path to the raw landmarks CSV",
    )
    parser.add_argument(
        "--output",
        default="data/raw/landmarks_augmented.csv",
        help="Output path for augmented CSV",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=5,
        help="Target multiplication factor (default 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading {input_path} …")
    df = pd.read_csv(input_path)

    feature_cols = [c for c in df.columns if c != "label"]
    label_col = "label"

    print(f"  {len(df)} original samples across {df[label_col].nunique()} classes.")
    print("  Class distribution:")
    print(df[label_col].value_counts().to_string(index=True))

    # ── Build augmented dataset ───────────────────────────────────────────────
    new_rows: list[dict] = []

    for _, row in df.iterrows():
        features = row[feature_cols].values.astype(float)
        label = row[label_col]

        # How many extra copies do we need?
        copies_needed = args.factor - 1  # the original counts as 1

        augmented = []
        while len(augmented) < copies_needed:
            augmented.extend(augment_row(features))

        for aug_features in augmented[:copies_needed]:
            new_row = dict(zip(feature_cols, aug_features))
            new_row[label_col] = label
            new_rows.append(new_row)

    aug_df = pd.DataFrame(new_rows, columns=df.columns)
    combined = pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"\nAugmented dataset saved to {output_path}")
    print(f"  Total samples : {len(combined)} (~{len(combined) / len(df):.1f}x original)")
    print("  Class distribution after augmentation:")
    print(combined[label_col].value_counts().to_string(index=True))


if __name__ == "__main__":
    main()
