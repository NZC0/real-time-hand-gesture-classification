"""
Data collection script for rude gesture detection.

Usage:
    python data/collect_landmarks.py

Controls:
    1 → middle_finger
    2 → reversed_v
    3 → thumbs_up
    4 → corna
    5 → crossed_fingers
    6 → ok_sign
    7 → neutral
    q → quit and save
    d → delete last sample

The collected landmarks are saved to data/raw/landmarks.csv.
Normalization: translate to wrist (landmark 0) as origin, scale by max
distance from wrist to any other landmark. This makes features invariant
to hand position and size in the frame.
"""

import csv
import time
import urllib.request

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CLASSES = {
    "1": "middle_finger",
    "2": "reversed_v",
    "3": "thumbs_up",
    "4": "corna",
    "5": "crossed_fingers",
    "6": "ok_sign",
    "7": "neutral",
}

NUM_LANDMARKS = 21
OUTPUT_PATH = Path(__file__).parent / "raw" / "landmarks.csv"
CSV_HEADERS = (
    [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
    + ["label"]
)

# MediaPipe Tasks model — downloaded once to data/models/
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand skeleton connections (MediaPipe standard)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# ──────────────────────────────────────────────────────────────────────────────
# Model download
# ──────────────────────────────────────────────────────────────────────────────

def ensure_model() -> None:
    """Download the hand landmarker model if not already present."""
    if MODEL_PATH.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  Done.")


# ──────────────────────────────────────────────────────────────────────────────
# Normalization (MUST match gestureClassifier.js)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks: list[tuple[float, float, float]]) -> list[float]:
    """Normalize 21 hand landmarks to be position/scale invariant.

    Steps:
      1. Translate so that the wrist (landmark 0) is the origin.
      2. Scale by the maximum Euclidean distance from the wrist to any
         other landmark. Result lives roughly in [-1, 1].

    Returns a flat list of 63 floats: [x0, y0, z0, x1, y1, z1, ...].
    """
    wrist = np.array(landmarks[0])
    pts = np.array(landmarks) - wrist          # translate to wrist origin

    # Scale factor: max distance from wrist to any other landmark
    distances = np.linalg.norm(pts[1:], axis=1)
    scale = distances.max()
    if scale < 1e-6:
        scale = 1.0                            # avoid division by zero

    pts /= scale
    return pts.flatten().tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_hand(
    frame: np.ndarray,
    landmarks: list[tuple[float, float, float]],
) -> None:
    """Draw hand skeleton (connections + keypoints) on the frame."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, z in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)

    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 163), -1, cv2.LINE_AA)


def draw_counter(frame: np.ndarray, counts: dict[str, int]) -> None:
    """Overlay per-class sample counts on the frame."""
    h, w = frame.shape[:2]
    panel_x = w - 260
    cv2.rectangle(frame, (panel_x - 10, 10), (w - 10, 20 + 30 * len(CLASSES)), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x - 10, 10), (w - 10, 20 + 30 * len(CLASSES)), (50, 50, 50), 1)

    for idx, (key, name) in enumerate(CLASSES.items()):
        y = 35 + idx * 30
        count = counts.get(name, 0)
        color = (0, 255, 120) if count >= 300 else (255, 200, 0)
        cv2.putText(
            frame,
            f"[{key}] {name}: {count}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_status(frame: np.ndarray, message: str, color=(255, 255, 255)) -> None:
    """Overlay a status message at the bottom of the frame."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, message, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_model()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Track how many samples we already have (resume from existing file)
    counts: dict[str, int] = {name: 0 for name in CLASSES.values()}
    existing_rows: list[list] = []

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) == len(CSV_HEADERS):
                    label = row[-1]
                    if label in counts:
                        counts[label] += 1
                    existing_rows.append(row)
        print(f"Resuming — loaded {len(existing_rows)} existing samples.")

    # Open CSV for appending
    csv_file = open(OUTPUT_PATH, "w", newline="")  # noqa: WPS515
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    for row in existing_rows:
        writer.writerow(row)

    # MediaPipe Tasks setup
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check that it is connected.")

    last_label: str | None = None
    flash_until: float = 0.0

    print("Press Shift+1 to Shift+7 to capture a sample for the corresponding class.")
    print("Press 'q' to quit and save. Press 'd' to delete the last sample.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        hand_detected = bool(result.hand_landmarks)
        landmarks_raw: list[tuple[float, float, float]] = []

        if hand_detected:
            landmarks_raw = [
                (lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]
            ]
            draw_hand(frame, landmarks_raw)

        # Capture sample
        if key != 255 and chr(key) in CLASSES and hand_detected:
            label = CLASSES[chr(key)]
            normalized = normalize_landmarks(landmarks_raw)
            row = normalized + [label]
            writer.writerow(row)
            csv_file.flush()
            counts[label] += 1
            last_label = label
            flash_until = time.time() + 0.4
            print(f"  Saved [{label}] — total: {counts[label]}")

        # Delete last sample
        if key == ord("d") and last_label is not None:
            csv_file.close()
            rows_to_keep = []
            deleted = False
            with open(OUTPUT_PATH, newline="") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                all_rows = list(reader)
            for row in reversed(all_rows):
                if not deleted and len(row) > 0 and row[-1] == last_label:
                    deleted = True
                    counts[last_label] = max(0, counts[last_label] - 1)
                else:
                    rows_to_keep.append(row)
            rows_to_keep.reverse()
            csv_file = open(OUTPUT_PATH, "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)
            for row in rows_to_keep:
                writer.writerow(row)
            csv_file.flush()
            print(f"  Deleted last [{last_label}] sample — total: {counts[last_label]}")

        # Draw UI
        draw_counter(frame, counts)

        if time.time() < flash_until:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            draw_status(frame, f"Captured: {last_label}", color=(0, 255, 0))
        elif not hand_detected:
            draw_status(frame, "No hand detected", color=(0, 100, 255))
        else:
            draw_status(frame, "Shift+1-7 to capture | q to quit | d to delete last")

        cv2.imshow("Rude Gestures — Data Collection", frame)

    cap.release()
    csv_file.close()
    detector.close()
    cv2.destroyAllWindows()

    total = sum(counts.values())
    print(f"\nDone! Saved {total} total samples to {OUTPUT_PATH}")
    for name, count in counts.items():
        status = "OK" if count >= 300 else "NEEDS MORE"
        print(f"  {name}: {count} [{status}]")


if __name__ == "__main__":
    main()
