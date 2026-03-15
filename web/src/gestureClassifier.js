/**
 * ONNX Runtime Web inference wrapper.
 *
 * Loads gesture_mlp.onnx, exposes a classify() method that:
 *   1. Normalises raw MediaPipe landmarks
 *   2. Runs ONNX inference
 *   3. Returns { className, confidence, probabilities }
 *
 * IMPORTANT: The normalisation here MUST be identical to the Python
 * implementation in data/collect_landmarks.py → normalize_landmarks().
 */

import * as ort from "onnxruntime-web";
import { CLASS_NAMES } from "./gestureInfo.js";

const MODEL_PATH = "/model/gesture_mlp.onnx";
const NUM_LANDMARKS = 21;
const NUM_FEATURES = 63; // 21 × 3

// ─────────────────────────────────────────────────────────────────────────────
// Normalisation (mirrors Python normalize_landmarks())
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Normalise 21 MediaPipe landmarks to be position / scale invariant.
 *
 * Steps (must match data/collect_landmarks.py exactly):
 *   1. Translate so wrist (landmark 0) is the origin.
 *   2. Scale by the max Euclidean distance from the wrist to any other point.
 *
 * @param {Array<{x: number, y: number, z: number}>} landmarks  - 21 points
 * @returns {Float32Array}  flat array of 63 normalised floats
 */
export function normalizeLandmarks(landmarks) {
  if (landmarks.length !== NUM_LANDMARKS) {
    throw new Error(`Expected ${NUM_LANDMARKS} landmarks, got ${landmarks.length}`);
  }

  const wrist = landmarks[0];

  // Translate: subtract wrist from every point
  const translated = landmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z,
  }));

  // Scale factor: max distance from wrist (now origin) to any other landmark
  let maxDist = 0;
  for (let i = 1; i < NUM_LANDMARKS; i++) {
    const { x, y, z } = translated[i];
    const dist = Math.sqrt(x * x + y * y + z * z);
    if (dist > maxDist) maxDist = dist;
  }
  if (maxDist < 1e-6) maxDist = 1.0; // avoid division by zero

  // Pack into flat Float32Array: [x0, y0, z0, x1, y1, z1, ...]
  const features = new Float32Array(NUM_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    features[i * 3 + 0] = translated[i].x / maxDist;
    features[i * 3 + 1] = translated[i].y / maxDist;
    features[i * 3 + 2] = translated[i].z / maxDist;
  }

  return features;
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────────────

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// ─────────────────────────────────────────────────────────────────────────────
// GestureClassifier class
// ─────────────────────────────────────────────────────────────────────────────

export class GestureClassifier {
  constructor() {
    /** @type {ort.InferenceSession | null} */
    this._session = null;
  }

  /** Load the ONNX model. Call once before classify(). */
  async load() {
    ort.env.wasm.wasmPaths = "/";
    this._session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    console.log("[GestureClassifier] Model loaded.");
  }

  /**
   * Classify a hand pose.
   *
   * @param {Array<{x: number, y: number, z: number}>} landmarks - 21 raw points
   * @returns {{ className: string, confidence: number, probabilities: number[] }}
   */
  async classify(landmarks) {
    if (!this._session) throw new Error("Model not loaded. Call load() first.");

    const features = normalizeLandmarks(landmarks);
    const inputTensor = new ort.Tensor("float32", features, [1, NUM_FEATURES]);

    const outputs = await this._session.run({ landmarks: inputTensor });
    const logits = Array.from(outputs["logits"].data);
    const probabilities = softmax(logits);

    let bestIdx = 0;
    let bestProb = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > bestProb) {
        bestProb = probabilities[i];
        bestIdx = i;
      }
    }

    return {
      className: CLASS_NAMES[bestIdx] ?? "neutral",
      confidence: bestProb,
      probabilities,
    };
  }

  get isLoaded() {
    return this._session !== null;
  }
}
