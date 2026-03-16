/**
 * MediaPipe Hands wrapper.
 *
 * Abstracts the MediaPipe Tasks Vision API into a simple interface:
 *   const tracker = new HandTracker();
 *   await tracker.load();
 *   const landmarks = tracker.detect(videoElement);
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

export class HandTracker {
  constructor() {
    /** @type {HandLandmarker | null} */
    this._landmarker = null;
    this._lastVideoTime = -1;
  }

  /** Initialise the MediaPipe HandLandmarker. */
  async load() {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    this._landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_ASSET_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });

    console.log("[HandTracker] MediaPipe HandLandmarker loaded.");
  }

  /**
   * Run hand landmark detection on a video frame.
   *
   * @param {HTMLVideoElement} video
   * @returns {Array<{x: number, y: number, z: number}> | null}
   *   21 normalised landmarks for the first detected hand, or null.
   */
  detect(video) {
    if (!this._landmarker) return null;
    if (video.currentTime === this._lastVideoTime) return null;

    this._lastVideoTime = video.currentTime;
    const result = this._landmarker.detectForVideo(video, performance.now());

    if (!result.landmarks || result.landmarks.length === 0) return null;

    // Return the first hand's 21 landmarks as plain objects
    return result.landmarks[0].map(({ x, y, z }) => ({ x, y, z }));
  }

  get isLoaded() {
    return this._landmarker !== null;
  }
}
