/**
 * Rude Gestures Detector — main application component.
 *
 * Flow:
 *   1. Camera permission → getUserMedia
 *   2. Load MediaPipe HandTracker + GestureClassifier
 *   3. rAF loop: detect landmarks → classify → display
 *      - draws bounding box around detected hand
 *      - shows a reference image panel with an arrow
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import { HandTracker } from "./handTracker.js";
import { GestureClassifier } from "./gestureClassifier.js";
import { getGestureInfo } from "./gestureInfo.js";
import "./App.css";

const CONFIDENCE_THRESHOLD = 0.70;

// ─────────────────────────────────────────────────────────────────────────────
// Canvas drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawLandmarks(ctx, landmarks, w, h) {
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ffa3";
    ctx.fill();
  }
}

/**
 * Draw a bounding box (corner brackets style) around the detected hand.
 * Returns the box { x, y, w, h } in canvas pixels for external use.
 */
function drawBoundingBox(ctx, landmarks, canvasW, canvasH, accentColor = "#00ffa3") {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }

  const PAD_X = (maxX - minX) * 0.22;
  const PAD_Y = (maxY - minY) * 0.22;
  const x = Math.max(0, (minX - PAD_X) * canvasW);
  const y = Math.max(0, (minY - PAD_Y) * canvasH);
  const w = Math.min(canvasW - x, (maxX - minX + 2 * PAD_X) * canvasW);
  const h = Math.min(canvasH - y, (maxY - minY + 2 * PAD_Y) * canvasH);

  const cornerLen = Math.min(w, h) * 0.18;
  const r = 4; // corner radius of the bracket tip

  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.shadowColor = accentColor;
  ctx.shadowBlur = 8;

  // Draw the 4 corner brackets
  const corners = [
    { ox: x,     oy: y,     dx: 1,  dy: 1  },
    { ox: x + w, oy: y,     dx: -1, dy: 1  },
    { ox: x + w, oy: y + h, dx: -1, dy: -1 },
    { ox: x,     oy: y + h, dx: 1,  dy: -1 },
  ];

  for (const { ox, oy, dx, dy } of corners) {
    ctx.beginPath();
    ctx.moveTo(ox + dx * cornerLen, oy);
    ctx.lineTo(ox + dx * r, oy);
    ctx.arcTo(ox, oy, ox, oy + dy * r, r);
    ctx.lineTo(ox, oy + dy * cornerLen);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
  return { x, y, w, h };
}

// ─────────────────────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Side panel with a pointing arrow and the reference image for the gesture.
 * Place images at: public/gestures/{className}.jpg (or .png)
 */
function GestureReference({ result }) {
  const [imgError, setImgError] = useState(false);

  // Reset error state when gesture changes
  useEffect(() => setImgError(false), [result?.className]);

  if (!result) {
    return (
      <div className="reference-panel reference-panel--idle">
        <div className="ref-arrow">←</div>
        <div className="ref-placeholder">
          <span>✋</span>
          <p>Show a gesture</p>
        </div>
      </div>
    );
  }

  const { info, className, confidence } = result;
  const pct = Math.round(confidence * 100);

  return (
    <div className="reference-panel" style={{ "--ref-accent": info.color }}>
      <div className="ref-arrow">←</div>
      <div className="ref-card">
        <div className="ref-label">Reference</div>
        <div className="ref-image-wrap">
          {!imgError ? (
            <img
              src={`/gestures/${className}.jpg`}
              alt={info.name}
              className="ref-image"
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="ref-image-fallback">{info.emoji}</div>
          )}
        </div>
        <div className="ref-name">{info.name}</div>
        <div className="ref-confidence-bar">
          <div className="ref-confidence-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="ref-confidence-label">{pct}% match</div>
        {info.cultures.length > 0 && (
          <div className="ref-cultures">
            {info.cultures.map((c) => (
              <span key={c} className="culture-tag">{c}</span>
            ))}
          </div>
        )}
        <p className="ref-description">{info.description}</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const trackerRef = useRef(new HandTracker());
  const classifierRef = useRef(new GestureClassifier());
  const rafRef = useRef(null);

  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null);

  const init = useCallback(async () => {
    setStatus("loading");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await Promise.all([trackerRef.current.load(), classifierRef.current.load()]);
      setStatus("running");
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message ?? String(err));
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    if (status !== "running") return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let lastClassifyTime = 0;
    const CLASSIFY_INTERVAL_MS = 50;

    const loop = async () => {
      if (video.readyState < 2) { rafRef.current = requestAnimationFrame(loop); return; }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw mirrored video
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0);
      ctx.restore();

      const landmarks = trackerRef.current.detect(video);

      if (landmarks) {
        drawLandmarks(ctx, landmarks, canvas.width, canvas.height);
        drawBoundingBox(ctx, landmarks, canvas.width, canvas.height);
      }

      const now = performance.now();
      if (landmarks && now - lastClassifyTime > CLASSIFY_INTERVAL_MS) {
        lastClassifyTime = now;
        try {
          const pred = await classifierRef.current.classify(landmarks);
          if (pred.confidence >= CONFIDENCE_THRESHOLD) {
            setResult({ className: pred.className, confidence: pred.confidence, info: getGestureInfo(pred.className) });
          } else {
            setResult(null);
          }
        } catch (e) { console.warn("Classify error:", e); }
      } else if (!landmarks) {
        setResult(null);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [status]);

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo"><span className="logo-accent">RUDE</span> GESTURES</h1>
        <p className="subtitle">Real-time cross-cultural gesture detection — MediaPipe + ONNX</p>
      </header>

      <main className="stage">
        {status === "idle" && (
          <div className="splash">
            <div className="splash-icon">✋</div>
            <h2>Gesture Detector</h2>
            <p>Recognises 6 culturally offensive hand gestures using your webcam and a neural network running entirely in your browser.</p>
            <p className="disclaimer">⚠️ Educational project about ML &amp; cultural diversity. No data leaves your device.</p>
            <button className="btn-primary" onClick={init}>Enable Camera</button>
          </div>
        )}

        {status === "loading" && (
          <div className="splash">
            <div className="spinner" />
            <p>Loading models…</p>
          </div>
        )}

        {status === "error" && (
          <div className="splash error">
            <div className="splash-icon">⚠️</div>
            <h2>Something went wrong</h2>
            <p>{errorMsg}</p>
            <button className="btn-primary" onClick={init}>Retry</button>
          </div>
        )}

        {status === "running" && (
          <div className="running-layout">
            <div className="camera-container">
              <video ref={videoRef} className="source-video" muted playsInline />
              <canvas ref={canvasRef} className="overlay-canvas" />
              {!result && (
                <div className="no-gesture-hint">Show a hand to the camera</div>
              )}
            </div>
            <GestureReference result={result} />
          </div>
        )}

        {/* Pre-mount video/canvas so refs are always valid */}
        {status !== "running" && (
          <div style={{ display: "none" }}>
            <video ref={videoRef} muted playsInline />
            <canvas ref={canvasRef} />
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          Built with{" "}
          <a href="https://mediapipe.dev" target="_blank" rel="noreferrer">MediaPipe</a>
          {" · "}
          <a href="https://onnxruntime.ai" target="_blank" rel="noreferrer">ONNX Runtime Web</a>
          {" · "}
          <a href="https://pytorch.org" target="_blank" rel="noreferrer">PyTorch</a>
        </p>
        <p className="disclaimer-footer">🎓 Educational project on ML &amp; cross-cultural diversity.</p>
      </footer>
    </div>
  );
}
