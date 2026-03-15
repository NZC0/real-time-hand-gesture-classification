/**
 * Rude Gestures Detector — main application component.
 *
 * Flow:
 *   1. Camera permission → getUserMedia
 *   2. Load MediaPipe HandTracker + GestureClassifier
 *   3. rAF loop: detect landmarks → classify → display
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import { HandTracker } from "./handTracker.js";
import { GestureClassifier } from "./gestureClassifier.js";
import { getGestureInfo } from "./gestureInfo.js";
import "./App.css";

const CONFIDENCE_THRESHOLD = 0.70;

// Draw skeleton on canvas overlay
function drawLandmarks(ctx, landmarks, width, height) {
  if (!landmarks) return;

  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    ctx.stroke();
  }

  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ffa3";
    ctx.fill();
  }
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const trackerRef = useRef(new HandTracker());
  const classifierRef = useRef(new GestureClassifier());
  const rafRef = useRef(null);

  const [status, setStatus] = useState("idle"); // idle | loading | running | error
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null); // { className, confidence, info }

  // ── Initialisation ────────────────────────────────────────────────────────
  const init = useCallback(async () => {
    setStatus("loading");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      await Promise.all([
        trackerRef.current.load(),
        classifierRef.current.load(),
      ]);

      setStatus("running");
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message ?? String(err));
      setStatus("error");
    }
  }, []);

  // ── Main loop ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (status !== "running") return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    let lastClassifyTime = 0;
    const CLASSIFY_INTERVAL_MS = 50; // ~20 fps inference cap

    const loop = async () => {
      if (video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw mirrored video
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0);
      ctx.restore();

      const landmarks = trackerRef.current.detect(video);
      drawLandmarks(ctx, landmarks, canvas.width, canvas.height);

      const now = performance.now();
      if (landmarks && now - lastClassifyTime > CLASSIFY_INTERVAL_MS) {
        lastClassifyTime = now;
        try {
          const pred = await classifierRef.current.classify(landmarks);
          if (pred.confidence >= CONFIDENCE_THRESHOLD) {
            setResult({
              className: pred.className,
              confidence: pred.confidence,
              info: getGestureInfo(pred.className),
            });
          } else {
            setResult(null);
          }
        } catch (e) {
          console.warn("Classify error:", e);
        }
      } else if (!landmarks) {
        setResult(null);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [status]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">
          <span className="logo-accent">RUDE</span> GESTURES
        </h1>
        <p className="subtitle">Real-time cross-cultural gesture detection — powered by MediaPipe + ONNX</p>
      </header>

      <main className="stage">
        {status === "idle" && (
          <div className="splash">
            <div className="splash-icon">✋</div>
            <h2>Gesture Detector</h2>
            <p>
              This app recognises 6 culturally offensive hand gestures in real
              time using your webcam and a lightweight neural network running
              entirely in your browser.
            </p>
            <p className="disclaimer">
              ⚠️ Educational project about ML &amp; cultural diversity.
              No data leaves your device.
            </p>
            <button className="btn-primary" onClick={init}>
              Enable Camera
            </button>
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

        <div className={`camera-container ${status === "running" ? "visible" : "hidden"}`}>
          {/* Hidden video element — used as MediaPipe source */}
          <video ref={videoRef} className="source-video" muted playsInline />
          {/* Canvas shows mirrored video + landmark overlay */}
          <canvas ref={canvasRef} className="overlay-canvas" />

          {result && (
            <ResultOverlay result={result} />
          )}

          {status === "running" && !result && (
            <div className="no-gesture-hint">Show a hand to the camera</div>
          )}
        </div>
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
        <p className="disclaimer-footer">
          🎓 Educational project on ML &amp; cross-cultural diversity. Not a surveillance tool.
        </p>
      </footer>
    </div>
  );
}

function ResultOverlay({ result }) {
  const { info, confidence } = result;
  const pct = Math.round(confidence * 100);
  return (
    <div className="result-overlay" style={{ "--accent": info.color }}>
      <div className="result-emoji">{info.emoji}</div>
      <div className="result-name">{info.name}</div>
      <div className="confidence-bar">
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="confidence-label">{pct}% confidence</div>
      {info.cultures.length > 0 && (
        <div className="cultures">
          {info.cultures.map((c) => (
            <span key={c} className="culture-tag">{c}</span>
          ))}
        </div>
      )}
      <p className="result-description">{info.description}</p>
    </div>
  );
}
