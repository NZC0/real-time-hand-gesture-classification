# Rude Gestures Detector 🖕

Real-time detection of culturally offensive hand gestures via webcam.
The model runs **entirely in the browser** — no backend, no data leaves your device.

> 🎓 Educational project on ML & cross-cultural diversity.

---

## Demo

| Gesture | Culture(s) | Meaning |
|---|---|---|
| Middle finger | Universal | Classic disrespect |
| Reversed V | UK, Australia | "V for Victory" flipped = insult |
| Thumbs up | Iran, Iraq, Afghanistan | Equivalent to middle finger |
| Corna | Mediterranean, Latin America | "Your partner is unfaithful" |
| Crossed fingers | Vietnam | Vulgar gesture |
| OK sign | Brazil, Turkey | Body-orifice implication |
| Neutral | — | No offensive gesture detected |

---

## Architecture

```
Webcam → MediaPipe HandLandmarker (21 landmarks)
       → Landmark normalisation (JS, matches Python exactly)
       → ONNX Runtime Web → GestureMLP (63→128→64→32→7)
       → Class + confidence displayed as overlay
```

The MLP is tiny (~30 KB ONNX) and runs at 15–20 FPS on a mid-range laptop.

---

## Pipeline — Reproduce from scratch

### 1. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Collect data

```bash
python data/collect_landmarks.py
```

Controls:
- `1–7` — capture a frame for the corresponding class (see terminal for mapping)
- `d` — delete the last captured sample
- `q` — quit and save

**Target: ~300–500 samples per class.**
Samples are saved to `data/raw/landmarks.csv`.

### 3. Augment

```bash
python data/augment_landmarks.py \
    --input  data/raw/landmarks.csv \
    --output data/raw/landmarks_augmented.csv \
    --factor 5
```

This multiplies your dataset ~5× using:
- Gaussian noise (σ ∈ [0.01, 0.03])
- 2-D rotation (±15°)
- Scaling (0.9–1.1×)
- Horizontal mirror (simulates opposite hand)

### 4. Train

Open `notebooks/train.ipynb` in **Google Colab** (or Jupyter locally).

- Upload `data/raw/landmarks_augmented.csv` when prompted
- The notebook trains the MLP, shows training curves + confusion matrix,
  and saves the best checkpoint to `model/best_model.pt`

### 5. Export to ONNX

```bash
python model/export_onnx.py \
    --checkpoint model/best_model.pt \
    --output     web/public/model/gesture_mlp.onnx
```

Add `--quantize` for an INT8-quantized variant (smaller, slightly less accurate).

### 6. Run the web app

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173, click **Enable Camera**, and start gesturing.

### 7. Deploy to Vercel

```bash
cd web
npx vercel --prod
```

The `vercel.json` in `web/` already sets the required COOP/COEP headers for
SharedArrayBuffer (needed by multi-threaded ONNX WASM).

---

## Repository structure

```
rude-gestures-detector/
├── data/
│   ├── collect_landmarks.py     # Step 2 — webcam data collection
│   ├── augment_landmarks.py     # Step 3 — landmark augmentation
│   └── raw/                     # CSVs (gitignored)
├── notebooks/
│   └── train.ipynb              # Step 4 — Colab training notebook
├── model/
│   ├── gesture_mlp.py           # MLP architecture definition
│   └── export_onnx.py           # Step 5 — ONNX export + parity check
├── web/
│   ├── public/model/            # Place gesture_mlp.onnx here
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── App.css              # Dark minimal UI styles
│   │   ├── gestureClassifier.js # ONNX inference + normalisation
│   │   ├── handTracker.js       # MediaPipe Tasks Vision wrapper
│   │   └── gestureInfo.js       # Gesture metadata (name, emoji, description)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Normalisation — the critical detail

Both the Python data-collection script and the JavaScript inference code apply
**identical** normalisation before any model call:

1. **Translate** — subtract the wrist (landmark 0) from every point so the
   wrist becomes the origin.
2. **Scale** — divide all coordinates by the maximum Euclidean distance from
   the wrist to any other landmark (landmarks land roughly in `[-1, 1]`).

Any difference between the two implementations would silently destroy model
accuracy. The JS code (`web/src/gestureClassifier.js → normalizeLandmarks()`)
is a direct translation of the Python function
(`data/collect_landmarks.py → normalize_landmarks()`).

---

## Tech stack

| Layer | Technology |
|---|---|
| Landmark detection | MediaPipe Tasks Vision (`HandLandmarker`) |
| Browser inference | ONNX Runtime Web (WASM backend) |
| Frontend | React 18 + Vite |
| Training | PyTorch 2, scikit-learn |
| Deployment | Vercel (static) |

---

## License

MIT — do whatever you want, just don't use it to actually insult people.
