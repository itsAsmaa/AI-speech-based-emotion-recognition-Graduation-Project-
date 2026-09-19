# 🎙️😊 AI Speech-Based Emotion Recognition

Real-time **speech + face** emotion recognition system — SVM / Random Forest ensemble, trained on RAVDESS, cross-validated on TESS, EmoDB, and SAVEE. Built as a graduation project (ENCS5300) at Birzeit University.

---

## 🎬 Demo Videos

See it running before you set anything up:

| Demo | Description |
|------|-------------|
| [🔴 Live voice + webcam](./Live%20voice%2Bwebcam.mp4) | Real-time multimodal detection using microphone + camera feed |
| [🎥 Pre-recorded video](./pre%20recorded%20video.mp4) | Face + speech emotion detection run on a pre-recorded video file |
| [🎤 Pre-recorded voice](./pre%20recorded%20voice.mp4) | Speech-only emotion detection on a pre-recorded audio clip |

> Click a link to stream it directly from GitHub, or clone the repo and open the `.mp4` files locally.

<details>
<summary>Prefer an inline player? (click to expand)</summary>

```html
<video src="./Live voice+webcam.mp4" controls width="600"></video>
```

GitHub renders this as an inline playable video on the repo page for files tracked with Git LFS or under the size limit — if it doesn't render for you, use the plain links above instead.

</details>

---

## 🧠 What This Is

A real-time ensemble pipeline that fuses:
- **Speech**: 162 acoustic features (MFCCs, pitch, energy, etc.) selected via `SelectKBest`
- **Face**: facial expression features from webcam/video frames

fused through an **SVM + Random Forest ensemble**, trained on **RAVDESS** and evaluated cross-corpus on **TESS**, **EmoDB**, and **SAVEE** for generalization.

8 emotions detected: Neutral · Calm · Happy · Sad · Angry · Fear · Disgust · Surprise

---

## 📁 Project Structure

```
emotion_project/
│
├── config.py              ← All paths & constants
├── train.py                ← Full training pipeline
├── realtime_app.py         ← Gradio real-time app
├── requirements.txt        ← Dependencies
│
├── src/
│   └── feature_extractor.py   ← Shared feature extraction module
│
├── data/
│   └── ravdess/            ← RAVDESS .wav files (auto-downloaded)
│
├── model/                  ← Saved model artifacts (created by train.py)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── emotion_labels.json
│
└── outputs/                ← Confusion matrix plots
```

---

## ⚙️ Setup

### 1. Open the project in VS Code
```
File → Open Folder → select the emotion_project folder
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python train.py
```

This will:
1. Download RAVDESS (~600 MB) into `data/ravdess/` — once only
2. Extract features from all audio files
3. Run GridSearchCV to find the best SVM
4. Train an SVM + Random Forest ensemble
5. Save `model.pkl`, `scaler.pkl`, `selector.pkl`, `emotion_labels.json` into `model/`
6. Save a confusion matrix plot to `outputs/`

Training takes **15–30 minutes** on a typical laptop (GridSearch is the slow part).

---

## 🚀 Run the Real-Time App

```bash
python realtime_app.py
```

A browser tab opens automatically at **http://localhost:7860**

- Click the microphone button → speak for 2–4 seconds → stop recording
- The app auto-analyses on stop, or click **Analyse Emotion**
- You can also drag-and-drop a `.wav` / `.mp4` file — try the sample recordings above!

---

## 🔄 Using an Already-Trained Model (e.g. from Colab)

1. In Colab, run:
   ```python
   from google.colab import files
   import zipfile, os, joblib, json

   os.makedirs('/content/emotion_model', exist_ok=True)
   joblib.dump(final_model, '/content/emotion_model/model.pkl')
   joblib.dump(scaler,      '/content/emotion_model/scaler.pkl')
   try:   joblib.dump(selector, '/content/emotion_model/selector.pkl')
   except NameError: pass
   with open('/content/emotion_model/emotion_labels.json','w') as f:
       json.dump(emotion_labels, f)

   with zipfile.ZipFile('/content/emotion_model.zip','w') as zf:
       for p in Path('/content/emotion_model').iterdir():
           zf.write(p, p.name)
   files.download('/content/emotion_model.zip')
   ```

2. Unzip into your local `model/` folder:
   ```
   emotion_project/model/model.pkl
   emotion_project/model/scaler.pkl
   emotion_project/model/selector.pkl        ← if it exists
   emotion_project/model/emotion_labels.json
   ```

3. Run directly — no training needed:
   ```bash
   python realtime_app.py
   ```

---

## 🛠️ VS Code Tips

| Action | Shortcut |
|--------|----------|
| Open terminal | Ctrl + `` ` `` |
| Run current Python file | F5 or click ▶ |
| Select Python interpreter | Ctrl+Shift+P → "Python: Select Interpreter" → choose `.venv` |
| Stop the app | Ctrl+C in terminal |

**Recommended VS Code Extensions:** Python (Microsoft), Pylance, Jupyter

---

## 📊 Emotions Detected

| Code | Emotion | Emoji |
|------|---------|-------|
| 01 | Neutral  | 😐 |
| 02 | Calm     | 😌 |
| 03 | Happy    | 😄 |
| 04 | Sad      | 😢 |
| 05 | Angry    | 😠 |
| 06 | Fear     | 😨 |
| 07 | Disgust  | 🤢 |
| 08 | Surprise | 😲 |

---

## ⚠️ Troubleshooting

**`ModuleNotFoundError: No module named 'librosa'`**
→ Make sure your `.venv` is activated and `pip install -r requirements.txt` ran successfully.

**`FileNotFoundError: Missing model files`**
→ Run `python train.py` first, or copy your Colab model files into the `model/` folder.

**Microphone not detected in browser**
→ Allow microphone access when the browser asks. On Windows, also check Privacy Settings → Microphone.

**App opens but shows blank page**
→ Manually go to http://localhost:7860 in your browser.
