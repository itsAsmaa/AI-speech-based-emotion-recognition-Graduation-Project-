<<<<<<< HEAD
# 🎙️ Voice Emotion Detection — Local VS Code Setup

Real-time speech emotion analyser built on RAVDESS · SVM / Ensemble · 8 emotions.

---

## Project Structure

```
emotion_project/
│
├── config.py              ← All paths & constants (edit here if needed)
├── train.py               ← Full training pipeline
├── realtime_app.py        ← Gradio real-time app (run this after training)
├── requirements.txt       ← All dependencies
│
├── src/
│   └── feature_extractor.py   ← Shared feature extraction module
│
├── data/
│   └── ravdess/           ← RAVDESS .wav files (auto-downloaded)
│
├── model/                 ← Saved model artifacts (created by train.py)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── emotion_labels.json
│
└── outputs/               ← Confusion matrix plots
```

---

## ⚙️ Setup (do this once)

### 1. Open the project in VS Code
```
File → Open Folder → select the emotion_project folder
```

### 2. Create a virtual environment
Open the **VS Code terminal** (`Ctrl + `` ` ```) and run:

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
1. Download RAVDESS (~600 MB) into `data/ravdess/` — **once only**
2. Extract features from all audio files
3. Run GridSearchCV to find the best SVM
4. Train an SVM + Random Forest ensemble
5. Save `model.pkl`, `scaler.pkl`, `selector.pkl`, `emotion_labels.json` into `model/`
6. Save a confusion matrix plot to `outputs/`

Training takes **15–30 minutes** on a typical laptop (the GridSearch is the slow part).

---

## 🚀 Run the Real-Time App

After training is done:

```bash
python realtime_app.py
```

A browser tab opens automatically at **http://localhost:7860**

- Click the microphone button → speak for 2–4 seconds → stop recording
- The app auto-analyses on stop, or click **Analyse Emotion**
- You can also drag-and-drop a `.wav` / `.mp3` file

---

## 🔄 Using Your Already-Trained Model (from Colab)

If you already trained in Colab and want to use that model locally:

1. In Colab, add and run this cell to download your artifacts:
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

   # Zip and download
   with zipfile.ZipFile('/content/emotion_model.zip','w') as zf:
       for p in Path('/content/emotion_model').iterdir():
           zf.write(p, p.name)
   files.download('/content/emotion_model.zip')
   ```

2. Unzip the downloaded file into your local `model/` folder:
   ```
   emotion_project/model/model.pkl
   emotion_project/model/scaler.pkl
   emotion_project/model/selector.pkl        ← if it exists
   emotion_project/model/emotion_labels.json
   ```

3. Run the app directly — no training needed:
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

**Recommended VS Code Extensions:**
- Python (Microsoft)
- Pylance
- Jupyter (if you want to run the original `.ipynb` notebook)

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
=======
# no-face-
>>>>>>> f78e82f7ab15c6a685e03ce7d46dfe0c22dd3ef6
