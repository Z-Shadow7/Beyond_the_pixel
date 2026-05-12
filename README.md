# Beyond the Pixel

### 🔍 Deepfake Detection System (DFDC + FastAPI + React)

A full-stack AI system that detects **deepfake images and videos** using a deep learning model trained on the DFDC dataset. The backend is built with FastAPI and PyTorch, and the frontend is built with React + TypeScript.

---

# 🚀 Features

- 📷 Image deepfake detection
- 🎥 Video deepfake detection (frame-based analysis)
- 🧠 DFDC-trained EfficientNet B7 model
- 👤 Face detection using MTCNN
- ⚡ FastAPI backend for inference
- 🌐 React + TypeScript frontend
- 🎯 Confidence score + prediction output

---

# 🏗️ Project Structure


```
project/
│
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── image_detector.py
│   │   ├── video_detector.py
│   ├── uploads/
|
├── dfdc_deepfake_challenge/
│   │   ├── training/
│   │   ├── weights/
|
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── styles.css
│
└── README.md

````

---

# ⚙️ Backend Setup (FastAPI)

## 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
````
### clone dfdc_deepfake_challenge
``` https://github.com/selimsef/dfdc_deepfake_challenge.git ```
### Download final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
``` https://github.com/selimsef/dfdc_deepfake_challenge/releases ```
### Paste it in 
``` ./dfdc_deepfake_challenge/weights/ ```

---

## 2. Run backend server

IMPORTANT (fix import paths):

```bash
PYTHONPATH=. uvicorn app:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## 3. API Endpoints

### 📷 Image Detection

```
POST /detect-image
```

Request:

* form-data → file (image)

Response:

```json
{
  "result": {
    "success": true,
    "prediction": "LIKELY REAL",
    "deepfake_score": 0.12
  }
}
```

---

### 🎥 Video Detection

```
POST /detect-video
```

Request:

* form-data → file (video)

Response:

```json
{
  "result": {
    "success": true,
    "prediction": "LIKELY FAKE",
    "deepfake_score": 0.78,
    "frames_analyzed": 32
  }
}
```

---

# 🌐 Frontend Setup (React + TypeScript)

## 1. Install dependencies

```bash
cd frontend
npm install
```

---

## 2. Run frontend

```bash
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# 🎨 UI Features

* 📷 Separate image upload section
* 🎥 Separate video upload section
* ⚡ Real-time inference
* 📊 Prediction + confidence score
* ⏳ Loading state during processing

---

# 🧠 Model Details

* Architecture: EfficientNet B7 (`tf_efficientnet_b7_ns`)
* Dataset: DFDC (DeepFake Detection Challenge)
* Face detection: MTCNN
* Input: Cropped face images (380×380)

---

# 🔄 Pipeline Flow

## Image Pipeline

```
Image Upload
→ Face Detection (MTCNN)
→ Preprocessing
→ DFDC Model Inference
→ Prediction Output
```

---

## Video Pipeline

```
Video Upload
→ Frame Extraction (OpenCV)
→ Face Detection per frame
→ Model inference per frame
→ Score averaging
→ Final prediction
```

---

# ⚠️ Known Limitations

* Best performance on DFDC-style videos
* Weak on GAN-generated static images
* Requires visible face in input
* CPU inference is slow for video processing

---

# 🚀 Future Improvements

* GPU acceleration
* Real-time webcam detection
* Frame-level heatmaps
* Model ensemble (image + video models)
* Docker deployment
* Cloud hosting (AWS / Render)

---

# 🧰 Tech Stack

* FastAPI
* PyTorch
* OpenCV
* facenet-pytorch (MTCNN)
* Albumentations
* React + TypeScript (Vite)

---

# 👨‍💻 Author

Full-stack deepfake detection system built using DFDC-trained models and FastAPI.

---

# 📄 License

For educational and research purposes only.




