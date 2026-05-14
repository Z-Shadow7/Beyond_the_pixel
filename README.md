# Beyond the Pixel

### 🔍 Deepfake Detection System (Multi-Model + FastAPI + React)

A full-stack AI system that detects **deepfake images and videos** using multiple state-of-the-art detection models including **Effort (ICML 2025)** and **GenConViT**. The backend is built with FastAPI and PyTorch, and the frontend is built with React + TypeScript.

---

# 🚀 Features

* 📷 Image deepfake detection
* 🎥 Video deepfake detection (frame-based analysis)
* 🧠 Multiple detection models:
  - **Effort** (CLIP-L14 + Orthogonal Subspace Decomposition)
  - **GenConViT** (Network A: encoder-decoder, Network B: VAE)
* 👤 Face detection using MTCNN
* ⚡ FastAPI backend for inference
* 🌐 React + TypeScript frontend
* 🎯 Confidence scores + detailed predictions

---

# 🏗️ Project Structure

```text
Beyond_the_pixel/
│
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_detector.py
│   │   ├── video_detector.py
│   │   └── genconvit_detector.py
│   ├── utils/
│   │   ├── extract_frames.py
│   │   └── preprocess.py
│   ├── uploads/
│   └── outputs/
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── App.css
│   │   ├── styles.css
│   │   ├── main.tsx
│   │   └── index.css
│   ├── vite.config.ts
│   ├── package.json
│   └── tsconfig.json
│
├── Effort-AIGI-Detection-main/          # Effort model repository
│   ├── DeepfakeBench/
│   │   ├── training/
│   │   │   └── weights/
│   │   └── preprocessing/
│   └── ...
│
├── GenConViT/                           # GenConViT model repository
│   ├── model/
│   │   ├── genconvit.py
│   │   ├── genconvit_ed.py
│   │   ├── genconvit_vae.py
│   │   └── config.yaml
│   ├── weights/
│   ├── train/
│   ├── dataset/
│   └── ...
│
├── requirements.txt
├── README.md
└── RESEARCH_PAPER.md
```

---

# ⚙️ Backend Setup (FastAPI)

## 1. Create Python Environment

Recommended Python version:

```text
Python 3.11
```

Create venv:

```bash
py -3.11 -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

## 2. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- FastAPI & Uvicorn
- PyTorch + TorchVision
- Transformers & TIMM
- OpenCV & Pillow
- FaceNet-PyTorch (MTCNN)
- MoviePy (video processing)
- And other utilities

---

## 3. Clone Model Repositories

### Clone Effort Repository

```bash
git clone https://github.com/YZY-stack/Effort-AIGI-Detection.git
```

This should create `Effort-AIGI-Detection-main/` directory in the project root.

### Clone GenConViT Repository

```bash
git clone https://github.com/erprogs/GenConViT.git
```

This should create `GenConViT/` directory in the project root.

Verify the cloned structure:
```text
Beyond_the_pixel/
├── Effort-AIGI-Detection-main/
├── GenConViT/
├── backend/
├── frontend/
└── ...
```

---

## 4. Download Model Weights

### Effort Model

Download the FaceForensics checkpoint:

[Download from Google Drive](https://drive.google.com/file/d/1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI/view)

Place inside:
```text
Effort-AIGI-Detection-main/DeepfakeBench/training/weights/
```

Example path:
```text
Effort-AIGI-Detection-main/DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth
```

### GenConViT Models

Download both network weights:

**Network A (Encoder-Decoder):**
```bash
cd GenConViT/weights
wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth
```

**Network B (VAE):**
```bash
wget https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth
```

Verify structure:
```text
GenConViT/weights/
├── genconvit_ed_inference.pth
└── genconvit_vae_inference.pth
```

---

## 5. Run backend server

From project root:

```bash
PYTHONPATH=. uvicorn backend.app:app --reload
```

Or from backend directory:

```bash
PYTHONPATH=.. uvicorn app:app --reload
```

Server runs at:

```text
http://127.0.0.1:8000
```

API documentation available at:
```text
http://127.0.0.1:8000/docs
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

```text
http://localhost:5173
```

---

# 📷 Image Detection Endpoint

```text
POST /detect-image
```

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

# 🎥 Video Detection Endpoint

```text
POST /detect-video
```

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

# 🧠 Model Details

### Effort Model
* Architecture: CLIP ViT-L/14 + Effort
* Method: Orthogonal Subspace Decomposition
* Dataset: FaceForensics++
* Input size: 224×224

### GenConViT Model
* Architecture: Vision Transformer with Generalized Convolution
* Two implementations:
  - **Network A (ED)**: Encoder-Decoder architecture
  - **Network B (VAE)**: Variational Autoencoder
* Training dataset: Multiple deepfake datasets
* Input size: Configurable (typically 224×224)

### Common Components
* Face detection: MTCNN
* Preprocessing: Normalization + Augmentation
* Output: Confidence score (0-1) + Classification

---

# 🔄 Pipeline Flow

## Image Detection Pipeline

```text
Image Upload
  ↓
Face Detection (MTCNN)
  ↓
Face Crop & Preprocessing
  ↓
Parallel Model Inference:
  - Effort Model
  - GenConViT Model
  ↓
Score Aggregation
  ↓
Final Prediction + Confidence Score
```

## Video Detection Pipeline

```text
Video Upload
  ↓
Frame Extraction (OpenCV)
  ↓
For each frame:
  → Face Detection (MTCNN)
  → Face Crop & Preprocessing
  → Parallel Model Inference (Effort + GenConViT)
  ↓
Score Averaging across frames
  ↓
Final Prediction + Confidence Score + Frame Count
```

# ⚠️ Known Limitations

* Requires visible/clear face in image
* Performance varies on low-quality or compressed media
* CPU inference is slow (recommend GPU for real-time use)
* May produce false positives on heavily stylized/artistic content
* Video processing is time-intensive (depends on resolution and frame count)
* Models have been primarily trained on frontal faces

---

# 🚀 Future Improvements

* Real-time webcam detection
* GPU acceleration & optimization
* Heatmap/attention visualization
* Confidence threshold adjustment UI
* Batch processing support
* Docker containerization
* Cloud deployment (AWS/Google Cloud)
* Model comparison interface
* Performance benchmarking dashboard
* Support for additional face detection backends

---

# 🧰 Tech Stack

**Backend:**
* FastAPI (REST API framework)
* PyTorch (Deep learning)
* Transformers & TIMM (Vision models)
* OpenCV (Video/image processing)
* facenet-pytorch (MTCNN face detection)
* MoviePy (Video handling)
* scikit-learn (ML utilities)

**Frontend:**
* React 18 (UI framework)
* TypeScript (Type safety)
* Vite (Build tool)
* CSS (Styling)

**Models:**
* Effort (ICML 2025)
* GenConViT (Vision Transformer variant)
* DFDC Challenge Models (Alternative)

**Additional Tools:**
* Uvicorn (ASGI server)
* CORS middleware
* HuggingFace Hub (model hosting)

---

# � API Usage Examples

## Image Detection

**Request:**
```bash
curl -X POST "http://localhost:8000/detect-image" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "result": {
    "success": true,
    "prediction": "LIKELY REAL",
    "deepfake_score": 0.12,
    "model": "effort"
  }
}
```

## Video Detection

**Request:**
```bash
curl -X POST "http://localhost:8000/detect-video" \
  -F "file=@/path/to/video.mp4"
```

**Response:**
```json
{
  "result": {
    "success": true,
    "prediction": "LIKELY FAKE",
    "deepfake_score": 0.78,
    "frames_analyzed": 32,
    "model": "effort"
  }
}
```

---

# 🔧 Troubleshooting

### Backend Issues
- **ModuleNotFoundError**: Ensure `PYTHONPATH` is set correctly
- **CUDA errors**: Install CPU version or NVIDIA CUDA/cuDNN
- **Weight loading fails**: Verify weight files are in correct directories

### Frontend Issues
- **CORS errors**: Check backend CORS middleware (allow_origins)
- **API unreachable**: Ensure backend is running on port 8000

### GPU Setup
For GPU acceleration on Windows:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

# 👨‍💻 Authors

Full-stack deepfake detection system integrating Effort (ICML 2025), GenConViT, and DFDC models with FastAPI backend and React frontend.

---

# 📄 License

For educational and research purposes only. The project integrates several open-source models with their respective licenses:
- **Effort**: As per original repository
- **GenConViT**: As per original repository
- **DFDC Challenge**: As per original repository
