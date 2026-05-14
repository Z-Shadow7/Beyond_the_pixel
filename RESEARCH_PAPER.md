# Beyond the Pixel: A Full-Stack Deepfake Detection System Using Multi-Model Ensemble and Face-Centered Analysis

## Abstract

Deepfake detection remains a practical challenge because manipulated media often preserves realistic global appearance while introducing subtle local artifacts around facial regions. This repository implements a full-stack deepfake detection system that combines multiple state-of-the-art detection models (Effort, GenConViT, and DFDC models), face detection with MTCNN, frame-based video analysis, and a FastAPI + React interface for interactive inference. The backend supports multiple detection architectures: a CLIP ViT-L/14 based Effort detector, a Vision Transformer-based GenConViT detector (with Encoder-Decoder and VAE variants), and EfficientNet-based DFDC detectors. Each model processes inputs by detecting and cropping a face before classification. For videos, the system samples frames, scores the detected faces independently using all available models, and aggregates the results into a final prediction. The frontend provides a simple operator-facing workflow for uploading images and videos and viewing confidence scores across multiple detection models. This paper describes the system design, multi-model pipeline, deployment architecture, and current limitations based on the code in the repository.

## 1. Introduction

Synthetic media generation tools have lowered the cost of creating convincing forged images and videos. As a result, there is increasing demand for lightweight forensic systems that can flag suspicious content without requiring specialized workflows. The project in this repository, Beyond the Pixel, addresses that need by integrating multiple modern deepfake detectors into a usable web application.

The design follows a practical assumption used by many face-forensics systems: a forged image or video is most likely to expose its artifacts in the face region. Instead of classifying the entire frame directly, the system first extracts a face crop and then applies one or more detection models to that crop. The multi-model approach improves robustness by leveraging different architectural strengths—Effort's orthogonal subspace decomposition, GenConViT's Vision Transformer design, and DFDC's EfficientNet backbone. This approach is used for both still images and sampled video frames.

## 2. System Overview

The application is organized into two layers:

1. A FastAPI backend that handles file uploads and parallel model inference across multiple detection models.
2. A React + TypeScript frontend that provides the upload and visualization interface with results from all available models.

The repository integrates three model repositories:

1. **Effort-AIGI-Detection**: Includes the DeepfakeBench framework, the Effort detector implementation (based on CLIP ViT-L/14), configuration files, preprocessing logic, and pretrained weights.
2. **GenConViT**: A Vision Transformer-based detector with two network variants—Encoder-Decoder (ED) and Variational Autoencoder (VAE)—for detecting generative model artifacts.
3. **DFDC Challenge**: EfficientNet-based detectors trained on the DFDC dataset with additional preprocessing utilities.

Each model is independently loaded and can be applied to images and video frames.

### 2.1 End-to-End Pipeline

The runtime inference flow is:

1. User uploads an image or video.
2. Backend stores the file locally in backend/uploads/.
3. The detector extracts a face with MTCNN.
4. The face is resized to 224 × 224.
5. All available detection models (Effort, GenConViT, DFDC) generate prediction scores in parallel.
6. Each score is mapped to a human-readable label and confidence indicator.
7. The frontend displays all model results for comparison.

For videos, the system reads frames with OpenCV, evaluates sampled frames independently using all models, averages the valid frame scores per model, and obtains a clip-level estimate for each detector.

## 3. Methodology

### 3.1 Face-Centered Classification

All detectors are intentionally face-centric. In backend/models/image_detector.py, the image is opened with PIL, passed through MTCNN, and converted into a face tensor before classification. If no face is detected, the system returns a failure response rather than making a weak prediction from the background.

This design choice reduces the search space for classifiers and aligns model inputs with the training regimes used by face deepfake datasets.

### 3.2 Image Detector

The image detector in backend/models/image_detector.py performs the following steps:

1. Load RGB image.
2. Detect a single face using MTCNN.
3. Convert the face tensor to NumPy format.
4. Resize to 224 × 224 with Albumentations.
5. Normalize using ImageNet statistics.
6. Run all available models (Effort, GenConViT, DFDC) in parallel.
7. Extract deepfake probability scores from each model.

The code labels scores as:

* LIKELY REAL for scores below 0.40
* SUSPICIOUS for scores in [0.40, 0.60)
* LIKELY FAKE for scores ≥ 0.60

### 3.3 GenConViT Detector

The GenConViT detector in backend/models/genconvit_detector.py leverages a Vision Transformer architecture with generalized convolutions:

1. Accept preprocessed face crops (224 × 224).
2. Use either the Encoder-Decoder (ED) or VAE variant.
3. Generate a detection score based on transformer attention patterns.
4. Return a continuous deepfake probability.

This model is particularly effective at detecting artifacts from vision-based generative models.

### 3.4 Video Detector

The video detector in backend/models/video_detector.py uses a frame-sampling strategy:

1. Open the video with OpenCV.
2. Iterate through frames.
3. Sample frames periodically (e.g., every N frames).
4. Save sampled frames temporarily.
5. Reuse the image detector on each frame, running all models.
6. Average all valid frame scores per model.
7. Assign the final clip-level label using the same thresholds as the image path.

This approach is computationally lighter than dense frame evaluation and keeps inference practical for real-time web usage while leveraging multiple detection models.

## 4. Model Architectures

### 4.1 Effort Detector

The Effort detector is built on top of CLIP ViT-L/14. Effort introduces orthogonal subspace decomposition into transformer weights to improve generalization across unseen deepfake generation methods.

The model uses:

* CLIP Vision Transformer Large Patch-14 backbone
* Orthogonal residual decomposition
* Face-centered inference
* DeepfakeBench integration
* Trained on FaceForensics++ dataset

The detector checkpoint is loaded at runtime in backend/models/image_detector.py.

### 4.2 GenConViT Detector

GenConViT is a Vision Transformer-based architecture that leverages generalized convolutions for improved robustness:

* Vision Transformer backbone with generalized convolution layers
* Two implementations: Encoder-Decoder (ED) and Variational Autoencoder (VAE)
* Trained on multiple deepfake generation methods
* Network A (ED): Focused on encoder-decoder-based detection
* Network B (VAE): Focused on generative model artifact detection

### 4.3 DFDC Challenge Models

EfficientNet-based detectors trained on the DFDC dataset:

* Multiple efficientnet variants (b5, b7)
* Trained on diverse compression and generation methods
* Configuration-based architecture selection

### 4.4 Preprocessing and Normalization

The runtime preprocessing pipeline includes:

* Face detection with MTCNN
* Resize to 224 × 224
* Normalization using ImageNet mean/std (consistent across all models)

All models receive a normalized face crop aligned with their respective training setups.

## 5. Backend Implementation

The backend is a FastAPI application defined in backend/app.py. It exposes two primary endpoints:

* POST /detect-image: Accepts image uploads and runs all available detectors
* POST /detect-video: Accepts video uploads, samples frames, and runs all detectors per frame

Both endpoints:

1. Accept multipart file uploads
2. Write the file to backend/uploads/
3. Call all available detection models in parallel where possible
4. Aggregate results from multiple detectors
5. Return a JSON payload containing all prediction results and confidence scores

The backend architecture includes:

* Modular detector loading (image_detector.py, video_detector.py, genconvit_detector.py)
* Lazy model initialization to reduce startup time
* Temporary frame storage in backend/outputs/
* CORS middleware configured for [http://localhost:5173](http://localhost:5173), which matches the Vite-based development server used by the frontend
* Error handling for missing faces, corrupted files, and model inference failures

## 6. Frontend Implementation

The frontend is implemented in React with TypeScript (Vite) and provides two independent detection panels:

* One for image uploads
* One for video uploads

The UI is intentionally minimal:

1. Upload a file (image or video)
2. Run analysis across all available models
3. Inspect confidence scores and prediction labels from each model
4. Compare results to identify consistent detections vs. model-specific signals

The interface displays:

* Loading states during inference
* Error messages for missing faces or processing failures
* Individual model scores from Effort, GenConViT, and DFDC detectors
* Aggregated confidence indicators
* Per-frame results for video analysis

Packaging multiple detectors into a simple web interface lowers the operational barrier for non-technical users and makes the forensic workflow more accessible. The multi-model display enables expert users to understand which detection approaches agree or diverge on a given input.

## 7. Inference Semantics

The system outputs continuous scores from multiple independent detectors rather than only a single binary prediction. This provides:

* Better operator interpretability through model comparison
* Clearer uncertainty handling—agreement across models signals confidence
* Improved review workflow integration—divergent scores flag ambiguous cases
* Robustness to model-specific failure modes

The video path uses score averaging across sampled frames per model, allowing the system to behave as a clip-level estimator. Per-frame scores are retained for detailed video-level analysis.

## 8. Limitations

The repository currently has several limitations:

* Requires a visible and clearly detectable face; returns error if MTCNN fails.
* Video inference is slower on CPU; GPU acceleration is strongly recommended for production use.
* Frame averaging may miss localized or transient manipulations present in only a few frames.
* False positives may occur on heavily stylized, heavily compressed, or AI-generated imagery.
* Model disagreement can occur across detectors, requiring user judgment to resolve.
* Uploaded files are stored locally in backend/uploads/ without automatic cleanup.
* Models have been primarily trained on frontal or near-frontal faces; profile or occluded faces may underperform.

## 9. Future Work

Several improvements would strengthen the system:

1. Add temporal aggregation strategies beyond simple averaging (e.g., weighted averaging based on confidence, temporal smoothing).
2. Add GPU-aware batching and model quantization for faster inference.
3. Implement ensemble voting and weighted aggregation across models.
4. Support webcam and stream-based real-time analysis.
5. Add explainability outputs such as attention maps, artifact localization heatmaps, or LIME-based attributions.
6. Add comprehensive benchmark logging and evaluation reporting across multiple datasets.
7. Containerize the application (Docker) for reproducible deployment across environments.
8. Implement model confidence calibration and uncertainty quantification.
9. Add support for different face detection backends (RetinaFace, YOLOv5-Face).
10. Optimize video processing with parallel frame batching.

## 10. Conclusion

This repository presents a full-stack deepfake detection system that combines multiple state-of-the-art detectors (Effort, GenConViT, DFDC models), MTCNN face localization, frame-level video scoring, and a modern web interface. The multi-model approach improves robustness and interpretability by leveraging complementary architectural strengths and providing detailed per-model scores for human review. The design focuses on practical deployment and usability while maintaining research-grade detection quality. Although the current implementation remains lightweight and research-oriented, it provides a strong engineering foundation for future work in media forensics and generalized deepfake detection.

## References

1. Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection. (ICML 2025)
2. GenConViT: Generative Vision Transformer Approach for Deepfake Detection.
3. DeepfakeBench Framework: A Comprehensive Benchmark for AI-generated and Manipulated Face Detection.
4. CLIP: Learning Transferable Visual Models From Natural Language Supervision. (OpenAI)
5. facenet-pytorch: PyTorch implementation of face detection and recognition using MTCNN and InceptionResnetV1.
6. Albumentations: Fast and Flexible Image Augmentation Library.
7. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
8. DFDC (Deepfake Detection Challenge) Dataset and Evaluation.
9. FaceForensics++: Learning to Detect Manipulated Facial Images.
