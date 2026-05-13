# Beyond the Pixel: A Full-Stack Deepfake Detection System Using DFDC-Trained EfficientNet and Face Cropping

## Abstract
Deepfake detection remains a practical challenge because manipulated media often preserves realistic global appearance while introducing subtle local artifacts around facial regions. This repository implements a full-stack deepfake detection system that combines a DFDC-trained image classifier, face detection with MTCNN, frame-based video analysis, and a FastAPI + React interface for interactive inference. The backend uses a PyTorch-based `DeepFakeClassifier` with an EfficientNet-B7 encoder and processes each input by detecting and cropping a face before classification. For videos, the system samples frames, scores the detected faces independently, and aggregates the results into a final prediction. The frontend provides a simple operator-facing workflow for uploading images and videos and viewing confidence scores. This paper describes the system design, model pipeline, training configuration, deployment architecture, and current limitations based on the code in the repository.

## 1. Introduction
Synthetic media generation tools have lowered the cost of creating convincing forged images and videos. As a result, there is increasing demand for lightweight forensic systems that can flag suspicious content without requiring specialized workflows. The project in this repository, *Beyond the Pixel*, addresses that need by integrating a deep learning detector into a usable web application.

The design follows a practical assumption used by many face-forensics systems: a forged image or video is most likely to expose its artifacts in the face region. Instead of classifying the entire frame directly, the system first extracts a face crop and then applies a deepfake classifier to that crop. This approach is used for both still images and sampled video frames.

## 2. System Overview
The application is organized into two layers:

1. A FastAPI backend that handles file uploads and model inference.
2. A React + TypeScript frontend that provides the upload and visualization interface.

The repository also includes a `dfdc_deepfake_challenge` subtree containing the original DFDC training and inference assets, including the classifier architecture, preprocessing utilities, configuration files, and model weights used by the runtime detector.

### 2.1 End-to-End Pipeline
The runtime inference flow is:

1. User uploads an image or video.
2. Backend stores the file locally in `backend/uploads/`.
3. The detector extracts a face with MTCNN.
4. The face is resized to `380 x 380`.
5. The `DeepFakeClassifier` generates a raw score.
6. The score is mapped to a human-readable label.
7. The frontend displays the result and confidence indicator.

For videos, the system reads frames with OpenCV, evaluates every fifth frame, and averages the valid frame scores to obtain a clip-level estimate.

## 3. Methodology

### 3.1 Face-Centered Classification
The detector is intentionally face-centric. In `backend/models/image_detector.py`, the image is opened with PIL, passed through MTCNN, and converted into a face tensor before classification. If no face is detected, the system returns a failure response rather than making a weak guess from the background.

This design choice reduces the search space for the classifier and aligns the model input with the DFDC training regime, which is centered on cropped face regions.

### 3.2 Image Detector
The image detector performs the following steps:

1. Load RGB image.
2. Detect a single face using MTCNN.
3. Convert the face tensor to NumPy format.
4. Resize to `380 x 380` with Albumentations.
5. Normalize pixel values to `[0, 1]`.
6. Run the PyTorch classifier.
7. Apply sigmoid to obtain a deepfake score in `[0, 1]`.

The code labels scores as:

- `LIKELY REAL` for scores below `0.40`
- `SUSPICIOUS` for scores in `[0.40, 0.60)`
- `LIKELY FAKE` for scores `>= 0.60`

### 3.3 Video Detector
The video detector in `backend/models/video_detector.py` uses a frame-sampling strategy:

1. Open the video with OpenCV.
2. Iterate through frames.
3. Keep every fifth frame.
4. Save the sampled frame to disk.
5. Reuse the image detector on the frame.
6. Average all valid frame scores.
7. Assign the final clip-level label using the same thresholds as the image path.

This approach is simple and computationally efficient compared with dense frame evaluation. It trades some temporal precision for lower latency and a narrower implementation surface.

## 4. Model Architecture
The repository uses a `DeepFakeClassifier` with an EfficientNet-B7 encoder (`tf_efficientnet_b7_ns`). The training configuration in `dfdc_deepfake_challenge/configs/b7.json` specifies:

- input size: `380`
- batch size: `12`
- optimizer: SGD
- momentum: `0.9`
- weight decay: `1e-4`
- learning rate: `0.01`
- schedule: polynomial, 40 epochs
- mixed precision: enabled
- loss: binary cross-entropy

The classifier checkpoint is loaded at runtime in `backend/models/image_detector.py`, and the code removes `module.` prefixes to support weights saved through `DataParallel`.

### 4.1 Preprocessing and Normalization
The DFDC training pipeline uses augmentation-heavy preprocessing in the original training scripts. For inference, the live detector keeps preprocessing narrower and deterministic:

- face detection with MTCNN
- resize to `380 x 380`
- normalize by dividing by `255.0`

The model is therefore exposed to the same semantic object class at inference time that it was trained on: a cropped face region.

## 5. Backend Implementation
The backend is a FastAPI application defined in `backend/app.py`. It exposes two endpoints:

- `POST /detect-image`
- `POST /detect-video`

Both endpoints accept multipart file uploads, write the file to disk, call the corresponding detector, and return a JSON payload containing the prediction result.

The backend also enables CORS for `http://localhost:5173`, which matches the Vite-based development server used by the frontend.

## 6. Frontend Implementation
The frontend is implemented in React with TypeScript and provides two independent detection panels:

- one for image uploads
- one for video uploads

The UI is intentionally minimal in terms of interaction logic: upload a file, run analysis, and inspect the score bar and label. The interface also displays a loading state and connection error state when the API is unavailable.

From a product perspective, the frontend is important because forensic models are often difficult to use directly from notebooks or scripts. Packaging the detector into a simple web interface lowers the operational barrier for non-technical users.

## 7. Inference Semantics
The current system does not output a binary only result. Instead, it exposes a continuous score and then maps that score to qualitative labels. This is useful because:

- it gives operators a sense of confidence
- it avoids overclaiming certainty on borderline inputs
- it makes the system easier to interpret in a practical review workflow

The video path uses score averaging across sampled frames. That means a short burst of suspicious frames can influence the final score, but the model still behaves as a clip-level estimator rather than a frame-by-frame alarm system.

## 8. Limitations
The repository already acknowledges several constraints, and the code confirms them:

- The detector is strongest when a clear face is visible.
- Inputs without detectable faces return an error instead of a prediction.
- Video inference is slower on CPU because the system processes frames serially.
- The video aggregator is a simple mean, which may miss localized manipulations or transient artifacts.
- The system has no built-in benchmark reporting in the current repository, so there are no reproducible accuracy numbers to cite from the code alone.

There is also a practical limitation in the backend implementation: uploaded files are stored locally without cleanup logic, so production deployment would need stronger file lifecycle management.

## 9. Future Work
Several improvements would strengthen the system:

1. Replace mean aggregation with a more robust temporal strategy, such as confidence-weighted aggregation or a top-k frame policy.
2. Add GPU-aware batching for faster video inference.
3. Support webcam or stream-based analysis.
4. Add explainability outputs such as heatmaps or facial artifact localization.
5. Introduce evaluation scripts and benchmark logging so that the system can report measurable performance on DFDC-style validation sets.
6. Containerize the full stack for easier deployment and reproducibility.

## 10. Conclusion
This repository presents a coherent deepfake detection system that combines a DFDC-trained EfficientNet-B7 classifier, MTCNN face localization, frame-level video scoring, and a modern web interface. The core strength of the design is its simplicity: it uses a face-centered pipeline that is easy to reason about, straightforward to deploy, and practical for interactive inspection. While the current implementation does not include reported experimental metrics, it provides a solid engineering foundation for a real-time media forensics tool and a clear base for future research on more robust aggregation and evaluation.

## References
1. The DeepFake Detection Challenge (DFDC) Preview Dataset.
2. Selim Seferbekov, DFDC solution repository and associated training pipeline.
3. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
4. facenet-pytorch / MTCNN face detection utilities.
5. Albumentations image augmentation library.

