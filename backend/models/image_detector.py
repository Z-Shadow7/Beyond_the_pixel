import sys
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A

# =========================================================
# PATH
# =========================================================

WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')
)

EFFORT_PATH = os.path.join(
    WORKSPACE_ROOT,
    'Effort-AIGI-Detection-main',
    'DeepfakeBench'
)

if EFFORT_PATH not in sys.path:
    sys.path.insert(0, EFFORT_PATH)

# =========================================================
# IMPORT DETECTOR
# =========================================================

from training.detectors import DETECTOR

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD MODEL
# =========================================================

cfg = {}

model = DETECTOR["effort"](cfg).to(device)

checkpoint_path = os.path.join(
    EFFORT_PATH,
    "training",
    "weights",
    "effort_clip_L14_trainOn_FaceForensic.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()

print("✅ Effort Model Loaded Successfully")

# =========================================================
# FACE DETECTOR
# =========================================================

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=device
)

# =========================================================
# TRANSFORM
# =========================================================

transform = A.Compose([
    A.Resize(224, 224)
])

# =========================================================
# MAIN FUNCTION
# =========================================================

def detect_fake_image(image_path):

    try:

        # =================================================
        # LOAD IMAGE
        # =================================================

        img = Image.open(image_path).convert("RGB")

        # =================================================
        # DETECT FACE
        # =================================================

        face = mtcnn(img)

        if face is None:
            return {
                "success": False,
                "error": "No face detected"
            }

        # =================================================
        # PREPROCESS
        # =================================================

        face = face.permute(1, 2, 0).cpu().numpy()

        face = (face * 255).astype(np.uint8)

        face = transform(image=face)["image"]

        face = face.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        face = (face - mean) / std

        # HWC -> CHW
        face = np.transpose(face, (2, 0, 1))

        # batch dimension
        face = np.expand_dims(face, axis=0)

        tensor = torch.tensor(face).float().to(device)

        # =================================================
        # INFERENCE
        # =================================================

        with torch.no_grad():

            data_dict = {
                "image": tensor
            }

            output = model(data_dict)

            print("OUTPUT TYPE:", type(output))
            print("OUTPUT:", output)

            # handle dict outputs
            if isinstance(output, dict):

                print("DICT KEYS:", output.keys())

                if "prob" in output:
                    output = output["prob"]

                elif "pred" in output:
                    output = output["pred"]

                elif "logits" in output:
                    output = output["logits"]

            print("FINAL OUTPUT SHAPE:", output.shape)

            output = output.flatten()
            print(output)

            score = output[0].item()

        # =================================================
        # LABEL
        # =================================================

        if score < 0.40:
            label = "LIKELY REAL"

        elif score < 0.60:
            label = "SUSPICIOUS"

        else:
            label = "LIKELY FAKE"

        return {
            "success": True,
            "prediction": label,
            "deepfake_score": round(float(score), 4)
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }