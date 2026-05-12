import sys
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A

# =========================================================
# FIX PYTHON PATH (IMPORTANT)
# =========================================================
# Add workspace root to Python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Define DFDC_PATH
DFDC_PATH = os.path.join(WORKSPACE_ROOT, 'dfdc_deepfake_challenge')

from dfdc_deepfake_challenge.training.zoo.classifiers import DeepFakeClassifier


# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD MODEL (ONCE)
# =========================================================
model = DeepFakeClassifier(
    encoder="tf_efficientnet_b7_ns"
).to(device)

checkpoint_path = os.path.join(
    DFDC_PATH,
    "weights",
    "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

# extract state dict
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# remove DataParallel prefix if exists
new_state_dict = {}

for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

print("✅ DFDC Model Loaded Successfully")


# =========================================================
# FACE DETECTOR
# =========================================================
mtcnn = MTCNN(
    image_size=380,
    margin=20,
    keep_all=False,
    device=device
)


# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = A.Compose([
    A.Resize(380, 380)
])


# =========================================================
# MAIN FUNCTION
# =========================================================
def detect_fake_image(image_path):

    try:
        img = Image.open(image_path).convert("RGB")

        # detect face
        face = mtcnn(img)

        if face is None:
            return {
                "success": False,
                "error": "No face detected"
            }

        # tensor -> numpy
        face = face.permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)

        # resize
        face = transform(image=face)["image"]

        # normalize
        face = face.astype(np.float32) / 255.0

        # HWC -> CHW
        face = np.transpose(face, (2, 0, 1))

        # batch
        face = np.expand_dims(face, axis=0)

        tensor = torch.tensor(face).float().to(device)

        # inference
        with torch.no_grad():
            output = model(tensor)
            score = torch.sigmoid(output).cpu().numpy()[0][0]

        # interpretation
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