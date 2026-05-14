import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
GENCONVIT_PATH = ROOT / "GenConViT"

sys.path.insert(0, str(GENCONVIT_PATH))

# -----------------------------
# IMPORTS
# -----------------------------
from model.config import load_config
from model.genconvit import GenConViT


# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# INIT MODEL COMPONENTS
# -----------------------------
config = load_config()

# Weight file names for GenConViT initialization
ed = "genconvit_ed_inference"
vae = "genconvit_vae_inference"
net = "both"  # Use both ED and VAE models


# -----------------------------
# BUILD MODEL
# -----------------------------
model = GenConViT(config, ed, vae, net, fp16=False).to(DEVICE)


# Weights are already loaded in GenConViT.__init__, no need to load again

# Model is already in eval mode from GenConViT.__init__


# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# INFERENCE
# -----------------------------
def predict_frame(image_path):

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Flatten and take mean if output has multiple elements
        with torch.no_grad():

            output = model(tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]

            probs = torch.softmax(output, dim=1)

            fake_score = probs[:, 0].mean().item()

            return fake_score