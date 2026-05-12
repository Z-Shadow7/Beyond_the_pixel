from fastapi import FastAPI, UploadFile, File
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DFDC_PATH = os.path.join(BASE_DIR, "dfdc_deepfake_challenge")

sys.path.insert(0, DFDC_PATH)
import shutil


from models.image_detector import detect_fake_image
from models.video_detector import detect_fake_video

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detect_fake_image(path)

    return {"result": result}


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):

    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detect_fake_video(path)

    return {"result": result}