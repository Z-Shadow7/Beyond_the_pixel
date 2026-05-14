import cv2
from models.genconvit_detector import predict_frame


def detect_fake_video(video_path):

    cap = cv2.VideoCapture(video_path)

    scores = []
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Analyze every 5th frame
        if frame_count % 5 != 0:
            continue

        temp_path = "temp_frame.jpg"

        cv2.imwrite(temp_path, frame)

        score = predict_frame(temp_path)

        scores.append(score)

    cap.release()

    if not scores:
        return {
            "success": False,
            "error": "No valid frames processed"
        }

    avg_score = sum(scores) / len(scores)

    if avg_score < 0.4:
        label = "LIKELY REAL"

    elif avg_score < 0.6:
        label = "SUSPICIOUS"

    else:
        label = "LIKELY FAKE"

    return {
        "success": True,
        "prediction": label,
        "deepfake_score": round(avg_score, 4),
        "frames_analyzed": len(scores)
    }