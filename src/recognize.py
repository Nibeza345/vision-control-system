import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import pickle

THRESHOLD = 0.62

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
base_options = python.BaseOptions(model_asset_path="../models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
face_mesh = vision.FaceLandmarker.create_from_options(options)
session = ort.InferenceSession("../models/embedder_arcface.onnx")

REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

INDICES = [33, 263, 1, 61, 291]

def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Load database and precompute mean embeddings
with open('../data/db/face_db.pkl', 'rb') as f:
    db = pickle.load(f)

reference = {}
for name, embs in db.items():
    mean_emb = np.mean(np.array(embs), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    reference[name] = mean_emb

cap = cv2.VideoCapture(0)

# Create resizable window
cv2.namedWindow('Live Recognition', cv2.WINDOW_NORMAL)
cv2.namedWindow('Aligned', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Recognition', 1280, 720)
cv2.resizeWindow('Aligned', 224, 224)

print("Recognition window is resizable!")
print("Use mouse to resize or maximize the window")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = face_mesh.detect(mp_image)
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in INDICES], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)
            blob = preprocess(aligned)
            query_emb = session.run(None, {'input.1': blob})[0][0]
            query_emb = query_emb / np.linalg.norm(query_emb)

            max_sim = -1
            identity = "Unknown"
            for name, ref_emb in reference.items():
                sim = np.dot(query_emb, ref_emb)
                if sim > max_sim:
                    max_sim = sim
                    identity = name

            label = identity if max_sim >= THRESHOLD else "Unknown"
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"{label} ({max_sim:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow('Aligned', aligned)

    # Get current window size and resize frame to match (maintain aspect ratio)
    window_width = cv2.getWindowImageRect('Live Recognition')[2]
    window_height = cv2.getWindowImageRect('Live Recognition')[3]

    if window_width > 0 and window_height > 0:
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        new_width = window_width
        new_height = int(window_width / aspect_ratio)

        if new_height > window_height:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)

        frame_resized = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Live Recognition', frame_resized)
    else:
        cv2.imshow('Live Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
