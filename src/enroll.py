import cv2
import numpy as np
from camera_utils import open_camera, camera_help
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import pickle
import os

# -----------------------------
# Models
# -----------------------------
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

base_options = python.BaseOptions(model_asset_path="../models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

session = ort.InferenceSession("../models/embedder_arcface.onnx")
input_name = session.get_inputs()[0].name

# -----------------------------
# Alignment reference points
# -----------------------------
REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

INDICES = [33, 263, 1, 61, 291]


def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Load DB
# -----------------------------
DB_PATH = "../data/db/face_db.pkl"

if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
else:
    db = {}

# -----------------------------
# Enrollment setup
# -----------------------------
name_input = input("Enter identity name [nicole]: ").strip()
name = (name_input or "nicole").lower()
os.makedirs(f"../data/enroll/{name}", exist_ok=True)

embeddings = []
count = 0

cv2.namedWindow("Enroll", cv2.WINDOW_NORMAL)
cv2.namedWindow("Saved Aligned", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Enroll", 1280, 720)
cv2.resizeWindow("Saved Aligned", 224, 224)

cap = open_camera(0)
if cap is None:
    camera_help()
    raise SystemExit(1)

print("Look at camera. Auto-capture on good face. Aim for 15+ samples. Press Q to finish.")

# -----------------------------
# Capture loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        x, y = max(0, x), max(0, y)
        crop = frame[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = face_mesh.detect(mp_image)

        if results.face_landmarks:
            lm = results.face_landmarks[0]
            ch, cw = crop.shape[:2]
            pts = np.array(
                [[lm[i].x * cw, lm[i].y * ch] for i in INDICES],
                dtype=np.float32,
            )

            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            if M is None:
                continue

            aligned = cv2.warpAffine(
                crop, M, (112, 112),
                flags=cv2.INTER_LINEAR,
                borderValue=0,
            )

            blob = preprocess(aligned)
            emb = session.run(None, {input_name: blob})[0][0]

            norm = np.linalg.norm(emb)
            if norm == 0:
                continue

            emb = emb / norm
            embeddings.append(emb)
            count += 1

            cv2.imwrite(f"../data/enroll/{name}/{count:04d}.jpg", aligned)
            cv2.imshow("Saved Aligned", aligned)

            cv2.putText(
                frame,
                f"Captured {count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    window_width = cv2.getWindowImageRect("Enroll")[2]
    window_height = cv2.getWindowImageRect("Enroll")[3]

    if window_width > 0 and window_height > 0:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        new_width = window_width
        new_height = int(window_width / aspect_ratio)
        if new_height > window_height:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("Enroll", frame_resized)
    else:
        cv2.imshow("Enroll", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        print("Exiting...")
        break
    if key == ord('s'):
        print("Saving...")
        break
    if cv2.getWindowProperty("Enroll", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed")
        break

# -----------------------------
# Save DB
# -----------------------------
if embeddings:
    db.setdefault(name, []).extend(embeddings)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)
    print(f"Enrolled {name} with {len(embeddings)} new samples (total: {len(db[name])})")
else:
    print("No samples captured.")

cap.release()
cv2.destroyAllWindows()
