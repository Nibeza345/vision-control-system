import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
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

cap = cv2.VideoCapture(0)
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
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in INDICES], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)
            blob = preprocess(aligned)
            emb = session.run(None, {'input.1': blob})[0][0]
            emb = emb / np.linalg.norm(emb)
            print(f"Embedding norm: {np.linalg.norm(emb):.4f} | Shape: {emb.shape}")
            cv2.imshow('Aligned', aligned)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()