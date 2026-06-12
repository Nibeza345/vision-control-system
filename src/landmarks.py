import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        indices = [33, 263, 1, 61, 291]  # left eye, right eye, nose, left mouth, right mouth
        for i in indices:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    cv2.imshow('5-Point Landmarks', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()