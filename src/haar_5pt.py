import cv2
import mediapipe as mp

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop = frame[y:y+h, x:x+w]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            indices = [33, 263, 1, 61, 291]
            for i in indices:
                px = x + int(lm.landmark[i].x * w)
                py = y + int(lm.landmark[i].y * h)
                cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)
    cv2.imshow('Haar + 5-Point Landmarks', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()