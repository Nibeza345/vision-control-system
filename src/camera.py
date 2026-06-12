import cv2
import numpy as np
from camera_utils import open_camera, camera_help

cap = open_camera(0)
if cap is None:
    camera_help()
    raise SystemExit(1)

# Create resizable window
cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Test', 1280, 720)  # Start with 720p

print("Camera window is resizable!")
print("Use mouse to resize or maximize the window")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Get current window size
    window_width = cv2.getWindowImageRect('Camera Test')[2]
    window_height = cv2.getWindowImageRect('Camera Test')[3]

    # Resize frame to match window size (maintain aspect ratio)
    h, w = frame.shape[:2]
    aspect_ratio = w / h

    # Calculate new dimensions maintaining aspect ratio
    new_width = window_width
    new_height = int(window_width / aspect_ratio)

    if new_height > window_height:
        new_height = window_height
        new_width = int(window_height * aspect_ratio)

    frame_resized = cv2.resize(frame, (new_width, new_height))

    cv2.imshow('Camera Test', frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.getWindowProperty('Camera Test', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
