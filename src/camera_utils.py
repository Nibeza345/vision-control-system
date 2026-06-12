import cv2
import sys


def open_camera(preferred_index=0):
    """Open the first working USB camera (Windows-friendly)."""
    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("DEFAULT", 0),
    ]

    for index in indices:
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Camera OK: index={index} backend={name}")
                    return cap
            cap.release()

    return None


def camera_help():
    print(
        "\nCamera could not be opened. Try:\n"
        "  1. Plug FalconEye USB directly into the PC (not through ESP8266)\n"
        "  2. Windows Settings → Privacy & security → Camera → ON for desktop apps\n"
        "  3. Close Zoom, Teams, browser tabs, or other apps using the camera\n"
        "  4. Unplug and replug the camera, then run: py camera.py\n"
    )
