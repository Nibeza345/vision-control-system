import csv
import cv2
import numpy as np
from camera_utils import open_camera, camera_help
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import pickle
import os
from datetime import datetime
import traceback
import paho.mqtt.client as mqtt
import json
import time

# ===================== CONFIGURATION =====================
LOCK_THRESHOLD = 0.62  # Similarity required to start a new lock
HOLD_THRESHOLD = 0.52  # Keep lock through brief score dips
LOCK_CONFIRM_FRAMES = 10  # Must see target N frames in a row before locking
LOCK_SETTLE_FRAMES = 45  # ~1.5s after lock: servo holds still so lock stabilizes
_target_input = input("Enter the identity to lock onto [nicole]: ").strip().lower()
TARGET_NAME = _target_input or "nicole"
EMPTY_SEARCH_FRAMES = 12  # Frames with no face before search (after lock)
MISS_SEARCH_FRAMES = 18   # Frames target missing before search (after lock)
SEARCH_PAN_MIN = 30
SEARCH_PAN_MAX = 150
REACQUIRE_SCAN_STEP = 8.0
SEARCH_PUBLISH_INTERVAL = 0.4   # Slow search sweep
TRACK_PUBLISH_INTERVAL = 0.35   # Slow follow — stops dancing
MIN_PAN_PUBLISH_DELTA = 6       # Ignore small angle changes
PAN_STEP_MAX = 5.0              # Small steps per correction
PAN_INVERT = False              # Set True if servo turns the wrong way
MAX_TRACK_JUMP_RATIO = 0.35     # Reject jumps when TRACKING
MOVEMENT_THRESHOLD = 40
DEADZONE_RATIO = 0.20           # Wide center — no move while face is here
FACE_CENTER_SMOOTHING = 0.35    # Smooth face position (reduces bbox jitter)
BLINK_EAR_THRESHOLD = 0.21
SMILE_CONFIDENCE_THRESHOLD = 0.65
CONSECUTIVE_SMILE_FRAMES = 3
MAX_FACES = 10  # Maximum faces to detect/process per frame

# ===================== MQTT CONFIGURATION =====================
TEAM_ID = "nicole"  # Must match esp8266_servo_controller.ino (your project name on MQTT)
MQTT_BROKER = "10.12.72.188"  # Your PC IP where Mosquitto runs (run ipconfig if Wi-Fi changes)
MQTT_PORT = 1883
MQTT_TOPIC = f"vision/{TEAM_ID}/movement"
MQTT_HEARTBEAT_TOPIC = f"vision/{TEAM_ID}/heartbeat"

# ===================== 2DOF SERVO CONFIGURATION =====================
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER = 90
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SERVO_SMOOTHING = 0.25
LOG_DIR = "../data/logs"

# ===================== DETECTION TUNING =====================
MIN_FACE_SIZE = 40  # Minimum face size to process
MAX_FACE_SIZE = 600  # Maximum face size to process
BBOX_PADDING = 0.25  # Padding around landmarks for bounding box (extra head room)

# ===================== INITIALIZATION =====================
print("Initializing multi-face detection system...")

# Initialize FaceLandmarker detector using Tasks API
base_options = python.BaseOptions(model_asset_path="../models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=MAX_FACES
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# Initialize ONNX Runtime session for ArcFace
try:
    model_path = "../models/embedder_arcface.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    session = ort.InferenceSession(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Alignment references
REF_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)
INDICES_5PT = [33, 263, 1, 61, 291]

# Landmark groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# ===================== UTILITY FUNCTIONS =====================
def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(aligned):
    blob = preprocess(aligned)
    emb = session.run(None, {'input.1': blob})[0][0]
    return emb / np.linalg.norm(emb)

def compute_ear(landmarks, eye_indices, h, w):
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def detect_smile(landmarks, h, w, baseline_mouth_width=None, baseline_lip_sep=None):
    left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
    upper_lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower_lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])

    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    lip_separation = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    face_width = np.linalg.norm(left_eye - right_eye)

    normalized_width = mouth_width / face_width if face_width > 0 else 0
    normalized_sep = lip_separation / face_width if face_width > 0 else 0

    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    left_corner_height = left_mouth[1] - nose_tip[1]
    right_corner_height = right_mouth[1] - nose_tip[1]

    smile_score = 0
    if normalized_width > 0.35:
        width_score = min((normalized_width - 0.35) * 10, 1.0)
        smile_score += width_score * 0.4
    if normalized_sep > 0.08:
        sep_score = min((normalized_sep - 0.08) * 20, 1.0)
        smile_score += sep_score * 0.3
    corner_up_score = 0
    if left_corner_height < -5 and right_corner_height < -5:
        corner_up_score = 1.0
    elif left_corner_height < 0 or right_corner_height < 0:
        corner_up_score = 0.6
    smile_score += corner_up_score * 0.3

    if baseline_mouth_width and baseline_lip_sep:
        width_increase = mouth_width / baseline_mouth_width if baseline_mouth_width > 0 else 1
        sep_increase = lip_separation / baseline_lip_sep if baseline_lip_sep > 0 else 1
        if width_increase > 1.15:
            smile_score += min(width_increase - 1, 0.2)
        if sep_increase > 1.3:
            smile_score += min(sep_increase - 1, 0.2)

    smile_score = min(max(smile_score, 0), 1)
    return smile_score > SMILE_CONFIDENCE_THRESHOLD, smile_score, normalized_width, normalized_sep

class ActionDetector:
    def __init__(self):
        self.prev_nose_x = None
        self.baseline_mouth_width = None
        self.baseline_lip_sep = None
        self.smile_frames = 0
        self.blink_frames = 0

    def update_baseline(self, landmarks, h, w):
        left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
        upper_lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
        lower_lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
        self.baseline_mouth_width = np.linalg.norm(left_mouth - right_mouth)
        self.baseline_lip_sep = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    def detect_actions(self, landmarks, h, w, locked):
        actions = []
        if not locked:
            return actions

        nose_x = int(landmarks[1].x * w)
        if self.prev_nose_x is not None:
            delta_x = nose_x - self.prev_nose_x
            if abs(delta_x) > MOVEMENT_THRESHOLD:
                direction = "right" if delta_x > 0 else "left"
                actions.append(f"moved {direction} ({abs(delta_x):.0f}px)")
        self.prev_nose_x = nose_x

        ear_left = compute_ear(landmarks, LEFT_EYE, h, w)
        ear_right = compute_ear(landmarks, RIGHT_EYE, h, w)
        ear = (ear_left + ear_right) / 2
        if ear < BLINK_EAR_THRESHOLD:
            self.blink_frames += 1
            if self.blink_frames == 2:
                actions.append(f"blink (EAR: {ear:.2f})")
        else:
            self.blink_frames = 0

        is_smiling, smile_score, mouth_width_norm, lip_sep_norm = detect_smile(
            landmarks, h, w, self.baseline_mouth_width, self.baseline_lip_sep
        )
        if is_smiling:
            self.smile_frames += 1
            if self.smile_frames >= CONSECUTIVE_SMILE_FRAMES:
                actions.append(f"smile (score: {smile_score:.2f})")
        else:
            self.smile_frames = 0

        if 0.25 < mouth_width_norm < 0.33 and 0.05 < lip_sep_norm < 0.08:
            self.update_baseline(landmarks, h, w)

        return actions

# ===================== TRACKING LOG =====================
class TrackingLogger:
    def __init__(self, speaker_id):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(LOG_DIR, f"tracking_{speaker_id}_{stamp}.csv")
        self.jsonl_path = os.path.join(LOG_DIR, f"tracking_{speaker_id}_{stamp}.jsonl")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp_iso", "unix_ts", "speaker_id", "lock_state", "confidence",
                "status", "pan_command", "tilt_command", "pan_angle", "tilt_angle",
                "face_x", "face_y",
            ])
        print(f"Evidence log: {self.csv_path}")

    def log(self, record):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                record["timestamp_iso"], record["unix_ts"], record["speaker_id"],
                record["lock_state"], record["confidence"], record["status"],
                record["pan_command"], record["tilt_command"],
                record["pan_angle"], record["tilt_angle"],
                record["face_x"], record["face_y"],
            ])
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ===================== MQTT FUNCTIONS =====================
mqtt_client = None
tracking_logger = None
current_pan_angle = SERVO_CENTER
current_tilt_angle = SERVO_CENTER
target_pan_angle = SERVO_CENTER
target_tilt_angle = SERVO_CENTER
last_known_face_x = None
last_known_face_y = None
search_sweep_pan = float(SERVO_CENTER)
search_sweep_dir = 1
smoothed_face_x = None
smoothed_face_y = None

def clamp_angle(angle):
    return max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        # Send initial heartbeat
        send_heartbeat()
    else:
        print(f"✗ Failed to connect to MQTT broker: {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    print(f"✗ Disconnected from MQTT broker: {rc}")

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_disconnect = on_mqtt_disconnect

        print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        return True
    except Exception as e:
        print(f"✗ Failed to initialize MQTT: {e}")
        return False

def calculate_pan_angle(face_center_x, frame_width):
    normalized_x = face_center_x / frame_width
    angle = SERVO_MAX_ANGLE - (normalized_x * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE))
    if PAN_INVERT:
        angle = SERVO_MAX_ANGLE - (angle - SERVO_CENTER) + SERVO_CENTER
    return clamp_angle(angle)


def smooth_face_center(face_center_x, face_center_y):
    """EMA smooth face center so bbox jitter does not shake the servo."""
    global smoothed_face_x, smoothed_face_y
    if smoothed_face_x is None:
        smoothed_face_x = float(face_center_x)
        smoothed_face_y = float(face_center_y)
    else:
        a = FACE_CENTER_SMOOTHING
        smoothed_face_x = a * face_center_x + (1 - a) * smoothed_face_x
        smoothed_face_y = a * face_center_y + (1 - a) * smoothed_face_y
    return smoothed_face_x, smoothed_face_y


def step_pan_toward_face(face_center_x, frame_width):
    """One small pan step toward centering the face (only call when publishing)."""
    global current_pan_angle
    cx = frame_width / 2.0
    error = face_center_x - cx
    dead = frame_width * DEADZONE_RATIO
    if abs(error) < dead:
        return current_pan_angle, False

    error_norm = max(-1.0, min(1.0, error / cx))
    delta = -error_norm * PAN_STEP_MAX
    if PAN_INVERT:
        delta = -delta
    current_pan_angle = clamp_angle(current_pan_angle + delta)
    return current_pan_angle, True

def calculate_tilt_angle(face_center_y, frame_height):
    normalized_y = face_center_y / frame_height
    return clamp_angle(SERVO_MIN_ANGLE + (normalized_y * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)))

def resolve_movement(face_center_x, face_center_y, frame_width, frame_height):
    cx, cy = frame_width // 2, frame_height // 2
    dx = face_center_x - cx
    dy = face_center_y - cy
    x_dead = int(frame_width * DEADZONE_RATIO)
    y_dead = int(frame_height * DEADZONE_RATIO)

    pan_cmd = "CENTERED"
    tilt_cmd = "CENTERED"
    if dx < -x_dead:
        pan_cmd = "LEFT"
    elif dx > x_dead:
        pan_cmd = "RIGHT"
    if dy < -y_dead:
        tilt_cmd = "UP"
    elif dy > y_dead:
        tilt_cmd = "DOWN"

    if pan_cmd == "CENTERED" and tilt_cmd == "CENTERED":
        status = "CENTERED"
    elif pan_cmd != "CENTERED" and tilt_cmd != "CENTERED":
        status = f"MOVE_{pan_cmd}_{tilt_cmd}"
    elif pan_cmd != "CENTERED":
        status = f"MOVE_{pan_cmd}"
    else:
        status = f"MOVE_{tilt_cmd}"
    return status, pan_cmd, tilt_cmd

def reset_search_sweep():
    global search_sweep_pan, search_sweep_dir
    search_sweep_pan = float(SERVO_CENTER)
    search_sweep_dir = 1


def advance_search_angles():
    """Sweep pan servo across full range until target face is seen again."""
    global search_sweep_pan, search_sweep_dir
    search_sweep_pan += REACQUIRE_SCAN_STEP * search_sweep_dir
    if search_sweep_pan >= SEARCH_PAN_MAX:
        search_sweep_pan = float(SEARCH_PAN_MAX)
        search_sweep_dir = -1
    elif search_sweep_pan <= SEARCH_PAN_MIN:
        search_sweep_pan = float(SEARCH_PAN_MIN)
        search_sweep_dir = 1
    return clamp_angle(search_sweep_pan), float(SERVO_CENTER)

def last_known_pan_angle():
    if last_known_face_x is None:
        return SERVO_CENTER
    return calculate_pan_angle(last_known_face_x, FRAME_WIDTH)

def last_known_tilt_angle():
    if last_known_face_y is None:
        return SERVO_CENTER
    return calculate_tilt_angle(last_known_face_y, FRAME_HEIGHT)

def send_heartbeat():
    """Send heartbeat message to MQTT broker"""
    if mqtt_client:
        heartbeat_msg = {
            "node": "pc",
            "status": "ONLINE",
            "timestamp": int(time.time())
        }
        try:
            mqtt_client.publish(MQTT_HEARTBEAT_TOPIC, json.dumps(heartbeat_msg))
        except Exception as e:
            print(f"Error sending heartbeat: {e}")

def publish_movement(
    status,
    confidence=0.0,
    face_center_x=None,
    face_center_y=None,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    lock_state="SEARCHING",
    pan_cmd="STOP",
    tilt_cmd="STOP",
    pan_angle=None,
    tilt_angle=None,
):
    """Publish 2DOF movement status and servo angles to MQTT."""
    global current_pan_angle, current_tilt_angle, target_pan_angle, target_tilt_angle
    global last_known_face_x, last_known_face_y

    if pan_angle is None or tilt_angle is None:
        if face_center_x is not None and face_center_y is not None:
            last_known_face_x = face_center_x
            last_known_face_y = face_center_y
            sx, sy = smooth_face_center(face_center_x, face_center_y)
            pan_angle, _ = step_pan_toward_face(sx, frame_width)
            target_tilt_angle = calculate_tilt_angle(sy, frame_height)
            current_tilt_angle = (SERVO_SMOOTHING * target_tilt_angle +
                                  (1 - SERVO_SMOOTHING) * current_tilt_angle)
            tilt_angle = current_tilt_angle
        else:
            pan_angle = current_pan_angle
            tilt_angle = current_tilt_angle

    current_pan_angle = pan_angle
    current_tilt_angle = tilt_angle
    unix_ts = int(time.time())
    now_iso = datetime.now().isoformat(timespec="milliseconds")
    movement_msg = {
        "status": status,
        "confidence": float(confidence),
        "timestamp": unix_ts,
        "speaker_id": TARGET_NAME,
        "lock_state": lock_state,
        "pan_command": pan_cmd,
        "tilt_command": tilt_cmd,
        "pan_angle": round(float(pan_angle), 1),
        "tilt_angle": round(float(tilt_angle), 1),
        "servo_angle": round(float(pan_angle), 1),
        "tilt_servo_angle": round(float(tilt_angle), 1),
        "face_position": int(face_center_x) if face_center_x is not None else None,
        "face_position_y": int(face_center_y) if face_center_y is not None else None,
        "degrees_from_center": round(abs(float(pan_angle) - SERVO_CENTER), 1),
        "direction": pan_cmd if pan_cmd != "CENTERED" else tilt_cmd,
    }

    if tracking_logger:
        tracking_logger.log({
            "timestamp_iso": now_iso,
            "unix_ts": unix_ts,
            "speaker_id": TARGET_NAME,
            "lock_state": lock_state,
            "status": status,
            "pan_command": pan_cmd,
            "tilt_command": tilt_cmd,
            "confidence": round(float(confidence), 4),
            "pan_angle": movement_msg["pan_angle"],
            "tilt_angle": movement_msg["tilt_angle"],
            "face_x": face_center_x if face_center_x is not None else "",
            "face_y": face_center_y if face_center_y is not None else "",
        })

    if mqtt_client:
        try:
            pan_int = int(round(float(pan_angle)))
            global last_published_pan, last_track_publish_time, last_search_publish_time
            now = time.time()
            is_search = status == "SEARCH"
            if is_search:
                if now - last_search_publish_time < SEARCH_PUBLISH_INTERVAL:
                    return
            else:
                if now - last_track_publish_time < TRACK_PUBLISH_INTERVAL:
                    return
                if status == "CENTERED":
                    return
                if (last_published_pan is not None and
                        abs(pan_int - last_published_pan) < MIN_PAN_PUBLISH_DELTA):
                    return

            servo_payload = json.dumps({
                "status": "SEARCH" if is_search else "TRACK",
                "pan_angle": pan_int,
                "servo_angle": pan_int,
            })
            mqtt_client.publish(MQTT_TOPIC, servo_payload)
            last_published_pan = pan_int
            if is_search:
                last_search_publish_time = now
            else:
                last_track_publish_time = now
            print(
                f"📡 {status} | pan {pan_int}° | {pan_cmd}/{tilt_cmd} | lock={lock_state}"
            )
        except Exception as e:
            print(f"Error publishing movement: {e}")

# ===================== LOAD FACE DATABASE =====================
try:
    db_path = '../data/db/face_db.pkl'
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    reference = {}
    for name, embs in db.items():
        if len(embs) > 0:
            mean_emb = np.mean(np.array(embs), axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            reference[name.lower()] = mean_emb
    if TARGET_NAME not in reference:
        print(f"Error: {TARGET_NAME} not found in database!")
        print(f"Available identities: {list(reference.keys())}")
        exit(1)
    target_emb = reference[TARGET_NAME]
    print(f"Loaded database with {len(reference)} identities")
except Exception as e:
    print(f"Error loading database: {e}")
    traceback.print_exc()
    exit(1)

# ===================== MAIN LOOP =====================
action_detector = ActionDetector()
locked = False
tracking_state = "SEARCHING"  # SEARCHING | TRACKING | REACQUIRING
locked_start = None
miss_count = 0
prev_bbox = None
history_file = None
fps_counter = 0
start_time = datetime.now()
tracking_logger = TrackingLogger(TARGET_NAME)

frame_count = 0
prev_position = None
last_published_status = None
last_lock_display = None  # Cached bbox for stable overlay when detection hiccups
empty_frame_count = 0
last_search_publish_time = 0.0
last_track_publish_time = 0.0
last_published_pan = None
lock_confirm_count = 0
frames_since_lock = 9999

print("\n" + "=" * 50)
print(f"Target: {TARGET_NAME.capitalize()} | 2DOF pan/tilt tracking")
print("Controls:")
print(" - Press 'q' to quit")
print(" - Press 'r' to manually release lock")
print(" - Colors: Thick Green = locked target | Thin Green = other target instances")
print("           Yellow = other enrolled people | Red = unknown")
print(" - Servo stays still until lock is stable, then follows slowly")
print(" - Step away after lock: servo scans until you return")
print("=" * 50 + "\n")

cap = open_camera(0)
if cap is None:
    camera_help()
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create resizable window
cv2.namedWindow('Face Locking System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Locking System', 1280, 720)

print("Face locking window is resizable!")
print("Use mouse to resize or maximize the window")

# Initialize MQTT connection
if not init_mqtt():
    print("Warning: MQTT connection failed. Continuing without servo control.")

# Initialize heartbeat timer
last_heartbeat_time = time.time()
HEARTBEAT_INTERVAL = 30  # Send heartbeat every 30 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    fps_counter += 1
    frame_count += 1

    # Run face detection every frame (skipping frames caused false "face lost" flicker)
    recognized_faces = []
    if True:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect(mp_image)

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Size filtering - skip small/large detections
                x_coords = np.array([l.x for l in face_landmarks])
                y_coords = np.array([l.y for l in face_landmarks])
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                face_width = (x_max - x_min) * w_frame
                face_height = (y_max - y_min) * h_frame

                if MIN_FACE_SIZE < face_width < MAX_FACE_SIZE and MIN_FACE_SIZE < face_height < MAX_FACE_SIZE:
                    # 5-point alignment on full frame
                    pts = np.array([[face_landmarks[i].x * w_frame,
                                     face_landmarks[i].y * h_frame] for i in INDICES_5PT],
                                   dtype=np.float32)
                    try:
                        M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
                        aligned = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

                        query_emb = get_embedding(aligned)

                        sim_to_target = np.dot(query_emb, target_emb)
                        sims = {n: np.dot(query_emb, emb) for n, emb in reference.items()}
                        best_sim = max(sims.values()) if sims else -1
                        best_name = max(sims, key=sims.get) if best_sim >= LOCK_THRESHOLD else "Unknown"

                        # Compute bounding box with padding
                        x_coords = np.array([l.x for l in face_landmarks])
                        y_coords = np.array([l.y for l in face_landmarks])
                        x_min, x_max = np.min(x_coords), np.max(x_coords)
                        y_min, y_max = np.min(y_coords), np.max(y_coords)

                        width = x_max - x_min
                        height = y_max - y_min

                        x_min -= width * BBOX_PADDING
                        x_max += width * BBOX_PADDING
                        y_min -= height * (BBOX_PADDING + 0.2)  # Extra room for forehead
                        y_max += height * BBOX_PADDING

                        x_min_pix = max(0, int(x_min * w_frame))
                        y_min_pix = max(0, int(y_min * h_frame))
                        x_max_pix = min(w_frame, int(x_max * w_frame))
                        y_max_pix = min(h_frame, int(y_max * h_frame))

                        w_bbox = x_max_pix - x_min_pix
                        h_bbox = y_max_pix - y_min_pix

                        if w_bbox <= 0 or h_bbox <= 0:
                            continue

                        recognized_faces.append({
                            'bbox': (x_min_pix, y_min_pix, w_bbox, h_bbox),
                            'name': best_name,
                            'sim': best_sim,
                            'sim_to_target': sim_to_target,
                            'lm': face_landmarks,
                            'aligned': aligned
                        })
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

    if recognized_faces:
        bb = recognized_faces[0]['bbox']
        prev_position = (bb[0] + bb[2] // 2, bb[1] + bb[3] // 2)

    match_threshold = HOLD_THRESHOLD if locked else LOCK_THRESHOLD
    target_faces = [fd for fd in recognized_faces if fd['sim_to_target'] >= match_threshold]

    # When TRACKING, reject huge position jumps. During REACQUIRING, accept face anywhere
    # in frame so we can re-lock after you leave and come back from a new angle.
    if locked and prev_bbox and target_faces and tracking_state == "TRACKING":
        prev_cx = prev_bbox[0] + prev_bbox[2] // 2
        prev_cy = prev_bbox[1] + prev_bbox[3] // 2
        max_jump = int(max(w_frame, h_frame) * MAX_TRACK_JUMP_RATIO)

        def center_dist(fd):
            cx = fd['bbox'][0] + fd['bbox'][2] // 2
            cy = fd['bbox'][1] + fd['bbox'][3] // 2
            return abs(cx - prev_cx) + abs(cy - prev_cy)

        target_faces = [fd for fd in target_faces if center_dist(fd) <= max_jump]

    locked_face = None
    actions = []

    if target_faces:
        empty_frame_count = 0
        if locked:
            if tracking_state == "REACQUIRING":
                locked_face = max(target_faces, key=lambda fd: fd['sim_to_target'])
            else:
                prev_cx = prev_bbox[0] + prev_bbox[2] // 2
                prev_cy = prev_bbox[1] + prev_bbox[3] // 2
                def dist(fd):
                    cx = fd['bbox'][0] + fd['bbox'][2] // 2
                    cy = fd['bbox'][1] + fd['bbox'][3] // 2
                    return (cx - prev_cx)**2 + (cy - prev_cy)**2
                locked_face = min(target_faces, key=dist)
        else:
            best = max(target_faces, key=lambda fd: fd['sim_to_target'])
            lock_confirm_count += 1
            if lock_confirm_count < LOCK_CONFIRM_FRAMES:
                locked_face = best
            else:
                locked_face = best
                locked = True
                tracking_state = "TRACKING"
                locked_start = datetime.now()
                frames_since_lock = 0
                miss_count = 0
                smoothed_face_x = None
                smoothed_face_y = None
                reset_search_sweep()
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                history_file = f"../data/{TARGET_NAME}_history_{timestamp_str}.txt"
                action_detector.update_baseline(locked_face['lm'], h_frame, w_frame)
                with open(history_file, 'w') as f:
                    f.write(f"Face locking started for {TARGET_NAME.capitalize()} at {datetime.now()}\n")
                    f.write(f"Initial similarity: {locked_face['sim_to_target']:.4f}\n")
                    f.write("-" * 50 + "\n")
                print(f"\n✓ LOCKED onto {TARGET_NAME.capitalize()} (similarity: {locked_face['sim_to_target']:.3f})")
                print(f"  Servo will hold still for ~{LOCK_SETTLE_FRAMES // 30}s, then follow slowly")
                print(f" History saved to: {history_file}")

        if locked:
            actions = action_detector.detect_actions(
                locked_face['lm'], h_frame, w_frame, locked=True
            )
            if actions and history_file:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                with open(history_file, 'a') as f:
                    for action in actions:
                        f.write(f"{timestamp} | {action}\n")

            if tracking_state == "REACQUIRING":
                print("\n✓ Speaker re-acquired")
            reset_search_sweep()

            prev_bbox = locked_face['bbox']
            miss_count = 0
            tracking_state = "TRACKING"
            frames_since_lock += 1
            last_lock_display = {
                'bbox': locked_face['bbox'],
                'sim': locked_face['sim_to_target'],
            }

            face_center_x = locked_face['bbox'][0] + locked_face['bbox'][2] // 2
            face_center_y = locked_face['bbox'][1] + locked_face['bbox'][3] // 2
            status, pan_cmd, tilt_cmd = resolve_movement(face_center_x, face_center_y, w_frame, h_frame)
            if frames_since_lock >= LOCK_SETTLE_FRAMES and status != "CENTERED":
                now = time.time()
                if now - last_track_publish_time >= TRACK_PUBLISH_INTERVAL:
                    publish_movement(
                        status, locked_face['sim_to_target'], face_center_x, face_center_y,
                        w_frame, h_frame, lock_state="TRACKING", pan_cmd=pan_cmd, tilt_cmd=tilt_cmd,
                    )
                    last_published_status = status
    else:
        lock_confirm_count = 0
        if locked:
            miss_count += 1
            if not recognized_faces:
                empty_frame_count += 1
            else:
                empty_frame_count = 0

            should_search = (
                empty_frame_count >= EMPTY_SEARCH_FRAMES
                or miss_count >= MISS_SEARCH_FRAMES
            )

            if should_search:
                if tracking_state != "REACQUIRING":
                    tracking_state = "REACQUIRING"
                    reset_search_sweep()
                    print("\n⚠ Speaker lost — servo scanning until you return (lock held)")
                    if history_file:
                        with open(history_file, 'a') as f:
                            f.write(f"{datetime.now().isoformat()} | Re-acquire scan started\n")
                now = time.time()
                if now - last_search_publish_time >= SEARCH_PUBLISH_INTERVAL:
                    scan_pan, scan_tilt = advance_search_angles()
                    publish_movement(
                        "SEARCH", 0.0, lock_state="REACQUIRING",
                        pan_cmd="SCAN", tilt_cmd="SCAN",
                        pan_angle=scan_pan, tilt_angle=scan_tilt,
                    )
                    last_published_status = "SEARCH"
                    last_search_publish_time = now
            elif last_lock_display and miss_count <= 2:
                locked_face = {
                    'bbox': last_lock_display['bbox'],
                    'sim_to_target': last_lock_display['sim'],
                    'name': TARGET_NAME,
                    'sim': last_lock_display['sim'],
                }
        elif not locked:
            tracking_state = "SEARCHING"
            empty_frame_count = 0
            # Servo stays still until locked — sit in front of camera to acquire lock

    # Candidate for aligned view when searching
    candidate_face = None
    if not locked and recognized_faces:
        candidate_face = max(recognized_faces, key=lambda fd: fd['sim_to_target'])

    # Remove aligned face window - only show main window

    # Draw detected faces
    faces_to_draw = list(recognized_faces)
    if locked and locked_face and locked_face not in faces_to_draw:
        faces_to_draw.append(locked_face)

    for fd in faces_to_draw:
        x, y, w, h = fd['bbox']
        sim_to_target = fd['sim_to_target']

        if locked_face and fd is locked_face:
            color = (0, 255, 0)
            thickness = 4
            text = f"LOCKED: {TARGET_NAME.capitalize()} ({sim_to_target:.3f})"
            for i, action in enumerate(actions[:3]):
                cv2.putText(frame, f"• {action}", (x, y + h + 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        elif sim_to_target >= HOLD_THRESHOLD:
            color = (0, 255, 0)
            thickness = 2
            text = f"{TARGET_NAME.capitalize()} ({sim_to_target:.3f})"
        elif fd['name'] != "Unknown":
            color = (0, 255, 255)  # Yellow
            thickness = 2
            text = f"{fd['name'].capitalize()} ({fd['sim']:.3f})"
        else:
            color = (0, 0, 255)  # Red
            thickness = 2
            text = f"Unknown ({sim_to_target:.3f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, text, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Log other enrolled people
    other_names = {fd['name'] for fd in recognized_faces
                   if fd['name'] != "Unknown" and fd['sim_to_target'] < HOLD_THRESHOLD}
    if locked and other_names:
        print(f"Other enrolled detected: {', '.join(sorted(other_names))}")
        if history_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(history_file, 'a') as f:
                f.write(f"{timestamp} | Others detected: {', '.join(sorted(other_names))}\n")

    # FPS and status
    elapsed = (datetime.now() - start_time).total_seconds()
    fps = fps_counter / elapsed if elapsed > 0 else 0
    face_count = len(recognized_faces) if recognized_faces else (1 if locked and last_lock_display else 0)
    lock_label = 'LOCKED' if locked else 'UNLOCKED'
    if not locked and lock_confirm_count > 0:
        lock_label = f"ACQUIRING {lock_confirm_count}/{LOCK_CONFIRM_FRAMES}"
    elif locked and frames_since_lock < LOCK_SETTLE_FRAMES:
        lock_label = f"LOCKED (settling {LOCK_SETTLE_FRAMES - frames_since_lock}f)"
    status_line = (
        f"FPS: {fps:.1f} | {tracking_state} | {lock_label} | Faces: {face_count}"
    )
    cv2.putText(frame, status_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize to window if open; exit cleanly if user closed the window (X button)
    window_name = 'Face Locking System'
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\nWindow closed — exiting...")
            break
        _, _, window_width, window_height = cv2.getWindowImageRect(window_name)
        if window_width > 0 and window_height > 0:
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            new_width = window_width
            new_height = int(window_width / aspect_ratio)
            if new_height > window_height:
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow(window_name, frame)
    except cv2.error:
        print("\nDisplay window unavailable — exiting...")
        break

    # Send periodic heartbeat
    current_time = time.time()
    if current_time - last_heartbeat_time > HEARTBEAT_INTERVAL:
        send_heartbeat()
        last_heartbeat_time = current_time

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nExiting...")
        break
    elif key == ord('r') and locked:
        print("\nManual lock release")
        locked = False
        tracking_state = "SEARCHING"
        miss_count = 0
        last_lock_display = None
        reset_search_sweep()
        lock_confirm_count = 0
        frames_since_lock = 9999
        smoothed_face_x = None
        smoothed_face_y = None
        publish_movement(
            "STOP", 0.0, lock_state="SEARCHING",
            pan_cmd="STOP", tilt_cmd="STOP",
            pan_angle=SERVO_CENTER, tilt_angle=SERVO_CENTER,
        )
        if history_file and locked_start:
            duration = datetime.now() - locked_start
            with open(history_file, 'a') as f:
                f.write(f"\nManual lock release at {datetime.now()}\n")
                f.write(f"Tracking duration: {str(duration).split('.')[0]}\n")
        history_file = None
        locked_start = None
        action_detector = ActionDetector()

# Cleanup MQTT connection
if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("✓ MQTT connection closed")

cap.release()
cv2.destroyAllWindows()
