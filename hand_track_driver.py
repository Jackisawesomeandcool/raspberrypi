"""

Mapping:
    left hand  left/right -> base
    left hand  up/down    -> shoulder
    right hand pinch      -> claw (pinched = closed, spread = open)
    right hand up/down    -> wrist (once WRIST_ENABLED, see below)

Install once:
    pip install mediapipe opencv-python --break-system-packages

Run:
    python3 hand_tracking_driver.py
The hand landmark model (~10MB) downloads automatically to this script's
folder on first run. Press Esc or q, or just close the window, to quit.
"""

import csv
import math
import os
import socket
import time
import urllib.request

import cv2
import mediapipe as mp

ROBOT_IP = "100.107.100.56"  # Pi's Tailscale IP -- keep in sync with mac_driver.py
PORT = 5005

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demonstrations")
RECORD_KEY = ord('r')  # press to reset + start recording, press again to stop

# Standard MediaPipe hand skeleton topology (21 landmarks), hardcoded here
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  
]

WRIST_ENABLED = False

JOINTS = [
    {"name": "base",     "min": 0,  "max": 180},
    {"name": "shoulder", "min": 0,  "max": 180},
]
if WRIST_ENABLED:
    JOINTS.append({"name": "wrist", "min": 0, "max": 180})
JOINTS += [
    {"name": "claw1", "min": 90, "max": 150},
    {"name": "claw2", "min": 30, "max": 90},
]

JOINT_INDEX = {j["name"]: i for i, j in enumerate(JOINTS)}
_HOME_BY_NAME = {"claw1": 150, "claw2": 30}  # everything else defaults to 90 (centered)
HOME_ANGLES = [_HOME_BY_NAME.get(j["name"], 90) for j in JOINTS]

# Exponential smoothing factor applied to angles every frame. Lower =
# smoother but laggier, higher = snappier but jitterier.
SMOOTHING = 0.15

# Pinch distance (normalized by hand size, so it doesn't depend on how
# far your hand is from the camera) that counts as fully closed vs fully
# open. Tune these by watching the printed pinch value while you pinch
# your fingers together and spread them apart.
PINCH_CLOSED = 0.15
PINCH_OPEN = 2.0

# Which half of the (mirrored) frame each hand needs to be on to control
# the arm vs the wrist/claw. Arm hand's usable horizontal range is a
# sub-range of the left half so it doesn't need to cross the midline
# (and risk flipping roles) to reach full base rotation.
HAND_SPLIT_X = 0.5
ARM_HAND_X_RANGE = (0.05, 0.45)
VERTICAL_RANGE = (0.2, 0.8)  # used for shoulder, and wrist once enabled

# If base/shoulder/wrist move opposite to your hand once this is
# running, swap True<->False here rather than rederiving the math.
INVERT_BASE = False
INVERT_SHOULDER = False
INVERT_WRIST = False
# ---------------------------------------------------------------------------


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def remap(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def landmark_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def clamp_to_joint(value, joint):
    return max(joint["min"], min(joint["max"], value))


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_seq = 0
_last_print_time = 0.0

# Print every outgoing command locally, throttled so it stays readable
# even though frames (and sends) happen 15-30 times a second. This lets
# you see exactly what would be sent to the Pi without pi_receiver.py
# running at all -- handy for testing the tracking/mapping on its own.
PRINT_COMMANDS = True
PRINT_INTERVAL_S = 0.2


def send_targets(angles, pinch=None):
    global _seq, _last_print_time
    _seq += 1
    payload = f"{_seq}:" + ",".join(str(int(round(a))) for a in angles)
    sock.sendto(payload.encode(), (ROBOT_IP, PORT))

    if PRINT_COMMANDS:
        now = time.time()
        if now - _last_print_time >= PRINT_INTERVAL_S:
            pinch_str = f"  pinch={pinch:.3f}" if pinch is not None else "  pinch=(no hand)"
            print(f"-> {payload}{pinch_str}")
            _last_print_time = now


_log_file = None
_log_writer = None
recording = False


def start_recording():
    """Reset every joint to its home position and begin logging."""
    global _log_file, _log_writer, recording

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"demo_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    _log_file = open(path, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["timestamp"] + [j["name"] for j in JOINTS])
    recording = True
    print(f"Recording started -> {path}")


def stop_recording():
    global _log_file, _log_writer, recording
    if _log_file is not None:
        _log_file.close()
    _log_file = None
    _log_writer = None
    recording = False
    print("Recording stopped.")


def log_angles(angles):
    if recording and _log_writer is not None:
        _log_writer.writerow([time.time()] + [round(a, 1) for a in angles])


def apply_arm_hand(landmarks, targets):
    """Left-half hand: drives base + shoulder."""
    wrist = landmarks[0]

    base = JOINTS[JOINT_INDEX["base"]]
    shoulder = JOINTS[JOINT_INDEX["shoulder"]]

    base_lo, base_hi = (base["max"], base["min"]) if not INVERT_BASE else (base["min"], base["max"])
    shoulder_lo, shoulder_hi = (shoulder["min"], shoulder["max"]) if not INVERT_SHOULDER else (shoulder["max"], shoulder["min"])

    targets[JOINT_INDEX["base"]] = remap(wrist.x, *ARM_HAND_X_RANGE, base_lo, base_hi)
    targets[JOINT_INDEX["shoulder"]] = remap(wrist.y, *VERTICAL_RANGE, shoulder_lo, shoulder_hi)


def apply_wrist_claw_hand(landmarks, targets):
    """Right-half hand: drives the claw (pinch), and the wrist once enabled."""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]  # base knuckle of the middle finger

    hand_scale = landmark_dist(wrist, middle_mcp) or 1e-6
    pinch = landmark_dist(thumb_tip, index_tip) / hand_scale

    claw1, claw2 = JOINTS[JOINT_INDEX["claw1"]], JOINTS[JOINT_INDEX["claw2"]]
    # Pinched (small distance) -> claw1 low end (closed); spread -> high end (open).
    targets[JOINT_INDEX["claw1"]] = remap(pinch, PINCH_CLOSED, PINCH_OPEN, claw1["min"], claw1["max"])
    # Claw2 is mirrored, same as in mac_driver.py.
    targets[JOINT_INDEX["claw2"]] = remap(pinch, PINCH_CLOSED, PINCH_OPEN, claw2["max"], claw2["min"])

    if WRIST_ENABLED:
        wrist_joint = JOINTS[JOINT_INDEX["wrist"]]
        wrist_lo, wrist_hi = (wrist_joint["min"], wrist_joint["max"]) if not INVERT_WRIST else (wrist_joint["max"], wrist_joint["min"])
        targets[JOINT_INDEX["wrist"]] = remap(wrist.y, *VERTICAL_RANGE, wrist_lo, wrist_hi)

    return pinch


def draw_landmarks(frame, landmarks, label, color):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 1)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 4, color, -1)
    cv2.putText(frame, label, (points[0][0] - 20, points[0][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different camera index.")

    angles = list(HOME_ANGLES)
    start_time = time.time()

    window_name = "Hand Tracking Teleop"
    cv2.namedWindow(window_name)

    print("Hand-tracking teleop active.")
    print("Keep your ARM hand on the LEFT half of frame: left/right -> base, up/down -> shoulder.")
    print("Keep your CLAW hand on the RIGHT half of frame: pinch -> claw open/close"
          + (", up/down -> wrist." if WRIST_ENABLED else " (wrist not enabled yet)."))
    print("Press 'r' to reset to home and start/stop recording a demonstration to CSV.")
    print("Press Esc or q, or close the window, to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror so hand-left feels like screen-left
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Default: hold each joint's current angle unless a hand is
            # actively driving it this frame.
            targets = list(angles)
            pinch_display = None

            for hand_landmarks in result.hand_landmarks:
                wrist_lm = hand_landmarks[0]
                if wrist_lm.x < HAND_SPLIT_X:
                    apply_arm_hand(hand_landmarks, targets)
                    draw_landmarks(frame, hand_landmarks, "ARM", (0, 255, 0))
                else:
                    pinch_display = apply_wrist_claw_hand(hand_landmarks, targets)
                    draw_landmarks(frame, hand_landmarks, "CLAW", (0, 200, 255))

            for i, joint in enumerate(JOINTS):
                smoothed = angles[i] + SMOOTHING * (targets[i] - angles[i])
                angles[i] = clamp_to_joint(smoothed, joint)

            # Always send, even with no hands in frame -- this just
            # re-sends the last known angles (holds position), the same
            # idea as mac_driver.py's heartbeat.
            send_targets(angles, pinch=pinch_display)
            log_angles(angles)  # no-op unless a recording is active

            h, w, _ = frame.shape
            cv2.line(frame, (int(w * HAND_SPLIT_X), 0), (int(w * HAND_SPLIT_X), h), (80, 80, 80), 1)
            if pinch_display is not None:
                cv2.putText(frame, f"pinch={pinch_display:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            angle_str = "  ".join(f"{j['name']}={angles[i]:.0f}" for i, j in enumerate(JOINTS))
            cv2.putText(frame, angle_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if recording:
                cv2.circle(frame, (w - 25, 25), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (w - 75, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            if key == 27 or key == ord('q') or window_closed:
                print("Exiting...")
                break
            if key == RECORD_KEY:
                if recording:
                    stop_recording()
                else:
                    angles = list(HOME_ANGLES)  # snap instantly, don't wait for smoothing
                    send_targets(angles)
                    start_recording()

    if recording:
        stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()