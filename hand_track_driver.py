import csv
import math
import os
import socket
import time
import urllib.request

import cv2
import mediapipe as mp

try:
    import serial
except ImportError:
    serial = None

ROBOT_IP = "100.107.100.56"  # Pi's Tailscale IP -- keep in sync with mac_driver.py
PORT = 5005

# Set True to write straight to the Arduino over USB from this Mac --
# no Pi, Tailscale, or pi_receiver.py needed. Set False to go back to
# the normal UDP-to-Pi path. Find your port with `ls /dev/cu.*` in a
# terminal while the Arduino is plugged in (genuine Mega boards usually
# show up as /dev/cu.usbmodemXXXX, CH340-based clones as
# /dev/cu.wchusbserialXXXX). Keep this in sync with mac_driver.py if you
# use both.
USE_DIRECT_SERIAL = False
SERIAL_PORT = "/dev/cu.usbmodem11201"
SERIAL_BAUD = 115200

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demonstrations")
RECORD_KEY = ord('r')  # press to reset + start recording, press again to stop
CENTER_KEY = ord('c')  # snap all "centerable" joints back to 90

# Grace period after launch before any command is actually sent to the
# arm -- camera/tracking runs and the preview window shows a countdown,
# but nothing gets relayed to the servos until this elapses. Gives you
# time to get your hands in position instead of the arm immediately
# snapping to wherever your hand happens to be when the script starts.
STARTUP_COOLDOWN_S = 3.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

FIST_FINGER_TIPS = [8, 12, 16, 20]
FIST_FINGER_PIPS = [6, 10, 14, 18]
FIST_FINGERS_REQUIRED = 4  # how many of the 4 must be curled to count as a fist

# ---- Joint configuration -------------------------------------------------
# Order here == wire position, and must match pca9685_arm_teleop.ino's
# JOINT_NAMES order exactly: base, shoulder, elbow, wrist_updown, claw.
# The Arduino sketch fans "shoulder" out to two ganged channels, but
# that's invisible from here -- this file only ever deals with the 5
# logical joints.
# wrist_leftright was dropped for now (not needed for basic pick-and-
# place) -- its physical channel/wiring is untouched, this file just no
# longer knows about it. To re-enable it later: add a 6th entry here,
# and re-add a hand-rotation function (see git history / prior version --
# it computed the palm's tilt in the camera plane from the index-knuckle
# to pinky-knuckle vector, scaled by frame width/height to avoid aspect-
# ratio skew, and mapped relative to a calibrated neutral) to drive it.
# Min/max are the real, measured mechanical safe-travel limits for this
# hardware (tuned by hand, not placeholders anymore).
# Direction conventions (confirmed on hardware):
#   base:         higher = turn right,  lower = turn left
#   shoulder:     higher = raise arm,   lower = lower arm
#   elbow:        higher = raise elbow, lower = lower elbow
#   wrist_updown: higher = lower wrist, lower = raise wrist (inverted vs. the others!)
#   claw:         higher = open,        lower = closed (pincers touch at 115)
JOINTS = [
    {"name": "base",         "min": 10,  "max": 200, "centerable": True},
    {"name": "shoulder",     "min": 20,  "max": 100, "centerable": True},
    {"name": "elbow",        "min": 0,   "max": 140, "centerable": True},
    {"name": "wrist_updown", "min": 0,   "max": 180, "centerable": True},
    {"name": "claw",         "min": 100, "max": 170, "home": 120, "centerable": False},
]

JOINT_INDEX = {j["name"]: i for i, j in enumerate(JOINTS)}
_HOME_BY_NAME = {"claw": 120}  # everything else defaults to 90 (centered)
HOME_ANGLES = [_HOME_BY_NAME.get(j["name"], 90) for j in JOINTS]
CENTER_INDICES = [i for i, j in enumerate(JOINTS) if j["centerable"]]  # 'c' key resets only these

SMOOTHING = 0.25

# Pinch distance (normalized by hand size) that counts as fully closed vs
# fully open. Same measurement, reused for both the arm hand's pinch
# (-> elbow) and the claw hand's pinch (-> claw). Tune by watching the
# printed pinch values while you pinch together and spread apart.
PINCH_CLOSED = 0.15
PINCH_OPEN = 2.0

# ---- Shoulder -> elbow/wrist ground-avoidance coupling -------------------
# NOT real inverse kinematics -- this doesn't know link lengths or
# actually compute end-effector height. It's a simple linear heuristic:
# per this build, a high shoulder angle is upright and a low shoulder
# angle swings the arm down toward the ground, so the lower the shoulder
# goes below its most-upright angle, the more this adds a compensating
# "lift" to elbow (and a smaller amount to wrist_updown) automatically,
# on top of whatever you're already commanding by hand/pinch. Tune the
# gains by feel, or set ENABLED = False any time it fights with manual
# control.
GROUND_AVOIDANCE_ENABLED = False
ELBOW_COMPENSATION_GAIN = 0.6  # deg ADDED to elbow per deg shoulder drops (higher elbow = raise, so += lifts it)
WRIST_COMPENSATION_GAIN = 0.3  # deg SUBTRACTED from wrist_updown per deg shoulder drops (wrist is inverted: lower = raise, so -= lifts it)


def shoulder_drop(current_shoulder_angle):
    """How far the shoulder currently is below its most-upright angle
    (>= 0). Pass in the arm's actual current shoulder position
    (angles[JOINT_INDEX["shoulder"]]), not a mid-frame in-progress target.
    """
    shoulder_joint = JOINTS[JOINT_INDEX["shoulder"]]
    upright_angle = shoulder_joint["max"]  # high shoulder degree = more upright, per this build
    return max(0.0, upright_angle - current_shoulder_angle)

HAND_SPLIT_X = 0.5
ARM_HAND_X_RANGE = (0.05, 0.45)
VERTICAL_RANGE = (0.2, 0.8)  # used for shoulder and wrist_updown

# Shared color scheme -- arm hand/joints vs. claw+wrist hand/joints, used
# consistently for landmarks, on-screen labels, and the joint readout.
ARM_COLOR = (0, 255, 0)
CLAW_COLOR = (0, 200, 255)
PAUSE_COLOR = (0, 0, 255)

# Which JOINTS indices belong to each side, for the color-coded readout.
ARM_JOINT_RANGE = (0, 3)    # base, shoulder, elbow
CLAW_JOINT_RANGE = (3, 5)   # wrist_updown, claw

INVERT_BASE = False
INVERT_SHOULDER = True
INVERT_ELBOW = False
INVERT_WRIST_UPDOWN = False
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


def is_fist(landmarks):
    wrist = landmarks[0]
    curled = 0
    for tip_idx, pip_idx in zip(FIST_FINGER_TIPS, FIST_FINGER_PIPS):
        tip_dist = landmark_dist(landmarks[tip_idx], wrist)
        pip_dist = landmark_dist(landmarks[pip_idx], wrist)
        if tip_dist < pip_dist:
            curled += 1
    return curled >= FIST_FINGERS_REQUIRED


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_seq = 0
_last_print_time = 0.0

_serial_conn = None
if USE_DIRECT_SERIAL:
    if serial is None:
        raise RuntimeError("USE_DIRECT_SERIAL is True but pyserial isn't installed -- run: pip install pyserial")
    _serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    time.sleep(2)  # let the Arduino reboot after the port opens

PRINT_COMMANDS = True
PRINT_INTERVAL_S = 0.2


def print_joint_angles(angles):
    """Print every joint's current commanded angle on one line, e.g.:
    base=90  shoulder=90  elbow=90  wrist_updown=90  claw=150
    Handy for sanity-checking from the terminal without reading the overlay,
    same idea as the Arduino sketch's own printAngles()/"?" query.
    """
    print("  ".join(f"{j['name']}={a:.0f}" for j, a in zip(JOINTS, angles)))


def send_targets(angles, elbow_pinch=None, claw_pinch=None):
    global _seq, _last_print_time
    _seq += 1
    values_str = ",".join(str(int(round(a))) for a in angles)

    if USE_DIRECT_SERIAL:
        _serial_conn.write((values_str + "\n").encode())
    else:
        sock.sendto(f"{_seq}:{values_str}".encode(), (ROBOT_IP, PORT))

    if PRINT_COMMANDS:
        now = time.time()
        if now - _last_print_time >= PRINT_INTERVAL_S:
            extras = []
            extras.append(f"elbow_pinch={elbow_pinch:.3f}" if elbow_pinch is not None else "elbow_pinch=(no hand)")
            extras.append(f"claw_pinch={claw_pinch:.3f}" if claw_pinch is not None else "claw_pinch=(no hand)")
            print(f"-> {values_str}  " + "  ".join(extras))
            print_joint_angles(angles)
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


def compute_arm_hand(landmarks):
    """Left-half hand: position -> base + shoulder, pinch -> elbow.

    Pure function -- returns the proposed angles/pinch, doesn't write
    into any shared target array. That way the caller can compute this
    every frame (keeping the reading live) while still deciding for
    itself whether to actually commit it to the servos this frame (e.g.
    skip committing while this hand is paused/fisted).
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]

    base = JOINTS[JOINT_INDEX["base"]]
    shoulder = JOINTS[JOINT_INDEX["shoulder"]]
    elbow = JOINTS[JOINT_INDEX["elbow"]]

    base_lo, base_hi = (base["max"], base["min"]) if not INVERT_BASE else (base["min"], base["max"])
    shoulder_lo, shoulder_hi = (shoulder["min"], shoulder["max"]) if not INVERT_SHOULDER else (shoulder["max"], shoulder["min"])
    elbow_lo, elbow_hi = (elbow["min"], elbow["max"]) if not INVERT_ELBOW else (elbow["max"], elbow["min"])

    base_angle = remap(wrist.x, *ARM_HAND_X_RANGE, base_lo, base_hi)
    shoulder_angle = remap(wrist.y, *VERTICAL_RANGE, shoulder_lo, shoulder_hi)

    hand_scale = landmark_dist(wrist, middle_mcp) or 1e-6
    pinch = landmark_dist(thumb_tip, index_tip) / hand_scale
    elbow_angle = remap(pinch, PINCH_CLOSED, PINCH_OPEN, elbow_lo, elbow_hi)

    return base_angle, shoulder_angle, elbow_angle, pinch


def compute_wrist_claw_hand(landmarks):
    """Right-half hand: up/down position -> wrist tilt, pinch -> claw.

    Pure function, same reasoning as compute_arm_hand(). wrist_leftright
    (rotation-driven) was removed -- see the JOINTS comment above for how
    to bring it back.
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]

    wrist_ud = JOINTS[JOINT_INDEX["wrist_updown"]]
    claw = JOINTS[JOINT_INDEX["claw"]]

    ud_lo, ud_hi = (wrist_ud["min"], wrist_ud["max"]) if not INVERT_WRIST_UPDOWN else (wrist_ud["max"], wrist_ud["min"])

    wrist_ud_angle = remap(wrist.y, *VERTICAL_RANGE, ud_lo, ud_hi)

    hand_scale = landmark_dist(wrist, middle_mcp) or 1e-6
    pinch = landmark_dist(thumb_tip, index_tip) / hand_scale
    claw_angle = remap(pinch, PINCH_CLOSED, PINCH_OPEN, claw["min"], claw["max"])

    return wrist_ud_angle, claw_angle, pinch


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
    print(f"Mode: {'DIRECT SERIAL to ' + SERIAL_PORT if USE_DIRECT_SERIAL else 'UDP to Pi at ' + ROBOT_IP}")
    print("ARM hand (LEFT half): position -> base/shoulder, pinch -> elbow.")
    print("CLAW hand (RIGHT half): up/down -> wrist tilt, pinch -> claw.")
    print("Press 'r' to reset to home and start/stop recording a demonstration to CSV.")
    print("Press 'c' to snap all arm joints back to center (claw stays put).")
    print("Make a fist with either hand to pause just that hand's axis, open it to resume.")
    print("(Readings keep updating live even while paused -- only sending to the servos stops.)")
    print("Press Esc or q, or close the window, to quit.")
    print(f"Startup cooldown: no commands will be sent for the first {STARTUP_COOLDOWN_S:.0f}s.")

    cooldown_announced = False

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

            h, w, _ = frame.shape

            targets = list(angles)
            elbow_pinch_display = None
            claw_pinch_display = None
            arm_paused = False
            claw_paused = False

            # Computed once per frame from the arm's actual current
            # position (not this frame's in-progress target), so both
            # hand branches below can use the same value regardless of
            # which hand(s) are visible this frame.
            drop = shoulder_drop(angles[JOINT_INDEX["shoulder"]]) if GROUND_AVOIDANCE_ENABLED else 0.0

            for hand_landmarks in result.hand_landmarks:
                wrist_lm = hand_landmarks[0]
                fisted = is_fist(hand_landmarks)

                if wrist_lm.x < HAND_SPLIT_X:
                    # Always compute -- keeps the pinch reading live even
                    # while paused -- but only commit into targets[] (and
                    # therefore the servos) when this hand isn't fisted.
                    base_a, shoulder_a, elbow_a, elbow_pinch_display = compute_arm_hand(hand_landmarks)
                    if fisted:
                        arm_paused = True
                        draw_landmarks(frame, hand_landmarks, "ARM PAUSED", PAUSE_COLOR)
                    else:
                        targets[JOINT_INDEX["base"]] = base_a
                        targets[JOINT_INDEX["shoulder"]] = shoulder_a
                        elbow_a += drop * ELBOW_COMPENSATION_GAIN
                        targets[JOINT_INDEX["elbow"]] = clamp_to_joint(elbow_a, JOINTS[JOINT_INDEX["elbow"]])
                        draw_landmarks(frame, hand_landmarks, "ARM", ARM_COLOR)
                else:
                    wrist_ud_a, claw_a, claw_pinch_display = compute_wrist_claw_hand(hand_landmarks)
                    if fisted:
                        claw_paused = True
                        draw_landmarks(frame, hand_landmarks, "CLAW PAUSED", PAUSE_COLOR)
                    else:
                        wrist_ud_a -= drop * WRIST_COMPENSATION_GAIN
                        targets[JOINT_INDEX["wrist_updown"]] = clamp_to_joint(wrist_ud_a, JOINTS[JOINT_INDEX["wrist_updown"]])
                        targets[JOINT_INDEX["claw"]] = claw_a
                        draw_landmarks(frame, hand_landmarks, "CLAW", CLAW_COLOR)

            cooldown_remaining = STARTUP_COOLDOWN_S - (time.time() - start_time)

            if cooldown_remaining <= 0:
                if not cooldown_announced:
                    print("Cooldown complete -- now sending commands to the arm.")
                    cooldown_announced = True

                for i, joint in enumerate(JOINTS):
                    smoothed = angles[i] + SMOOTHING * (targets[i] - angles[i])
                    angles[i] = clamp_to_joint(smoothed, joint)

                send_targets(angles, elbow_pinch=elbow_pinch_display, claw_pinch=claw_pinch_display)
                log_angles(angles)

            cv2.line(frame, (int(w * HAND_SPLIT_X), 0), (int(w * HAND_SPLIT_X), h), (80, 80, 80), 1)

            cv2.putText(frame, "POSITION=BASE/SHOULDER  PINCH=ELBOW", (int(w * 0.02), h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ARM_COLOR, 1)
            cv2.putText(frame, "HEIGHT=WRIST TILT  PINCH=CLAW", (int(w * 0.52), h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLAW_COLOR, 1)

            if elbow_pinch_display is not None:
                cv2.putText(frame, f"elbow_pinch={elbow_pinch_display:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ARM_COLOR, 2)
            if claw_pinch_display is not None:
                cv2.putText(frame, f"claw_pinch={claw_pinch_display:.2f}", (int(w * 0.55), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLAW_COLOR, 2)

            # Joint readout, color-coded to match which hand drives which
            # joints: arm-hand joints in ARM_COLOR, wrist/claw-hand joints
            # in CLAW_COLOR.
            arm_lo, arm_hi = ARM_JOINT_RANGE
            claw_lo, claw_hi = CLAW_JOINT_RANGE
            arm_angle_str = "  ".join(f"{JOINTS[i]['name']}={angles[i]:.0f}" for i in range(arm_lo, arm_hi))
            claw_angle_str = "  ".join(f"{JOINTS[i]['name']}={angles[i]:.0f}" for i in range(claw_lo, claw_hi))
            cv2.putText(frame, arm_angle_str, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ARM_COLOR, 2)
            cv2.putText(frame, claw_angle_str, (int(w * 0.52), h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLAW_COLOR, 2)

            if cooldown_remaining > 0:
                countdown_text = f"STARTING IN {cooldown_remaining:.1f}s"
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                cv2.putText(frame, countdown_text, ((w - text_size[0]) // 2, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, PAUSE_COLOR, 3)

            if recording:
                cv2.circle(frame, (w - 25, 25), 8, PAUSE_COLOR, -1)
                cv2.putText(frame, "REC", (w - 75, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PAUSE_COLOR, 2)
            if arm_paused:
                cv2.putText(frame, "ARM PAUSED", (int(w * 0.05), 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PAUSE_COLOR, 2)
            if claw_paused:
                cv2.putText(frame, "CLAW PAUSED", (int(w * 0.58), 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PAUSE_COLOR, 2)
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
                    angles = list(HOME_ANGLES)
                    send_targets(angles)
                    start_recording()
            if key == CENTER_KEY:
                for idx in CENTER_INDICES:
                    angles[idx] = 90
                send_targets(angles)
                print("Centered base/shoulder/elbow/wrist.")

    if recording:
        stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    if _serial_conn is not None:
        _serial_conn.close()


if __name__ == "__main__":
    main()