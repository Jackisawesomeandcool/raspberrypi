import socket
import threading
import time
from pynput import keyboard

try:
    import serial
except ImportError:
    serial = None

ROBOT_IP = "100.107.100.56"  # Pi's Tailscale IP
PORT = 5005

# Set True to write straight to the Arduino over USB from this Mac --
# no Pi, Tailscale, or pi_receiver.py needed. Set False to go back to
# the normal UDP-to-Pi path. Find your port with `ls /dev/cu.*` in a
# terminal while the Arduino is plugged in (genuine Mega boards usually
# show up as /dev/cu.usbmodemXXXX, CH340-based clones as
# /dev/cu.wchusbserialXXXX).
USE_DIRECT_SERIAL = False
SERIAL_PORT = "/dev/cu.usbmodem11201"
SERIAL_BAUD = 115200

COMMAND_INTERVAL_S = 0.10  # Send 10 targets per second while a key is held.
STEP = 5                   # degrees nudged per tick

# ---- Joint configuration -----------------------------------------------
# Index here == position in the wire payload == logical joint on the
# Arduino side (must stay in the same order as pca9685_arm_teleop.ino's
# JOINT_NAMES -- the sketch fans "shoulder" out to two ganged channels,
# but that's invisible from here, it's still just one entry/one value).
# wrist_leftright was dropped for now (not needed for basic pick-and-place)
# -- its physical channel/wiring is untouched, this file just no longer
# knows about it. Add it back as a 6th entry here to re-enable it later.
# Min/max are the real, measured mechanical safe-travel limits for this
# hardware (tuned by hand, not placeholders anymore).
# Direction conventions (confirmed on hardware):
#   base:         higher = turn right,  lower = turn left
#   shoulder:     higher = raise arm,   lower = lower arm
#   elbow:        higher = raise elbow, lower = lower elbow
#   wrist_updown: higher = lower wrist, lower = raise wrist (inverted vs. the others!)
#   claw:         higher = open,        lower = closed (pincers touch at 115)
JOINTS = [
    {"name": "base",         "min": 10,  "max": 190, "home": 90,  "centerable": True},
    {"name": "shoulder",     "min": 30,  "max": 100, "home": 90,  "centerable": True},
    {"name": "elbow",        "min": 0,   "max": 160, "home": 90,  "centerable": True},
    {"name": "wrist_updown", "min": 0,   "max": 180, "home": 90,  "centerable": True},
    {"name": "claw",         "min": 115, "max": 150, "home": 120, "centerable": False},
]

# Each held-key command nudges one or more joints by +/-STEP per tick
# (or +/-STEP*multiplier for a compensation entry -- see apply_command()).
# Directions here match the confirmed hardware convention: base higher =
# right, shoulder/elbow higher = raise, wrist_updown higher = LOWER
# (inverted vs. the others), claw higher = open.
CONTROLS = {
    "BASE_LEFT":         [(0, -1)],
    "BASE_RIGHT":        [(0, +1)],
    "SHOULDER_UP":       [(1, +1)],
    "SHOULDER_DOWN":     [(1, -1)],
    "ELBOW_UP":          [(2, +1)],
    "ELBOW_DOWN":        [(2, -1)],
    "WRIST_UPDOWN_UP":   [(3, -1)],
    "WRIST_UPDOWN_DOWN": [(3, +1)],
    "CLAW_OPEN":         [(4, +1)],
    "CLAW_CLOSE":        [(4, -1)],
}

# Ground-avoidance coupling: every shoulder nudge also nudges elbow (and
# a smaller amount of wrist_updown) to compensate, same idea as
# hand_tracking_driver.py's continuous version -- not real inverse
# kinematics, just "raise elbow/wrist a bit whenever the shoulder lowers,
# and undo that as it rises back up." Elbow moves opposite to shoulder's
# own direction (higher elbow = raise); wrist_updown moves the *same*
# direction as shoulder since wrist_updown's convention is inverted
# (higher = lower wrist). Set False, or dial the gains down, if it fights
# manual control.
GROUND_AVOIDANCE_ENABLED = True
ELBOW_COMPENSATION_GAIN = 0.5
WRIST_COMPENSATION_GAIN = 0.5

if GROUND_AVOIDANCE_ENABLED:
    CONTROLS["SHOULDER_UP"].append((2, -1, ELBOW_COMPENSATION_GAIN))
    CONTROLS["SHOULDER_UP"].append((3, +1, WRIST_COMPENSATION_GAIN))
    CONTROLS["SHOULDER_DOWN"].append((2, +1, ELBOW_COMPENSATION_GAIN))
    CONTROLS["SHOULDER_DOWN"].append((3, -1, WRIST_COMPENSATION_GAIN))

KEY_MAP = {
    "w": "SHOULDER_UP", "up": "SHOULDER_UP",
    "s": "SHOULDER_DOWN", "down": "SHOULDER_DOWN",
    "a": "BASE_LEFT", "left": "BASE_LEFT",
    "d": "BASE_RIGHT", "right": "BASE_RIGHT",
    "e": "ELBOW_UP",
    "f": "ELBOW_DOWN",
    "i": "WRIST_UPDOWN_UP",
    "k": "WRIST_UPDOWN_DOWN",
    "o": "CLAW_OPEN",
    "p": "CLAW_CLOSE",
    "space": "STOP",
    "c": "CENTER",
}

CENTER_JOINTS = [i for i, j in enumerate(JOINTS) if j["centerable"]]
angles = [j["home"] for j in JOINTS]
# --------------------------------------------------------------------------

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

_serial_conn = None
if USE_DIRECT_SERIAL:
    if serial is None:
        raise RuntimeError("USE_DIRECT_SERIAL is True but pyserial isn't installed -- run: pip install pyserial")
    _serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    time.sleep(2)  # let the Arduino reboot after the port opens

print(f"Mac Controller Active - {len(JOINTS)}-joint arm teleop (absolute-angle protocol).")
print(f"Mode: {'DIRECT SERIAL to ' + SERIAL_PORT if USE_DIRECT_SERIAL else 'UDP to Pi at ' + ROBOT_IP}")
print("Base: A/D or Left/Right | Shoulder: W/S or Up/Down | Elbow: E/F")
print("Wrist up/down: I/K | Claw: O=open P=close")
print("Center: C | Stop: Space | Quit: Esc")

seq = 0
send_lock = threading.Lock()
state_lock = threading.Lock()
shutdown = threading.Event()

held_commands = []
current_command = None


def send_targets():
    global seq
    with send_lock:
        seq += 1
        values_str = ",".join(str(a) for a in angles)

        if USE_DIRECT_SERIAL:
            _serial_conn.write((values_str + "\n").encode())
        else:
            sock.sendto(f"{seq}:{values_str}".encode(), (ROBOT_IP, PORT))


def command_for_key(key):
    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name

    return KEY_MAP.get(k)


def apply_command(command):
    """Nudge whichever joint(s) this command drives by one STEP, scaled
    per-entry by an optional multiplier (3rd tuple element) -- used for
    ground-avoidance compensation nudges that should move less than the
    STEP applied to the primary joint.
    """
    for entry in CONTROLS.get(command, []):
        joint_idx, direction = entry[0], entry[1]
        multiplier = entry[2] if len(entry) > 2 else 1.0
        j = JOINTS[joint_idx]
        delta = round(direction * STEP * multiplier)
        angles[joint_idx] = max(j["min"], min(j["max"], angles[joint_idx] + delta))


def command_heartbeat():
    while not shutdown.is_set():
        with state_lock:
            if current_command is not None:
                apply_command(current_command)
            send_targets()
        time.sleep(COMMAND_INTERVAL_S)


def on_press(key):
    global current_command

    if key == keyboard.Key.esc:
        with state_lock:
            current_command = None
            held_commands.clear()
        shutdown.set()
        print("Exiting...")
        return False

    command = command_for_key(key)

    if command == "CENTER":
        with state_lock:
            for idx in CENTER_JOINTS:
                angles[idx] = JOINTS[idx]["home"]
            send_targets()
        return

    if command == "STOP":
        with state_lock:
            current_command = None
            held_commands.clear()
        return

    if command is None:
        return

    with state_lock:
        # Ignore automatic repeated keydown events.
        if command not in held_commands:
            held_commands.append(command)
            current_command = command
            print(f"Jogging: {command}")


def on_release(key):
    global current_command

    command = command_for_key(key)
    if command is None or command in ("STOP", "CENTER"):
        return

    with state_lock:
        if command not in held_commands:
            return
        held_commands.remove(command)
        current_command = held_commands[-1] if held_commands else None


sender_thread = threading.Thread(target=command_heartbeat, daemon=True)
sender_thread.start()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

shutdown.set()
sock.close()
if _serial_conn is not None:
    _serial_conn.close()