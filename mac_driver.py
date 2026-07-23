import socket
import threading
import time
from pynput import keyboard

ROBOT_IP = "100.107.100.56" 
PORT = 5005

COMMAND_INTERVAL_S = 0.10  

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Mac Controller Active - 2-DOF arm teleop.")
print("Base: A/D or Left/Right | Shoulder: W/S or Up/Down | Center: C | Stop: Space | Quit: Esc")

seq = 0
current_command = None
held_commands = []
state_lock = threading.Lock()
send_lock = threading.Lock()
shutdown = threading.Event()

# One-shot commands: sent once on press, never added to the held/heartbeat
# stack (there's nothing to keep "holding" for these).
ONE_SHOT_COMMANDS = {"STOP", "CENTER"}


def send_command(cmd):
    global seq

    with send_lock:
        seq += 1
        payload = f"{seq}:{cmd}"
        sock.sendto(payload.encode(), (ROBOT_IP, PORT))


def command_for_key(key):
    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name

    key_map = {
         "w": "SHOULDER_UP",
        "up": "SHOULDER_UP",
        "s": "SHOULDER_DOWN",
        "down": "SHOULDER_DOWN",
        "a": "BASE_LEFT",
        "left": "BASE_LEFT",
        "d": "BASE_RIGHT",
        "right": "BASE_RIGHT",
        "space": "STOP",
        "c": "CENTER",
        "o": "CLAW_OPEN",
        "p": "CLAW_CLOSE",
        
    }

    return key_map.get(k)


def command_heartbeat():
    while not shutdown.is_set():
        with state_lock:
            command = current_command

        if command is not None:
            send_command(command)

        time.sleep(COMMAND_INTERVAL_S)


def on_press(key):
    global current_command

    if key == keyboard.Key.esc:
        with state_lock:
            current_command = None
            held_commands.clear()

        send_command("STOP")
        shutdown.set()
        print("Exiting...")
        return False

    command = command_for_key(key)

    if command in ONE_SHOT_COMMANDS:
        if command == "STOP":
            with state_lock:
                current_command = None
                held_commands.clear()

        send_command(command)
        return

    if command is None:
        return

    with state_lock:
        # Ignore automatic repeated keydown events.
        if command not in held_commands:
            held_commands.append(command)
            current_command = command
            print(f"Sending: {command}")
            send_command(command)  # Start movement immediately.


def on_release(key):
    global current_command

    command = command_for_key(key)

    if command is None or command in ONE_SHOT_COMMANDS:
        return

    with state_lock:
        if command not in held_commands:
            return

        held_commands.remove(command)
        current_command = held_commands[-1] if held_commands else None

        next_command = current_command

    # Stop immediately if no movement key remains held.
    if next_command is None:
        send_command("STOP")
    else:
        send_command(next_command)


sender_thread = threading.Thread(target=command_heartbeat, daemon=True)
sender_thread.start()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

shutdown.set()
send_command("STOP")
sock.close()