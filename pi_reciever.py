import socket
import serial
import threading
import time
import csv


try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
except:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)

time.sleep(2)  

UDP_IP = "0.0.0.0"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, PORT))
sock.setblocking(False)  

print(f"Pi Listener active on port {PORT}. Awaiting arm commands from Mac...")

# --- Joint angle listener -------------------------------------------------
# The main loop below no longer blocks on sock.recvfrom(), but this still
# runs in its own thread since it's driven by the serial port instead of
# the socket. Keeps draining the serial line, printing angles and logging
# them with a timestamp -- useful now for a sanity check, and later for
# lining joint angles up against camera frames for imitation learning.
LOG_PATH = "joint_log.csv"
FLUSH_INTERVAL_S = 0.5 


def parse_angles(line):
    # Expects a line like: "Base: 93  Shoulder: 120"
    try:
        base = int(line.split("Base:")[1].split("Shoulder:")[0].strip())
        shoulder = int(line.split("Shoulder:")[1].strip())
        return base, shoulder
    except (IndexError, ValueError):
        return None


def serial_reader():
    last_flush = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            angles = parse_angles(line)
            if angles is None:
                continue  

            base, shoulder = angles
            ts = time.time()
            print(f"[{ts:.3f}] base={base}  shoulder={shoulder}")
            writer.writerow([ts, base, shoulder])
            if ts - last_flush >= FLUSH_INTERVAL_S:
                f.flush()
                last_flush = ts


reader_thread = threading.Thread(target=serial_reader, daemon=True)
reader_thread.start()
# ---------------------------------------------------------------------------

try:
    last_seq = -1

    while True:
        latest = None
        while True:
            try:
                latest, _ = sock.recvfrom(1024)
            except BlockingIOError:
                break

        if latest is None:
            time.sleep(0.01)
            continue

        try:
            seq_str, command = latest.decode().split(":", 1)
            seq = int(seq_str)
        except ValueError:
            continue  # malformed packet ignore

        if seq <= last_seq:
            continue  # discard
        last_seq = seq

        if command == "BASE_LEFT":
            ser.write(b'a')
        elif command == "BASE_RIGHT":
            ser.write(b'd')
        elif command == "SHOULDER_UP":
            ser.write(b'w')
        elif command == "SHOULDER_DOWN":
            ser.write(b's')
        elif command == "CENTER":
            ser.write(b'c')
        elif command == "CLAW_OPEN":
             ser.write(b'o')
        elif command == "CLAW_CLOSE":
            ser.write(b'p')
        elif command == "STOP":
            pass 

except KeyboardInterrupt:
    print("\nShutting down listener.")
finally:
    ser.close()