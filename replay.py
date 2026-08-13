"""Replay a recorded joint-angle demonstration over UDP.

The outgoing packet format matches hand_track_driver.py:

    sequence:base,shoulder,claw1,claw2

Example:

    python replay.py demonstrations/demo_20260723_135244.csv --dry-run
    python replay.py demonstrations/demo_20260723_135244.csv
"""

import argparse
import csv
import socket
import time
from pathlib import Path


ROBOT_IP = "100.107.100.56"
PORT = 5005

EXPECTED_COLUMNS = ["timestamp", "base", "shoulder", "claw1", "claw2"]
JOINT_LIMITS = {
    "base": (0.0, 180.0),
    "shoulder": (0.0, 180.0),
    "claw1": (90.0, 150.0),
    "claw2": (30.0, 90.0),
}
HOME_ANGLES = [90, 90, 150, 30]
FINAL_HOLD_SENDS = 5
FINAL_HOLD_INTERVAL_S = 0.05


def load_demo(path):
    rows = []

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Expected CSV columns {EXPECTED_COLUMNS}, got {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp"])
                angles = [float(row[name]) for name in EXPECTED_COLUMNS[1:]]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number} contains a non-numeric value"
                ) from exc

            rows.append((timestamp, angles))

    if not rows:
        raise ValueError("The demonstration contains no data rows")

    validate_demo(rows)
    return rows


def validate_demo(rows):
    previous_timestamp = None

    for index, (timestamp, angles) in enumerate(rows, start=1):
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"Data row {index} has a non-increasing timestamp: {timestamp}"
            )
        previous_timestamp = timestamp

        for name, angle in zip(EXPECTED_COLUMNS[1:], angles):
            lower, upper = JOINT_LIMITS[name]
            if not lower <= angle <= upper:
                raise ValueError(
                    f"Data row {index}: {name}={angle} is outside "
                    f"[{lower}, {upper}]"
                )

        claw1, claw2 = angles[2], angles[3]
        if abs((claw1 + claw2) - 180.0) > 0.2:
            raise ValueError(
                f"Data row {index}: claw1 + claw2 must equal 180 degrees"
            )


def format_payload(sequence, angles):
    rounded_angles = ",".join(str(int(round(angle))) for angle in angles)
    return f"{sequence}:{rounded_angles}"


def print_summary(path, rows, speed):
    duration = rows[-1][0] - rows[0][0]
    replay_duration = duration / speed
    first_angles = format_payload(0, rows[0][1]).split(":", 1)[1]
    last_angles = format_payload(0, rows[-1][1]).split(":", 1)[1]

    print(f"Demo: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Recorded duration: {duration:.3f} s")
    print(f"Replay duration: {replay_duration:.3f} s (speed={speed:g}x)")
    print(f"First target: {first_angles}")
    print(f"Last target:  {last_angles}")
    print(f"Expected home: {','.join(map(str, HOME_ANGLES))}")


def send_targets(sock, sequence, angles, robot_ip, port):
    payload = format_payload(sequence, angles)
    sock.sendto(payload.encode(), (robot_ip, port))
    return payload


def replay(rows, speed, robot_ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    first_timestamp = rows[0][0]
    start = time.monotonic()

    # A time-based initial sequence avoids being rejected as older than a
    # previous Mac sender whose sequence counter started from zero.
    sequence = int(time.time() * 1000)
    last_angles = rows[0][1]

    try:
        for timestamp, angles in rows:
            deadline = start + (timestamp - first_timestamp) / speed

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))

            send_targets(sock, sequence, angles, robot_ip, port)
            sequence += 1
            last_angles = angles

        # UDP is unreliable, so repeat the final target briefly to make the
        # intended hold position more likely to reach the receiver.
        for _ in range(FINAL_HOLD_SENDS):
            send_targets(sock, sequence, last_angles, robot_ip, port)
            sequence += 1
            time.sleep(FINAL_HOLD_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nReplay interrupted. No further targets will be sent.")
        print("The arm should retain its last commanded target.")
        raise
    finally:
        sock.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a recorded arm demonstration over UDP."
    )
    parser.add_argument("demo", type=Path, help="path to a demonstration CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarize the CSV without sending UDP packets",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="replay speed multiplier (default: 1.0)",
    )
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero")
    if not args.demo.is_file():
        raise SystemExit(f"Demonstration file not found: {args.demo}")

    try:
        rows = load_demo(args.demo)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid demonstration: {exc}") from exc

    print_summary(args.demo, rows, args.speed)

    if args.dry_run:
        print("Dry run complete. No UDP packets were sent.")
        return

    print()
    print("Before continuing:")
    print("  1. Stop hand_track_driver.py or any other arm controller.")
    print("  2. Put the arm at HOME and clear the workspace.")
    print("  3. Keep the hardware emergency stop within reach.")
    confirmation = input("Type REPLAY to start: ")
    if confirmation != "REPLAY":
        print("Replay cancelled.")
        return

    print(f"Replaying to {args.robot_ip}:{args.port} ...")
    try:
        replay(rows, args.speed, args.robot_ip, args.port)
    except KeyboardInterrupt:
        raise SystemExit(130)

    print("Replay complete. Final target has been sent repeatedly.")


if __name__ == "__main__":
    main()
