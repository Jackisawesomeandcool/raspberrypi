#!/usr/bin/env python3
"""Plan, preview and optionally execute one pick-and-place task."""

from __future__ import annotations

import argparse
import math
import socket
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from robot_inverse_kinematics import (
    CONTROL_NAMES,
    IKError,
    control_to_joint,
    controls_within_limits,
    joint_to_control,
    solve_ik,
)


DEFAULT_ROBOT_IP = "100.107.100.56"
DEFAULT_ROBOT_PORT = 5005
DEFAULT_RATE_HZ = 10.0
DEFAULT_LIFT_MM = 60.0
DEFAULT_GRIPPER_OPEN_DEG = 150.0
DEFAULT_GRIPPER_CLOSED_DEG = 115.0

JOINT_SPEED_DEG_S = 5.0
VERTICAL_SPEED_MM_S = 15.0
TRANSFER_SPEED_MM_S = 20.0
GRIPPER_DURATION_S = 5.0
GRIPPER_HOLD_S = 2.0
MAX_CONTROL_STEP_DEG = 5.0


@dataclass(frozen=True)
class Frame:
    time_s: float
    phase: str
    q_rad: tuple[float, float, float, float, float]
    control_deg: tuple[float, float, float, float, float]


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _minimum_jerk(fraction: float) -> float:
    return (
        10.0 * fraction**3
        - 15.0 * fraction**4
        + 6.0 * fraction**5
    )


def _interval_count(duration_s: float, rate_hz: float) -> int:
    return max(1, int(math.ceil(duration_s * rate_hz)))


def _frame(
    time_s: float,
    phase: str,
    q_rad: Sequence[float],
    gripper_deg: float,
) -> Frame:
    q = _vector(q_rad, 5, "q_rad")
    controls = joint_to_control(q, gripper_deg)
    return Frame(
        time_s=float(time_s),
        phase=phase,
        q_rad=tuple(float(value) for value in q),
        control_deg=tuple(float(value) for value in controls),
    )


def _append_joint_segment(
    frames: list[Frame],
    phase: str,
    end_q_rad: Sequence[float],
    gripper_deg: float,
    rate_hz: float,
    speed_scale: float,
) -> None:
    start_q = np.asarray(frames[-1].q_rad)
    end_q = _vector(end_q_rad, 5, "end_q_rad")
    largest_change_deg = float(
        np.max(np.abs(np.degrees(end_q[:4] - start_q[:4])))
    )
    duration_s = max(2.0, largest_change_deg / JOINT_SPEED_DEG_S) / speed_scale
    intervals = _interval_count(duration_s, rate_hz)
    start_time = frames[-1].time_s

    for index in range(1, intervals + 1):
        fraction = index / intervals
        blend = _minimum_jerk(fraction)
        q = start_q + blend * (end_q - start_q)
        q[4] = 0.0
        frames.append(
            _frame(
                start_time + duration_s * fraction,
                phase,
                q,
                gripper_deg,
            )
        )


def _append_cartesian_segment(
    frames: list[Frame],
    phase: str,
    start_xyz_mm: Sequence[float],
    end_xyz_mm: Sequence[float],
    gripper_deg: float,
    speed_mm_s: float,
    rate_hz: float,
    speed_scale: float,
) -> None:
    start_xyz = _vector(start_xyz_mm, 3, "start_xyz_mm")
    end_xyz = _vector(end_xyz_mm, 3, "end_xyz_mm")
    distance_mm = float(np.linalg.norm(end_xyz - start_xyz))
    duration_s = max(2.0, distance_mm / speed_mm_s) / speed_scale
    intervals = _interval_count(duration_s, rate_hz)
    start_time = frames[-1].time_s
    q_seed = np.asarray(frames[-1].q_rad)

    for index in range(1, intervals + 1):
        fraction = index / intervals
        target = start_xyz + _minimum_jerk(fraction) * (end_xyz - start_xyz)
        solution = solve_ik(target, q_seed)
        q_seed = solution.q_rad
        frames.append(
            _frame(
                start_time + duration_s * fraction,
                phase,
                q_seed,
                gripper_deg,
            )
        )


def _append_gripper_segment(
    frames: list[Frame],
    phase: str,
    end_gripper_deg: float,
    rate_hz: float,
    speed_scale: float,
) -> None:
    q = np.asarray(frames[-1].q_rad)
    start_gripper = frames[-1].control_deg[4]
    duration_s = GRIPPER_DURATION_S / speed_scale
    intervals = _interval_count(duration_s, rate_hz)
    start_time = frames[-1].time_s

    for index in range(1, intervals + 1):
        fraction = index / intervals
        gripper = start_gripper + _minimum_jerk(fraction) * (
            end_gripper_deg - start_gripper
        )
        frames.append(
            _frame(
                start_time + duration_s * fraction,
                phase,
                q,
                gripper,
            )
        )


def _append_hold(
    frames: list[Frame],
    phase: str,
    duration_s: float,
    rate_hz: float,
) -> None:
    previous = frames[-1]
    q = np.asarray(previous.q_rad)
    gripper = previous.control_deg[4]
    intervals = _interval_count(duration_s, rate_hz)
    start_time = previous.time_s

    for index in range(1, intervals + 1):
        frames.append(
            _frame(
                start_time + duration_s * index / intervals,
                phase,
                q,
                gripper,
            )
        )


def _validate_plan(
    frames: Sequence[Frame],
    current_control_deg: np.ndarray,
) -> None:
    if not frames:
        raise ValueError("empty motion plan")
    if not np.allclose(frames[0].control_deg, current_control_deg, atol=1e-9):
        raise ValueError("first frame does not match current control angles")

    previous_control = np.asarray(frames[0].control_deg)
    for index, frame in enumerate(frames):
        q = np.asarray(frame.q_rad)
        controls = np.asarray(frame.control_deg)
        if not math.isclose(float(q[4]), 0.0, abs_tol=1e-10):
            raise ValueError(f"frame {index}: J5 is not zero")
        if controls.shape != (5,) or not controls_within_limits(controls):
            raise ValueError(f"frame {index}: invalid control angles")
        if index:
            step = float(np.max(np.abs(controls - previous_control)))
            if step > MAX_CONTROL_STEP_DEG:
                raise ValueError(
                    f"frame {index}: control step {step:.2f} deg exceeds "
                    f"{MAX_CONTROL_STEP_DEG:.2f} deg"
                )
        previous_control = controls


def plan_pick_and_place(
    pick_xyz_mm: Sequence[float],
    place_xyz_mm: Sequence[float],
    current_control_deg: Sequence[float],
    *,
    lift_mm: float = DEFAULT_LIFT_MM,
    rate_hz: float = DEFAULT_RATE_HZ,
    speed_scale: float = 1.0,
    gripper_open_deg: float = DEFAULT_GRIPPER_OPEN_DEG,
    gripper_closed_deg: float = DEFAULT_GRIPPER_CLOSED_DEG,
) -> list[Frame]:
    """Precompute the complete task; no command is sent from this function."""

    pick = _vector(pick_xyz_mm, 3, "pick_xyz_mm")
    place = _vector(place_xyz_mm, 3, "place_xyz_mm")
    current_control = _vector(current_control_deg, 5, "current_control_deg")
    if not controls_within_limits(current_control):
        raise ValueError("current control angles exceed preliminary limits")
    if not math.isfinite(lift_mm) or lift_mm <= 0.0:
        raise ValueError("lift_mm must be positive")
    if not math.isfinite(rate_hz) or not 2.0 <= rate_hz <= 50.0:
        raise ValueError("rate_hz must be between 2 and 50")
    if not math.isfinite(speed_scale) or not 0.1 <= speed_scale <= 2.0:
        raise ValueError("speed_scale must be between 0.1 and 2.0")
    frame_rate_hz = min(50.0, rate_hz * max(1.0, speed_scale))

    safe_z = max(float(pick[2]), float(place[2])) + lift_mm
    pick_above = np.asarray([pick[0], pick[1], safe_z])
    place_above = np.asarray([place[0], place[1], safe_z])

    current_q = control_to_joint(current_control)
    frames = [
        Frame(
            time_s=0.0,
            phase="START",
            q_rad=tuple(float(value) for value in current_q),
            control_deg=tuple(float(value) for value in current_control),
        )
    ]

    _append_gripper_segment(
        frames,
        "OPEN",
        gripper_open_deg,
        frame_rate_hz,
        speed_scale,
    )

    above_pick_q = solve_ik(pick_above, current_q).q_rad
    _append_joint_segment(
        frames,
        "MOVE_PICK_ABOVE",
        above_pick_q,
        gripper_open_deg,
        frame_rate_hz,
        speed_scale,
    )
    _append_cartesian_segment(
        frames,
        "DESCEND_PICK",
        pick_above,
        pick,
        gripper_open_deg,
        VERTICAL_SPEED_MM_S,
        frame_rate_hz,
        speed_scale,
    )
    _append_gripper_segment(
        frames,
        "CLOSE",
        gripper_closed_deg,
        frame_rate_hz,
        speed_scale,
    )
    _append_hold(frames, "HOLD_GRASP", GRIPPER_HOLD_S, frame_rate_hz)
    _append_cartesian_segment(
        frames,
        "LIFT",
        pick,
        pick_above,
        gripper_closed_deg,
        VERTICAL_SPEED_MM_S,
        frame_rate_hz,
        speed_scale,
    )
    _append_cartesian_segment(
        frames,
        "TRANSFER",
        pick_above,
        place_above,
        gripper_closed_deg,
        TRANSFER_SPEED_MM_S,
        frame_rate_hz,
        speed_scale,
    )
    _append_cartesian_segment(
        frames,
        "LOWER_PLACE",
        place_above,
        place,
        gripper_closed_deg,
        VERTICAL_SPEED_MM_S,
        frame_rate_hz,
        speed_scale,
    )
    _append_gripper_segment(
        frames,
        "OPEN",
        gripper_open_deg,
        frame_rate_hz,
        speed_scale,
    )
    _append_hold(frames, "HOLD_RELEASE", GRIPPER_HOLD_S, frame_rate_hz)
    _append_cartesian_segment(
        frames,
        "RETREAT",
        place,
        place_above,
        gripper_open_deg,
        VERTICAL_SPEED_MM_S,
        frame_rate_hz,
        speed_scale,
    )

    _validate_plan(frames, current_control)
    return frames


def _payload(sequence: int, control_deg: Sequence[float]) -> bytes:
    values = ",".join(str(int(round(value))) for value in control_deg)
    return f"{sequence}:{values}".encode()


def run_frames(
    frames: Sequence[Frame],
    *,
    send: bool,
    robot_ip: str,
    robot_port: int,
) -> None:
    """Print only control output and optionally send the same five values."""

    if not 1 <= robot_port <= 65535:
        raise ValueError("robot_port must be between 1 and 65535")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if send else None
    sequence = int(time.time() * 1000)
    start_time = time.monotonic()
    try:
        for index, frame in enumerate(frames):
            remaining = start_time + frame.time_s - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            rounded = tuple(int(round(value)) for value in frame.control_deg)
            values = " ".join(f"{value:3d}" for value in rounded)
            print(
                f"t={frame.time_s:6.2f}s "
                f"phase={frame.phase:<16} "
                f"control_deg={values}",
                flush=True,
            )
            if sock is not None:
                sock.sendto(
                    _payload(sequence + index, frame.control_deg),
                    (robot_ip, robot_port),
                )
    finally:
        if sock is not None:
            sock.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick", nargs=3, type=float, required=True)
    parser.add_argument("--place", nargs=3, type=float, required=True)
    parser.add_argument(
        "--current-control",
        nargs=5,
        type=float,
        required=True,
        metavar=CONTROL_NAMES,
    )
    parser.add_argument("--lift-mm", type=float, default=DEFAULT_LIFT_MM)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="0.1-2.0; 0.5 is half speed, 1.0 is baseline",
    )
    parser.add_argument(
        "--gripper-open-deg",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_DEG,
    )
    parser.add_argument(
        "--gripper-closed-deg",
        type=float,
        default=DEFAULT_GRIPPER_CLOSED_DEG,
    )
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        frames = plan_pick_and_place(
            args.pick,
            args.place,
            args.current_control,
            lift_mm=args.lift_mm,
            rate_hz=args.rate_hz,
            speed_scale=args.speed_scale,
            gripper_open_deg=args.gripper_open_deg,
            gripper_closed_deg=args.gripper_closed_deg,
        )
        run_frames(
            frames,
            send=args.send,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
        )
    except (IKError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
