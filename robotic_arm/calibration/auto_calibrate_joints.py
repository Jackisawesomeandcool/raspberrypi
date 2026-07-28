#!/usr/bin/env python3
"""Automatically move J2-J4, capture iPhone frames, and fit static mappings."""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from _shared import CalibrationError, load_intrinsics, load_json, save_json
from calibrate_joints import (
    JOINTS,
    calibrate_joint_samples,
    observe_image,
)


CONTROL_NAMES = ("J1", "J2", "J3", "J4", "J6")
CONTROL_INDEX = {name: index for index, name in enumerate(CONTROL_NAMES)}
CONTROL_LIMITS_DEG = np.asarray(
    [
        [0.0, 170.0],
        [40.0, 100.0],
        [0.0, 140.0],
        [0.0, 180.0],
        [115.0, 150.0],
    ]
)

DEFAULT_ROBOT_IP = os.environ.get("ROBOT_IP", "127.0.0.1")
DEFAULT_ROBOT_PORT = 5005
MAX_SWEEP_OFFSET_DEG = 20.0
MAX_SPEED_DEG_S = 5.0
MAX_RATE_HZ = 20.0


@dataclass(frozen=True)
class SampleTarget:
    joint: str
    split: str
    command_deg: float
    controls: np.ndarray


@dataclass(frozen=True)
class SweepPlan:
    reference: np.ndarray
    targets: tuple[SampleTarget, ...]
    settle_seconds: float
    max_speed_deg_s: float
    rate_hz: float
    camera_warmup_seconds: float
    capture_timeout_seconds: float
    source: dict[str, Any]


def _control_vector(values: Sequence[float], name: str) -> np.ndarray:
    controls = np.asarray(values, dtype=float)
    if controls.shape != (5,) or not np.all(np.isfinite(controls)):
        raise CalibrationError(f"{name} must contain five finite control angles")
    if not np.allclose(controls, np.round(controls), atol=1e-9):
        raise CalibrationError(
            f"{name} must use integer degrees because the current UDP protocol "
            "rounds every command"
        )
    if np.any(controls < CONTROL_LIMITS_DEG[:, 0]) or np.any(
        controls > CONTROL_LIMITS_DEG[:, 1]
    ):
        raise CalibrationError(f"{name} exceeds preliminary control limits")
    return controls


def _positive_number(
    payload: dict[str, Any],
    key: str,
    default: float,
    maximum: float,
) -> float:
    value = float(payload.get(key, default))
    if not math.isfinite(value) or value <= 0.0 or value > maximum:
        raise CalibrationError(f"{key} must be in (0, {maximum}]")
    return value


def load_sweep_plan(path: Path) -> SweepPlan:
    payload = load_json(path)
    reference = _control_vector(
        payload.get("reference_control_deg", []),
        "reference_control_deg",
    )
    sweeps = payload.get("sweeps")
    if not isinstance(sweeps, dict):
        raise CalibrationError("sweeps must be an object containing J2, J3 and J4")

    targets: list[SampleTarget] = []
    for joint in JOINTS:
        joint_plan = sweeps.get(joint)
        if not isinstance(joint_plan, dict):
            raise CalibrationError(f"sweeps.{joint} must be an object")
        fit_values = joint_plan.get("fit")
        validation_values = joint_plan.get("validation")
        if not isinstance(fit_values, list) or len(fit_values) < 3:
            raise CalibrationError(f"{joint} requires at least three fit commands")
        if not isinstance(validation_values, list) or len(validation_values) < 1:
            raise CalibrationError(
                f"{joint} requires at least one validation command"
            )

        all_values = [("fit", value) for value in fit_values]
        all_values.extend(("validation", value) for value in validation_values)
        commands_seen: set[float] = set()
        joint_index = CONTROL_INDEX[joint]
        for split, raw_value in all_values:
            command = float(raw_value)
            if not math.isfinite(command) or not math.isclose(
                command,
                round(command),
                abs_tol=1e-9,
            ):
                raise CalibrationError(
                    f"{joint} {split} commands must be integer degrees"
                )
            if command in commands_seen:
                raise CalibrationError(f"{joint} command {command:g} is duplicated")
            commands_seen.add(command)
            if (
                command < CONTROL_LIMITS_DEG[joint_index, 0]
                or command > CONTROL_LIMITS_DEG[joint_index, 1]
            ):
                raise CalibrationError(
                    f"{joint} command {command:g} exceeds preliminary limits"
                )
            offset = abs(command - reference[joint_index])
            if offset > MAX_SWEEP_OFFSET_DEG:
                raise CalibrationError(
                    f"{joint} command {command:g} is {offset:g} deg from the "
                    f"reference; maximum automatic sweep offset is "
                    f"{MAX_SWEEP_OFFSET_DEG:g} deg"
                )
            controls = reference.copy()
            controls[joint_index] = command
            targets.append(
                SampleTarget(
                    joint=joint,
                    split=split,
                    command_deg=command,
                    controls=controls,
                )
            )

    return SweepPlan(
        reference=reference,
        targets=tuple(targets),
        settle_seconds=_positive_number(
            payload,
            "settle_seconds",
            default=3.0,
            maximum=30.0,
        ),
        max_speed_deg_s=_positive_number(
            payload,
            "max_speed_deg_s",
            default=3.0,
            maximum=MAX_SPEED_DEG_S,
        ),
        rate_hz=_positive_number(
            payload,
            "rate_hz",
            default=10.0,
            maximum=MAX_RATE_HZ,
        ),
        camera_warmup_seconds=_positive_number(
            payload,
            "camera_warmup_seconds",
            default=2.0,
            maximum=15.0,
        ),
        capture_timeout_seconds=_positive_number(
            payload,
            "capture_timeout_seconds",
            default=10.0,
            maximum=60.0,
        ),
        source=payload,
    )


def _minimum_jerk(fraction: float) -> float:
    return 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5


def _payload(sequence: int, controls: Sequence[float]) -> bytes:
    values = ",".join(str(int(round(value))) for value in controls)
    return f"{sequence}:{values}".encode()


class RobotSender:
    """Send slow joint-space command ramps using the current UDP wire format."""

    def __init__(
        self,
        *,
        robot_ip: str,
        robot_port: int,
        reference: np.ndarray,
        rate_hz: float,
        max_speed_deg_s: float,
    ) -> None:
        if not 1 <= robot_port <= 65535:
            raise CalibrationError("robot_port must be between 1 and 65535")
        self._address = (robot_ip, robot_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = int(time.time() * 1000)
        self.current = reference.copy()
        self.rate_hz = rate_hz
        self.max_speed_deg_s = max_speed_deg_s

    def close(self) -> None:
        self._socket.close()

    def send_once(self, controls: np.ndarray) -> None:
        self._socket.sendto(
            _payload(self._sequence, controls),
            self._address,
        )
        self._sequence += 1

    def move_to(self, target: np.ndarray) -> None:
        delta = target - self.current
        largest_change = float(np.max(np.abs(delta)))
        if largest_change <= 1e-9:
            self.send_once(target)
            return

        duration = largest_change / self.max_speed_deg_s
        intervals = max(1, int(math.ceil(duration * self.rate_hz)))
        started = time.monotonic()
        start = self.current.copy()
        for index in range(1, intervals + 1):
            blend = _minimum_jerk(index / intervals)
            controls = start + blend * (target - start)
            self.send_once(controls)
            remaining = started + index / self.rate_hz - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
        self.current = target.copy()


def open_camera(
    camera_index: int,
    *,
    expected_size: tuple[int, int],
    warmup_seconds: float,
) -> tuple[cv2.VideoCapture, np.ndarray]:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise CalibrationError(
            f"cannot open iPhone camera index {camera_index} with AVFoundation"
        )

    deadline = time.monotonic() + warmup_seconds
    latest: np.ndarray | None = None
    while time.monotonic() < deadline:
        ok, frame = camera.read()
        if ok and frame is not None:
            latest = frame
    if latest is None:
        camera.release()
        raise CalibrationError("camera opened but did not return a frame")

    actual_size = (int(latest.shape[1]), int(latest.shape[0]))
    if actual_size != expected_size:
        camera.release()
        raise CalibrationError(
            f"camera resolution {actual_size} does not match intrinsics "
            f"{expected_size}"
        )
    return camera, latest


def capture_valid_observation(
    camera: cv2.VideoCapture,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    pixel_to_plane: np.ndarray,
    marker_config: dict[str, Any],
    timeout_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    # Drain frames accumulated while the arm was moving or settling.
    drain_deadline = time.monotonic() + 0.5
    while time.monotonic() < drain_deadline:
        camera.read()

    deadline = time.monotonic() + timeout_seconds
    last_error = "no camera frame"
    while time.monotonic() < deadline:
        ok, frame = camera.read()
        if not ok or frame is None:
            last_error = "camera read failed"
            continue
        try:
            observations, annotated = observe_image(
                frame,
                camera_matrix=camera_matrix,
                distortion=distortion,
                image_size=image_size,
                pixel_to_plane=pixel_to_plane,
                marker_config=marker_config,
            )
        except (CalibrationError, ValueError) as error:
            last_error = str(error)
            continue
        angles = {
            name: float(observation.angle_deg)
            for name, observation in observations.items()
        }
        return frame, annotated, angles
    raise CalibrationError(
        f"no valid marker observation within {timeout_seconds:g}s: {last_error}"
    )


def append_sample(
    csv_path: Path,
    *,
    image_path: Path,
    sweep_joint: str,
    command_deg: float,
    split: str,
) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_path", "sweep_joint", "command_deg", "split"),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image_path),
                "sweep_joint": sweep_joint,
                "command_deg": f"{command_deg:g}",
                "split": split,
            }
        )


def save_observation(
    *,
    run_dir: Path,
    index: int,
    name: str,
    frame: np.ndarray,
    annotated: np.ndarray,
) -> Path:
    image_dir = run_dir / "images"
    annotated_dir = run_dir / "observations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{index:03d}_{name}.png"
    image_path = image_dir / filename
    annotated_path = annotated_dir / filename
    if not cv2.imwrite(str(image_path), frame):
        raise CalibrationError(f"failed to save {image_path}")
    if not cv2.imwrite(str(annotated_path), annotated):
        raise CalibrationError(f"failed to save {annotated_path}")
    return image_path


def print_plan(plan: SweepPlan) -> None:
    reference = " ".join(
        f"{name}={value:g}"
        for name, value in zip(CONTROL_NAMES, plan.reference, strict=True)
    )
    print(f"reference: {reference}")
    print(
        f"motion: max_speed={plan.max_speed_deg_s:g} deg/s, "
        f"rate={plan.rate_hz:g} Hz, settle={plan.settle_seconds:g} s"
    )
    previous_joint: str | None = None
    for target in plan.targets:
        if target.joint != previous_joint:
            print(f"{target.joint}:")
            previous_joint = target.joint
        print(
            f"  {target.split:<10} command={target.command_deg:g} deg",
        )


def _confirm_motion(plan: SweepPlan, assume_yes: bool) -> None:
    if assume_yes:
        return
    print()
    print("确认机械臂当前已经处于上面显示的 reference 控制姿态。")
    print("自动流程没有舵机到达反馈；中止后不会自动回位。")
    answer = input("输入 MOVE 开始实体运动，其他输入取消：").strip()
    if answer != "MOVE":
        raise CalibrationError("motion cancelled")


def collect_and_fit(
    *,
    plan: SweepPlan,
    config_path: Path,
    intrinsics_path: Path,
    plane_path: Path,
    marker_config_path: Path,
    output_dir: Path,
    camera_index: int,
    robot_ip: str,
    robot_port: int,
    assume_yes: bool,
) -> None:
    camera_matrix, distortion, image_size, _ = load_intrinsics(intrinsics_path)
    plane_payload = load_json(plane_path)
    plane_size = tuple(int(value) for value in plane_payload["image_size_px"])
    if plane_size != image_size:
        raise CalibrationError(
            f"plane resolution {plane_size} differs from intrinsics {image_size}"
        )
    pixel_to_plane = np.asarray(
        plane_payload["pixel_to_plane_homography"],
        dtype=float,
    )
    marker_config = load_json(marker_config_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    samples_csv = output_dir / "joint_samples.csv"
    camera: cv2.VideoCapture | None = None
    sender: RobotSender | None = None
    completed_normally = False

    try:
        camera, _ = open_camera(
            camera_index,
            expected_size=image_size,
            warmup_seconds=plan.camera_warmup_seconds,
        )
        print(
            f"camera index={camera_index}, backend={camera.getBackendName()}, "
            f"size={image_size[0]}x{image_size[1]}"
        )
        _confirm_motion(plan, assume_yes)

        sender = RobotSender(
            robot_ip=robot_ip,
            robot_port=robot_port,
            reference=plan.reference,
            rate_hz=plan.rate_hz,
            max_speed_deg_s=plan.max_speed_deg_s,
        )
        sender.send_once(plan.reference)
        time.sleep(plan.settle_seconds)

        frame, annotated, angles = capture_valid_observation(
            camera,
            camera_matrix=camera_matrix,
            distortion=distortion,
            image_size=image_size,
            pixel_to_plane=pixel_to_plane,
            marker_config=marker_config,
            timeout_seconds=plan.capture_timeout_seconds,
        )
        image_path = save_observation(
            run_dir=output_dir,
            index=0,
            name="reference",
            frame=frame,
            annotated=annotated,
        )
        append_sample(
            samples_csv,
            image_path=image_path.relative_to(output_dir),
            sweep_joint="reference",
            command_deg=0.0,
            split="reference",
        )
        print(f"captured reference: {angles}")

        sample_index = 1
        active_joint: str | None = None
        for target in plan.targets:
            if active_joint is not None and target.joint != active_joint:
                print(f"returning to reference after {active_joint}")
                sender.move_to(plan.reference)
                time.sleep(plan.settle_seconds)
            active_joint = target.joint

            print(
                f"moving {target.joint} to {target.command_deg:g} deg "
                f"({target.split})"
            )
            sender.move_to(target.controls)
            time.sleep(plan.settle_seconds)
            frame, annotated, angles = capture_valid_observation(
                camera,
                camera_matrix=camera_matrix,
                distortion=distortion,
                image_size=image_size,
                pixel_to_plane=pixel_to_plane,
                marker_config=marker_config,
                timeout_seconds=plan.capture_timeout_seconds,
            )
            name = (
                f"{target.joint.lower()}_{target.split}_"
                f"{int(round(target.command_deg))}"
            )
            image_path = save_observation(
                run_dir=output_dir,
                index=sample_index,
                name=name,
                frame=frame,
                annotated=annotated,
            )
            append_sample(
                samples_csv,
                image_path=image_path.relative_to(output_dir),
                sweep_joint=target.joint,
                command_deg=target.command_deg,
                split=target.split,
            )
            print(f"captured {image_path.name}: {angles}")
            sample_index += 1

        print("all samples captured; returning to reference")
        sender.move_to(plan.reference)
        time.sleep(plan.settle_seconds)
        completed_normally = True
    finally:
        if sender is not None:
            sender.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

    if not completed_normally:
        raise CalibrationError(
            "acquisition stopped; no automatic return command was sent"
        )

    run_record = {
        "schema_version": 1,
        "config_file": str(config_path),
        "config": plan.source,
        "camera_index": camera_index,
        "image_size_px": list(image_size),
        "robot_ip": robot_ip,
        "robot_port": robot_port,
        "samples_csv": str(samples_csv),
        "note": (
            "UDP transport has no servo acknowledgement; samples use fixed "
            "settle time followed by visual marker validation."
        ),
    }
    save_json(output_dir / "run.json", run_record)

    print("fitting static command-to-joint mappings")
    result = calibrate_joint_samples(
        samples_csv=samples_csv,
        intrinsics_path=intrinsics_path,
        plane_path=plane_path,
        marker_config_path=marker_config_path,
        output_json=output_dir / "joint_calibration.json",
        measured_csv=output_dir / "joint_samples_measured.csv",
        plot_path=output_dir / "joint_calibration_plot.png",
        annotated_dir=output_dir / "fit_observations",
    )
    print(f"saved: {output_dir / 'joint_calibration.json'}")
    for joint in JOINTS:
        calibration = result["joints"][joint]["servo_calibration"]
        metrics = result["joints"][joint]["validation_metrics"]
        print(
            f"{joint}: zero={calibration['servo_zero_deg']:.3f}, "
            f"direction={calibration['direction']:+d}, "
            f"scale={calibration['joint_deg_per_servo_deg']:.5f}, "
            f"validation={metrics}"
        )
    print("review the validation results before changing production constants")


def _build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--side-plane", type=Path, required=True)
    parser.add_argument(
        "--marker-config",
        type=Path,
        default=base / "yellow_tape.example.json",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--send",
        action="store_true",
        help="execute physical motion; otherwise only print the validated plan",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the MOVE confirmation; only meaningful with --send",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        plan = load_sweep_plan(args.config)
        print_plan(plan)
        if not args.send:
            print()
            print("dry-run only; add --send after reviewing every target")
            return 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = (
                Path(__file__).resolve().parent
                / "data"
                / "joint_runs"
                / timestamp
            )
        collect_and_fit(
            plan=plan,
            config_path=args.config,
            intrinsics_path=args.intrinsics,
            plane_path=args.side_plane,
            marker_config_path=args.marker_config,
            output_dir=output_dir,
            camera_index=args.camera_index,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            assume_yes=args.yes,
        )
        return 0
    except KeyboardInterrupt:
        print()
        print("interrupted: no further command or automatic return was sent")
        return 130
    except (CalibrationError, OSError, ValueError) as error:
        print(f"automatic calibration failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
