#!/usr/bin/env python3
"""Forward kinematics for the current robotic-arm geometry.

The kinematic base frame is centred on the J1 axis:
  +Z: J1 positive axis (up)
  +X: radial arm direction when all mathematical joint angles are zero
  +Y: completes a right-handed frame

The module deliberately separates:
  1. servo command angle -> robot joint angle calibration; and
  2. robot joint angle -> TCP pose forward kinematics.

All public angle inputs use radians unless their name ends in ``_deg``.
All geometry is stored in millimetres, so the returned translation is in mm.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5")


@dataclass(frozen=True)
class RobotGeometry:
    """Fixed joint-to-joint translations expressed in the preceding frame."""

    j1_to_j2_mm: tuple[float, float, float] = (9.5, 0.0, 50.0)
    j2_to_j3_mm: tuple[float, float, float] = (200.0, 0.0, 0.0)
    j3_to_j4_mm: tuple[float, float, float] = (140.0, 0.0, 0.0)
    j4_to_j5_mm: tuple[float, float, float] = (30.0, 0.0, -6.0)
    j5_to_tcp_mm: tuple[float, float, float] = (87.533, 0.0, 8.0)


DEFAULT_GEOMETRY = RobotGeometry()


@dataclass(frozen=True)
class ServoCalibration:
    """Affine mapping between one servo command and one robot joint angle.

    ``servo_zero_deg`` is the servo command that places the robot joint at
    ``joint_zero_rad``. ``direction`` must be +1 or -1.
    ``joint_deg_per_servo_deg`` is a positive scale measured at the joint,
    not an assumed servo specification.
    """

    servo_zero_deg: float
    joint_zero_rad: float = 0.0
    direction: int = 1
    joint_deg_per_servo_deg: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if not math.isfinite(self.joint_deg_per_servo_deg):
            raise ValueError("joint_deg_per_servo_deg must be finite")
        if self.joint_deg_per_servo_deg <= 0.0:
            raise ValueError("joint_deg_per_servo_deg must be positive")
        if not math.isfinite(self.servo_zero_deg):
            raise ValueError("servo_zero_deg must be finite")
        if not math.isfinite(self.joint_zero_rad):
            raise ValueError("joint_zero_rad must be finite")


def _validate_five(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (5,):
        raise ValueError(f"{label} must contain exactly five values ordered as {JOINT_NAMES}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    result = np.eye(4)
    result[:3, :3] = (
        (1.0, 0.0, 0.0),
        (0.0, cosine, -sine),
        (0.0, sine, cosine),
    )
    return result


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    result = np.eye(4)
    result[:3, :3] = (
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    )
    return result


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    result = np.eye(4)
    result[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    return result


def _translation(vector_mm: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector_mm, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("translation must contain three finite values")
    result = np.eye(4)
    result[:3, 3] = vector
    return result


def forward_kinematics(
    joint_angles_rad: Sequence[float],
    geometry: RobotGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Return the 4x4 base-to-TCP transform for J1..J5.

    The translation part is in millimetres. The rotation part is dimensionless.
    J6 is intentionally absent because the current TCP is fixed at the nominal
    closed-gripper midpoint in ``gripper_base_link``.
    """

    q1, q2, q3, q4, q5 = _validate_five(joint_angles_rad, "joint_angles_rad")

    transform = _rotation_z(q1)
    transform = transform @ _translation(geometry.j1_to_j2_mm)
    transform = transform @ _rotation_y(q2)
    transform = transform @ _translation(geometry.j2_to_j3_mm)
    transform = transform @ _rotation_y(q3)
    transform = transform @ _translation(geometry.j3_to_j4_mm)
    transform = transform @ _rotation_y(q4)
    transform = transform @ _translation(geometry.j4_to_j5_mm)
    transform = transform @ _rotation_x(q5)
    transform = transform @ _translation(geometry.j5_to_tcp_mm)
    return transform


def tcp_position_mm(
    joint_angles_rad: Sequence[float],
    geometry: RobotGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Return only the TCP XYZ position in the J1-centred base frame."""

    return forward_kinematics(joint_angles_rad, geometry)[:3, 3].copy()


def servo_command_to_joint_angle(
    servo_command_deg: float,
    calibration: ServoCalibration,
) -> float:
    """Convert one commanded servo angle in degrees to a joint angle in radians."""

    servo_delta_deg = float(servo_command_deg) - calibration.servo_zero_deg
    joint_delta_deg = (
        calibration.direction
        * calibration.joint_deg_per_servo_deg
        * servo_delta_deg
    )
    return calibration.joint_zero_rad + math.radians(joint_delta_deg)


def joint_angle_to_servo_command(
    joint_angle_rad: float,
    calibration: ServoCalibration,
) -> float:
    """Convert one robot joint angle in radians to a commanded servo angle."""

    joint_delta_deg = math.degrees(float(joint_angle_rad) - calibration.joint_zero_rad)
    servo_delta_deg = joint_delta_deg / (
        calibration.direction * calibration.joint_deg_per_servo_deg
    )
    return calibration.servo_zero_deg + servo_delta_deg


def servo_commands_to_joint_angles(
    servo_commands_deg: Mapping[str, float],
    calibrations: Mapping[str, ServoCalibration],
) -> np.ndarray:
    """Convert J1..J5 servo commands to an ordered joint-angle vector."""

    missing_commands = [name for name in JOINT_NAMES if name not in servo_commands_deg]
    missing_calibrations = [name for name in JOINT_NAMES if name not in calibrations]
    if missing_commands:
        raise KeyError(f"missing servo commands: {missing_commands}")
    if missing_calibrations:
        raise KeyError(f"missing servo calibrations: {missing_calibrations}")

    return np.asarray(
        [
            servo_command_to_joint_angle(
                servo_commands_deg[name],
                calibrations[name],
            )
            for name in JOINT_NAMES
        ],
        dtype=float,
    )


def _run_self_tests() -> None:
    zero_pose = forward_kinematics(np.zeros(5))
    np.testing.assert_allclose(
        zero_pose[:3, 3],
        np.array([467.033, 0.0, 52.0]),
        atol=1e-12,
    )

    yaw_90_pose = forward_kinematics(np.radians([90.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        yaw_90_pose[:3, 3],
        np.array([0.0, 467.033, 52.0]),
        atol=1e-12,
    )

    calibration = ServoCalibration(
        servo_zero_deg=90.0,
        joint_zero_rad=0.0,
        direction=-1,
        joint_deg_per_servo_deg=0.5,
    )
    joint_angle = servo_command_to_joint_angle(110.0, calibration)
    np.testing.assert_allclose(math.degrees(joint_angle), -10.0, atol=1e-12)
    np.testing.assert_allclose(
        joint_angle_to_servo_command(joint_angle, calibration),
        110.0,
        atol=1e-12,
    )

    rotation = forward_kinematics(np.radians([30.0, -20.0, 45.0, -10.0, 60.0]))[
        :3, :3
    ]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-deg",
        nargs=5,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        default=(0.0, 0.0, 0.0, 0.0, 0.0),
        help="robot joint angles in degrees, ordered J1 J2 J3 J4 J5",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in geometry and calibration conversion checks first",
    )
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        print("Self-tests passed.")

    joint_angles_rad = np.radians(_validate_five(args.joint_deg, "--joint-deg"))
    transform = forward_kinematics(joint_angles_rad)

    np.set_printoptions(precision=6, suppress=True)
    print(f"Joint order: {JOINT_NAMES}")
    print(f"Joint angles (deg): {np.asarray(args.joint_deg, dtype=float)}")
    print(f"TCP position in J1-centred base frame (mm): {transform[:3, 3]}")
    print("Base-to-TCP transform (translation in mm):")
    print(transform)


if __name__ == "__main__":
    main()
