#!/usr/bin/env python3
"""Internal inverse kinematics and control-angle conversion for robot v0.6."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.robot_forward_kinematics import (  # noqa: E402
    DEFAULT_GEOMETRY,
    DEFAULT_MODEL_CONFIG_PATH,
    forward_kinematics,
)


CONTROL_NAMES = ("J1", "J2", "J3", "J4", "J6")
ACTIVE_JOINTS = 4
FIXED_J5_RAD = 0.0

# Preliminary physical synchronization. These values remain calibration
# candidates until the first low-speed hardware test is complete.
SERVO_ZERO_DEG = np.asarray([90.0, 60.0, 90.0, 80.0])
SERVO_DIRECTION = np.asarray([-1.0, 1.0, -1.0, -1.0])
JOINT_DEG_PER_SERVO_DEG = np.ones(ACTIVE_JOINTS)
CONTROL_LIMITS_DEG = np.asarray(
    [
        [0.0, 170.0],
        [40.0, 100.0],
        [0.0, 140.0],
        [0.0, 180.0],
        [115.0, 150.0],
    ]
)

LOCAL_TOOL_NORMAL = np.asarray([0.0, 0.0, 1.0])
UPRIGHT_AXIS_BASE = np.asarray([0.0, 0.0, -1.0])
URDF_JOINT_NAMES = (
    "J1_base_yaw",
    "J2_shoulder_pitch",
    "J3_elbow_pitch",
    "J4_wrist_pitch",
    "J5_wrist_roll",
)


class IKError(RuntimeError):
    """Raised when no acceptable bounded IK solution exists."""


@dataclass(frozen=True)
class IKSolution:
    q_rad: np.ndarray
    position_error_mm: float
    upright_error_deg: float


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _load_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    with DEFAULT_MODEL_CONFIG_PATH.open(encoding="utf-8") as handle:
        limits_by_name = json.load(handle)["joint_limits_rad"]
    limits = np.asarray(
        [limits_by_name[name] for name in URDF_JOINT_NAMES],
        dtype=float,
    )
    return limits[:, 0], limits[:, 1]


JOINT_LOWER_RAD, JOINT_UPPER_RAD = _load_joint_limits()


def controls_within_limits(control_deg: Sequence[float]) -> bool:
    controls = _vector(control_deg, 5, "control_deg")
    return bool(
        np.all(controls >= CONTROL_LIMITS_DEG[:, 0])
        and np.all(controls <= CONTROL_LIMITS_DEG[:, 1])
    )


def control_to_joint(control_deg: Sequence[float]) -> np.ndarray:
    """Convert J1,J2,J3,J4,J6 controls to internal J1..J5 radians."""

    controls = _vector(control_deg, 5, "control_deg")
    if not controls_within_limits(controls):
        raise ValueError("current control angles exceed preliminary limits")
    servo_delta_deg = controls[:ACTIVE_JOINTS] - SERVO_ZERO_DEG
    joint_deg = SERVO_DIRECTION * JOINT_DEG_PER_SERVO_DEG * servo_delta_deg
    return np.concatenate((np.radians(joint_deg), [FIXED_J5_RAD]))


def joint_to_control(
    q_rad: Sequence[float],
    gripper_deg: float,
) -> np.ndarray:
    """Convert internal J1..J5 radians to J1,J2,J3,J4,J6 controls."""

    q = _vector(q_rad, 5, "q_rad")
    if not math.isclose(float(q[4]), FIXED_J5_RAD, abs_tol=1e-10):
        raise ValueError("J5 must remain fixed at zero")
    if not math.isfinite(gripper_deg):
        raise ValueError("gripper_deg must be finite")

    joint_deg = np.degrees(q[:ACTIVE_JOINTS])
    servo_delta_deg = joint_deg / (
        SERVO_DIRECTION * JOINT_DEG_PER_SERVO_DEG
    )
    controls = np.concatenate(
        (SERVO_ZERO_DEG + servo_delta_deg, [float(gripper_deg)])
    )
    if not controls_within_limits(controls):
        raise ValueError("generated control angles exceed preliminary limits")
    return controls


def _pose_error(
    q_rad: np.ndarray,
    target_xyz_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = forward_kinematics(q_rad, DEFAULT_GEOMETRY)
    position_error = transform[:3, 3] - target_xyz_mm
    tool_axis = transform[:3, :3] @ LOCAL_TOOL_NORMAL
    tool_axis /= np.linalg.norm(tool_axis)
    return position_error, tool_axis - UPRIGHT_AXIS_BASE


def _upright_error_deg(q_rad: np.ndarray) -> float:
    transform = forward_kinematics(q_rad, DEFAULT_GEOMETRY)
    tool_axis = transform[:3, :3] @ LOCAL_TOOL_NORMAL
    tool_axis /= np.linalg.norm(tool_axis)
    cosine = float(np.clip(np.dot(tool_axis, UPRIGHT_AXIS_BASE), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def solve_ik(
    target_xyz_mm: Sequence[float],
    previous_q_rad: Sequence[float],
    *,
    position_tolerance_mm: float = 1.0,
    upright_tolerance_deg: float = 1.0,
) -> IKSolution:
    """Solve J1..J4 for one upright TCP target while keeping J5 at zero."""

    target = _vector(target_xyz_mm, 3, "target_xyz_mm")
    previous = _vector(previous_q_rad, 5, "previous_q_rad").copy()
    previous[4] = FIXED_J5_RAD
    lower = JOINT_LOWER_RAD[:ACTIVE_JOINTS]
    upper = JOINT_UPPER_RAD[:ACTIVE_JOINTS]
    previous_active = np.clip(previous[:ACTIVE_JOINTS], lower, upper)

    azimuth = previous_active.copy()
    azimuth[0] = np.clip(
        math.atan2(float(target[1]), float(target[0])),
        lower[0],
        upper[0],
    )
    starts = (
        previous_active,
        azimuth,
        np.clip(np.zeros(ACTIVE_JOINTS), lower, upper),
        (lower + upper) / 2.0,
    )

    def full_q(active_q: np.ndarray) -> np.ndarray:
        return np.concatenate((active_q, [FIXED_J5_RAD]))

    def residual(active_q: np.ndarray) -> np.ndarray:
        position_error, upright_error = _pose_error(full_q(active_q), target)
        return np.concatenate((position_error, 50.0 * upright_error))

    candidates: list[tuple[bool, float, float, float, np.ndarray]] = []
    for start in starts:
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            method="trf",
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        q = full_q(np.asarray(result.x, dtype=float))
        position_error_mm = float(np.linalg.norm(_pose_error(q, target)[0]))
        upright_error_deg = _upright_error_deg(q)
        feasible = (
            position_error_mm <= position_tolerance_mm
            and upright_error_deg <= upright_tolerance_deg
        )
        distance = float(np.linalg.norm(q[:ACTIVE_JOINTS] - previous_active))
        candidates.append(
            (feasible, distance, position_error_mm, upright_error_deg, q)
        )

    feasible_candidates = [item for item in candidates if item[0]]
    if not feasible_candidates:
        best = min(candidates, key=lambda item: (item[2], item[3]))
        raise IKError(
            "IK failed: "
            f"position error={best[2]:.3f} mm, "
            f"upright error={best[3]:.3f} deg"
        )

    best = min(feasible_candidates, key=lambda item: item[1])
    return IKSolution(
        q_rad=best[4],
        position_error_mm=best[2],
        upright_error_deg=best[3],
    )
