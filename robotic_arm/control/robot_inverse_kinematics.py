#!/usr/bin/env python3
"""Internal inverse kinematics and control-angle conversion for robot v0.6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


CONTROL_NAMES = ("J1", "J2", "J3", "J4", "J6")
ACTIVE_JOINTS = 4
FIXED_J5_RAD = 0.0

# The control implementation is deliberately self-contained. URDF/USD and the
# vision package are simulation/visualisation inputs and must never define the
# real robot controller at runtime.
BASE_TO_J1_MM = np.asarray([9.666, 0.0, 70.0])
J1_TO_J2_MM = np.asarray([9.5, 0.0, 50.0])
J2_TO_J3_MM = np.asarray([200.0, 0.0, 0.0])
J3_TO_J4_MM = np.asarray([140.0, 0.0, 0.0])
J4_TO_J5_MM = np.asarray([27.5, 0.0, 0.0])
J5_TO_TCP_MM = np.asarray([93.258628, 0.0, 0.0])

J1_ORIGIN_RPY_RAD = np.asarray([0.0, 0.0, 3.141592654])
J2_ORIGIN_RPY_RAD = np.asarray([0.0, -2.331005512, 0.0])
J3_ORIGIN_RPY_RAD = np.asarray([0.0, -0.7608, 0.0])
J4_ORIGIN_RPY_RAD = np.asarray([0.0, 0.0, 0.0])
J5_ORIGIN_RPY_RAD = np.asarray([0.0, 0.0, 0.0])

# Real-servo calibration and safe command limits. These constants are the only
# runtime source of joint limits for control and IK.
SERVO_ZERO_DEG = np.asarray([90.0, 60.0, 90.0, 90.0])
SERVO_DIRECTION = np.asarray([-1.0, 1.0, -1.0, -1.0])
JOINT_DEG_PER_SERVO_DEG = np.ones(ACTIVE_JOINTS)
CONTROL_LIMITS_DEG = np.asarray(
    [
        [0.0, 180.0],
        [0.0, 180.0],
        [0.0, 180.0],
        [0.0, 180.0],
        [100.0, 170.0],
    ]
)

# Measured J1 actuator behaviour. These are fixed controller constants, not
# command-line tuning parameters.
J1_BACKLASH_DEG = 30.0
J1_COMMAND_STEP_DEG = 5.0
J1_BACKLASH_FAST_DEG = 10.0
J1_BACKLASH_FAST_INTERVAL_S = 0.2
J1_BACKLASH_SLOW_INTERVAL_S = 0.75

# Keep both directional offsets on the 5-degree actuator grid. An odd number
# of 5-degree backlash steps (for example 25 degrees) cannot use symmetric
# +/- offsets, so the positive side receives the extra half-step.
J1_POSITIVE_OFFSET_DEG = J1_COMMAND_STEP_DEG * math.ceil(
    J1_BACKLASH_DEG / (2.0 * J1_COMMAND_STEP_DEG)
)
J1_NEGATIVE_OFFSET_DEG = J1_POSITIVE_OFFSET_DEG - J1_BACKLASH_DEG

# Reserve directional backlash headroom so compensated motor commands always
# remain within the real J1 control range.
IK_CONTROL_LIMITS_DEG = CONTROL_LIMITS_DEG.copy()
IK_CONTROL_LIMITS_DEG[0] = np.asarray(
    [
        CONTROL_LIMITS_DEG[0, 0] - J1_NEGATIVE_OFFSET_DEG,
        CONTROL_LIMITS_DEG[0, 1] - J1_POSITIVE_OFFSET_DEG,
    ]
)

LOCAL_TOOL_NORMAL = np.asarray([0.0, 0.0, 1.0])
UPRIGHT_AXIS_BASE = np.asarray([0.0, 0.0, -1.0])


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


def _joint_limits_from_servo_config() -> tuple[np.ndarray, np.ndarray]:
    """Derive mathematical J1..J4 limits from the real control limits."""

    endpoint_delta_deg = (
        IK_CONTROL_LIMITS_DEG[:ACTIVE_JOINTS]
        - SERVO_ZERO_DEG[:, np.newaxis]
    )
    endpoint_joint_deg = (
        SERVO_DIRECTION[:, np.newaxis]
        * JOINT_DEG_PER_SERVO_DEG[:, np.newaxis]
        * endpoint_delta_deg
    )
    return (
        np.radians(np.min(endpoint_joint_deg, axis=1)),
        np.radians(np.max(endpoint_joint_deg, axis=1)),
    )


JOINT_LOWER_RAD, JOINT_UPPER_RAD = _joint_limits_from_servo_config()


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


def _rotation_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy_rad
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _translation(vector_mm: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = vector_mm
    return result


def forward_kinematics(q_rad: Sequence[float]) -> np.ndarray:
    """Return the J1-centred base-to-TCP transform using control-local data."""

    q1, q2, q3, q4, q5 = _vector(q_rad, 5, "q_rad")
    transform = _rotation_rpy(J1_ORIGIN_RPY_RAD)
    transform @= _rotation_z(q1)
    transform @= _translation(J1_TO_J2_MM)
    transform @= _rotation_rpy(J2_ORIGIN_RPY_RAD)
    transform @= _rotation_y(q2)
    transform @= _translation(J2_TO_J3_MM)
    transform @= _rotation_rpy(J3_ORIGIN_RPY_RAD)
    transform @= _rotation_y(q3)
    transform @= _translation(J3_TO_J4_MM)
    transform @= _rotation_rpy(J4_ORIGIN_RPY_RAD)
    transform @= _rotation_y(q4)
    transform @= _translation(J4_TO_J5_MM)
    transform @= _rotation_rpy(J5_ORIGIN_RPY_RAD)
    transform @= _rotation_x(q5)
    transform @= _translation(J5_TO_TCP_MM)
    return transform


def _pose_error(
    q_rad: np.ndarray,
    target_xyz_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = forward_kinematics(q_rad)
    position_error = transform[:3, 3] - target_xyz_mm
    tool_axis = transform[:3, :3] @ LOCAL_TOOL_NORMAL
    tool_axis /= np.linalg.norm(tool_axis)
    return position_error, tool_axis - UPRIGHT_AXIS_BASE


def _upright_error_deg(q_rad: np.ndarray) -> float:
    transform = forward_kinematics(q_rad)
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
        target_text = ", ".join(f"{value:.3f}" for value in target)
        raise IKError(
            f"IK failed for target=[{target_text}] mm: no solution within "
            f"position<={position_tolerance_mm:.3f} mm and "
            f"upright<={upright_tolerance_deg:.3f} deg under the "
            "control-derived joint limits; best weighted candidate has "
            f"position error={best[2]:.3f} mm and "
            f"upright error={best[3]:.3f} deg"
        )

    best = min(feasible_candidates, key=lambda item: item[1])
    return IKSolution(
        q_rad=best[4],
        position_error_mm=best[2],
        upright_error_deg=best[3],
    )
