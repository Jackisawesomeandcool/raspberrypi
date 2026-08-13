#!/usr/bin/env python3
"""Scan upright TCP reach in the J1-centred control coordinate system."""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

from robot_inverse_kinematics import (
    IKError,
    control_to_joint,
    joint_to_control,
    solve_ik,
)

def _reachable_radii(
    base_z_mm: float,
    radii_mm: np.ndarray,
    initial_q_rad: np.ndarray,
) -> list[float]:
    reachable: list[float] = []
    q_seed = initial_q_rad
    for radius in radii_mm:
        try:
            solution = solve_ik([radius, 0.0, base_z_mm], q_seed)
            joint_to_control(solution.q_rad, 125.0)
        except (IKError, ValueError):
            continue
        reachable.append(float(radius))
        q_seed = solution.q_rad
    return reachable


def _intervals(values: Sequence[float], step_mm: float) -> list[tuple[float, float]]:
    if not values:
        return []
    result: list[tuple[float, float]] = []
    start = previous = float(values[0])
    for value in values[1:]:
        value = float(value)
        if value - previous > step_mm * 1.5:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _format_intervals(intervals: Sequence[tuple[float, float]]) -> str:
    if not intervals:
        return "unreachable"
    return ", ".join(f"{start:.0f}-{end:.0f} mm" for start, end in intervals)


def _suggest_straight_transfer(
    common_radii: Sequence[float],
    lifted_radii: Sequence[float],
    base_z_mm: float,
    step_mm: float,
) -> str | None:
    if not common_radii or not lifted_radii:
        return None

    outer_radius = max(common_radii) - step_mm
    inner_clearance = min(lifted_radii) + step_mm
    if outer_radius <= inner_clearance:
        return None

    lateral = math.sqrt(outer_radius**2 - inner_clearance**2)
    distance = 2.0 * lateral
    return (
        f"pick=[{inner_clearance:.0f}, {-lateral:.0f}, {base_z_mm:.0f}] "
        f"place=[{inner_clearance:.0f}, {lateral:.0f}, {base_z_mm:.0f}] "
        f"straight_distance={distance:.0f} mm"
    )


def scan_workspace(
    *,
    grasp_z_mm: Sequence[float],
    lift_mm: float,
    radius_min_mm: float,
    radius_max_mm: float,
    radius_step_mm: float,
    current_control_deg: Sequence[float],
) -> None:
    if lift_mm <= 0.0:
        raise ValueError("lift_mm must be positive")
    if radius_step_mm <= 0.0 or radius_max_mm <= radius_min_mm:
        raise ValueError("invalid radius scan range")
    grasp_z_values = np.asarray(grasp_z_mm, dtype=float)
    if (
        grasp_z_values.ndim != 1
        or grasp_z_values.size == 0
        or not np.all(np.isfinite(grasp_z_values))
    ):
        raise ValueError("grasp_z_mm must contain finite values")

    radii = np.arange(
        radius_min_mm,
        radius_max_mm + radius_step_mm * 0.5,
        radius_step_mm,
    )
    initial_q = control_to_joint(current_control_deg)

    print("coordinate_frame=J1 axis centre; z values are used directly")

    for grasp_base_z in grasp_z_values:
        grasp_base_z = float(grasp_base_z)
        lifted_base_z = grasp_base_z + lift_mm
        grasp_radii = _reachable_radii(grasp_base_z, radii, initial_q)
        lifted_radii = _reachable_radii(lifted_base_z, radii, initial_q)
        common = sorted(set(grasp_radii).intersection(lifted_radii))

        print(
            f"grasp_z={grasp_base_z:.0f} mm"
        )
        print(
            "  grasp_radius="
            f"{_format_intervals(_intervals(grasp_radii, radius_step_mm))}"
        )
        print(
            f"  lifted_base_z={lifted_base_z:.0f} mm "
            "lifted_radius="
            f"{_format_intervals(_intervals(lifted_radii, radius_step_mm))}"
        )
        print(
            "  common_radius="
            f"{_format_intervals(_intervals(common, radius_step_mm))}"
        )
        suggestion = _suggest_straight_transfer(
            common,
            lifted_radii,
            grasp_base_z,
            radius_step_mm,
        )
        if suggestion is not None:
            print(f"  conservative_candidate: {suggestion}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-z-mm",
        type=float,
        nargs="+",
        default=(0.0, 20.0, 60.0),
        help="candidate TCP Z coordinates relative to the J1 axis centre",
    )
    parser.add_argument("--lift-mm", type=float, default=80.0)
    parser.add_argument("--radius-min-mm", type=float, default=100.0)
    parser.add_argument("--radius-max-mm", type=float, default=500.0)
    parser.add_argument("--radius-step-mm", type=float, default=10.0)
    parser.add_argument(
        "--current-control",
        type=float,
        nargs=5,
        default=(90.0, 60.0, 90.0, 80.0, 125.0),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        scan_workspace(
            grasp_z_mm=args.grasp_z_mm,
            lift_mm=args.lift_mm,
            radius_min_mm=args.radius_min_mm,
            radius_max_mm=args.radius_max_mm,
            radius_step_mm=args.radius_step_mm,
            current_control_deg=args.current_control,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
