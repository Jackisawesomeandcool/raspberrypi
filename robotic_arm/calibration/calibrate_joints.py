#!/usr/bin/env python3
"""Fit static J2-J4 command-to-joint-angle mappings from side-view images.

This is deliberately an offline acquisition interface: the CSV associates each
settled image with the logical command sent to one swept joint.  It does not
assume the final hardware communication protocol.  J1 is treated as already
fixed/calibrated and J5 must remain fixed at zero.

Each rigid link carries one elongated yellow tape marker.  PCA estimates its
unoriented axis in the calibrated side plane; differences are wrapped modulo
180 degrees around the reference pose.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from _shared import (
    CalibrationError,
    load_intrinsics,
    load_json,
    require_image_size,
    save_json,
    transform_points_homography,
    undistort_image,
)


JOINTS = ("J2", "J3", "J4")
URDF_JOINT_NAMES = {
    "J2": "J2_shoulder_pitch",
    "J3": "J3_elbow_pitch",
    "J4": "J4_wrist_pitch",
}


@dataclass(frozen=True)
class LinkObservation:
    angle_deg: float
    center_pixel: tuple[float, float]
    axis_start_pixel: tuple[float, float]
    axis_end_pixel: tuple[float, float]
    center_plane_mm: tuple[float, float]
    major_length_mm: float
    area_px: float
    aspect_ratio: float
    contour: np.ndarray


@dataclass(frozen=True)
class TapeCandidate:
    observation: LinkObservation
    center_normalized: tuple[float, float]


def wrap_axis_degrees(value: float) -> float:
    """Wrap an unoriented line-angle difference to [-90, 90)."""

    return (float(value) + 90.0) % 180.0 - 90.0


def joint_angles_from_link_angles(
    current_deg: dict[str, float],
    reference_deg: dict[str, float],
) -> dict[str, float]:
    upper = current_deg["upper_arm"]
    forearm = current_deg["forearm"]
    tool = current_deg["tool"]
    upper_zero = reference_deg["upper_arm"]
    forearm_zero = reference_deg["forearm"]
    tool_zero = reference_deg["tool"]
    return {
        "J2": wrap_axis_degrees(upper - upper_zero),
        "J3": wrap_axis_degrees(
            (forearm - upper) - (forearm_zero - upper_zero)
        ),
        "J4": wrap_axis_degrees(
            (tool - forearm) - (tool_zero - forearm_zero)
        ),
    }


def _yellow_mask(hsv: np.ndarray, marker_config: dict[str, Any]) -> np.ndarray:
    ranges = marker_config.get("hsv_ranges", [])
    if not ranges:
        raise CalibrationError("yellow tape config must define hsv_ranges")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
        upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
        if lower.shape != (3,) or upper.shape != (3,):
            raise CalibrationError("HSV lower/upper values must each have length 3")
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def _contour_centroid(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) <= 1e-12:
        raise CalibrationError("yellow tape contour has zero area")
    return np.asarray(
        (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ),
        dtype=float,
    )


def _candidate_from_contour(
    contour: np.ndarray,
    *,
    pixel_to_plane: np.ndarray,
    image_size: tuple[int, int],
) -> TapeCandidate:
    points_pixel = contour.reshape(-1, 2).astype(np.float64)
    points_plane = transform_points_homography(points_pixel, pixel_to_plane)
    center_plane = np.mean(points_plane, axis=0)
    centered = points_plane - center_plane
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ axis
    start_plane = center_plane + axis * float(np.min(projections))
    end_plane = center_plane + axis * float(np.max(projections))
    major_length = float(np.max(projections) - np.min(projections))
    if major_length <= 1e-6:
        raise CalibrationError("yellow tape major axis collapsed in plane")

    plane_to_pixel = np.linalg.inv(pixel_to_plane)
    axis_pixels = transform_points_homography(
        np.asarray((start_plane, end_plane)),
        plane_to_pixel,
    )
    center_pixel = _contour_centroid(contour)
    angle = wrap_axis_degrees(
        math.degrees(math.atan2(float(-axis[1]), float(axis[0])))
    )
    rect = cv2.minAreaRect(contour)
    width_px, height_px = rect[1]
    minor_px = min(float(width_px), float(height_px))
    major_px = max(float(width_px), float(height_px))
    aspect_ratio = major_px / max(minor_px, 1e-6)
    area = float(cv2.contourArea(contour))
    image_width, image_height = image_size
    observation = LinkObservation(
        angle_deg=angle,
        center_pixel=tuple(float(value) for value in center_pixel),
        axis_start_pixel=tuple(float(value) for value in axis_pixels[0]),
        axis_end_pixel=tuple(float(value) for value in axis_pixels[1]),
        center_plane_mm=tuple(float(value) for value in center_plane),
        major_length_mm=major_length,
        area_px=area,
        aspect_ratio=aspect_ratio,
        contour=contour,
    )
    return TapeCandidate(
        observation=observation,
        center_normalized=(
            float(center_pixel[0] / image_width),
            float(center_pixel[1] / image_height),
        ),
    )


def detect_tape_observations(
    image_undistorted: np.ndarray,
    *,
    marker_config: dict[str, Any],
    pixel_to_plane: np.ndarray,
    morphology_kernel_px: int,
) -> tuple[dict[str, LinkObservation], np.ndarray]:
    hsv = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2HSV)
    mask = _yellow_mask(hsv, marker_config)
    kernel_size = max(1, int(morphology_kernel_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_height, image_width = mask.shape
    image_area = float(image_width * image_height)
    min_area = image_area * float(marker_config.get("min_area_fraction", 0.0003))
    max_area = image_area * float(marker_config.get("max_area_fraction", 0.03))
    min_aspect = float(marker_config.get("min_aspect_ratio", 1.5))
    min_major_px = float(marker_config.get("min_major_length_px", 30.0))

    candidates: list[TapeCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        width_px, height_px = cv2.minAreaRect(contour)[1]
        major_px = max(float(width_px), float(height_px))
        minor_px = min(float(width_px), float(height_px))
        aspect = major_px / max(minor_px, 1e-6)
        if major_px < min_major_px or aspect < min_aspect:
            continue
        candidates.append(
            _candidate_from_contour(
                contour,
                pixel_to_plane=pixel_to_plane,
                image_size=(image_width, image_height),
            )
        )

    link_names = ("upper_arm", "forearm", "tool")
    if len(candidates) < len(link_names):
        raise CalibrationError(
            f"found only {len(candidates)} elongated yellow tape candidates; "
            "three are required"
        )
    links = marker_config.get("links")
    if not isinstance(links, dict):
        raise CalibrationError("yellow tape config must define links")
    seeds = []
    for link_name in link_names:
        values = np.asarray(
            links.get(link_name, {}).get("reference_center_normalized", []),
            dtype=float,
        )
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise CalibrationError(
                f"{link_name} must define reference_center_normalized [x, y]"
            )
        seeds.append(values)

    best_assignment: tuple[int, ...] | None = None
    best_distances: np.ndarray | None = None
    best_score = float("inf")
    candidate_centers = np.asarray(
        [candidate.center_normalized for candidate in candidates],
        dtype=float,
    )
    for assignment in itertools.permutations(
        range(len(candidates)),
        len(link_names),
    ):
        distances = np.linalg.norm(
            candidate_centers[np.asarray(assignment)] - np.asarray(seeds),
            axis=1,
        )
        score = float(np.sum(distances))
        if score < best_score:
            best_score = score
            best_assignment = assignment
            best_distances = distances
    if best_assignment is None or best_distances is None:
        raise CalibrationError("failed to assign yellow tapes to robot links")

    max_distance = float(
        marker_config.get("max_center_distance_normalized", 0.18)
    )
    observations: dict[str, LinkObservation] = {}
    for link_name, candidate_index, distance in zip(
        link_names,
        best_assignment,
        best_distances,
        strict=True,
    ):
        if float(distance) > max_distance:
            raise CalibrationError(
                f"{link_name} yellow tape is too far from its reference area: "
                f"normalized distance={distance:.3f}, limit={max_distance:.3f}"
            )
        observations[link_name] = candidates[candidate_index].observation
    return observations, mask


def observe_image(
    image_bgr: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    pixel_to_plane: np.ndarray,
    marker_config: dict[str, Any],
) -> tuple[dict[str, LinkObservation], np.ndarray]:
    require_image_size(image_bgr, image_size)
    undistorted = undistort_image(image_bgr, camera_matrix, distortion)
    kernel = int(marker_config.get("morphology_kernel_px", 5))
    observations, _ = detect_tape_observations(
        undistorted,
        marker_config=marker_config,
        pixel_to_plane=pixel_to_plane,
        morphology_kernel_px=kernel,
    )

    annotated = undistorted.copy()
    colours = {
        "upper_arm": (0, 255, 255),
        "forearm": (0, 255, 0),
        "tool": (255, 0, 0),
    }
    for link_name, observation in observations.items():
        center = tuple(int(round(value)) for value in observation.center_pixel)
        start = tuple(int(round(value)) for value in observation.axis_start_pixel)
        end = tuple(int(round(value)) for value in observation.axis_end_pixel)
        colour = colours[link_name]
        cv2.drawContours(
            annotated,
            [observation.contour],
            -1,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            annotated,
            start,
            end,
            colour,
            3,
            cv2.LINE_AA,
        )
        cv2.circle(annotated, center, 5, colour, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            (
                f"{link_name}: {observation.angle_deg:.2f} deg, "
                f"L={observation.major_length_mm:.1f} mm"
            ),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    return observations, annotated


def _load_sample_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "sweep_joint", "command_deg", "split"}
    if not rows:
        raise CalibrationError("sample CSV is empty")
    if not required.issubset(rows[0]):
        raise CalibrationError(
            f"sample CSV must contain columns: {sorted(required)}"
        )
    reference_rows = [
        row
        for row in rows
        if row["sweep_joint"].strip().lower() == "reference"
        or row["split"].strip().lower() == "reference"
    ]
    if len(reference_rows) != 1:
        raise CalibrationError(
            f"exactly one reference row is required, found {len(reference_rows)}"
        )
    return rows


def _fit_mapping(
    command_deg: np.ndarray,
    actual_deg: np.ndarray,
) -> tuple[float, float]:
    if command_deg.size < 3:
        raise CalibrationError("at least three fit samples are required per joint")
    design = np.column_stack((command_deg, np.ones_like(command_deg)))
    slope, intercept = np.linalg.lstsq(design, actual_deg, rcond=None)[0]
    if abs(float(slope)) <= 1e-8:
        raise CalibrationError("fitted command-to-joint slope is nearly zero")
    return float(slope), float(intercept)


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float] | None:
    if actual.size == 0:
        return None
    errors = actual - predicted
    return {
        "count": int(actual.size),
        "rmse_deg": float(np.sqrt(np.mean(errors * errors))),
        "mean_error_deg": float(np.mean(errors)),
        "max_abs_error_deg": float(np.max(np.abs(errors))),
    }


def _save_plot(
    output_path: Path,
    measured_rows: list[dict[str, Any]],
    fit_results: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, joint in zip(axes, JOINTS, strict=True):
        joint_rows = [row for row in measured_rows if row["sweep_joint"] == joint]
        for split, marker, colour in (
            ("fit", "o", "#2767a8"),
            ("validation", "s", "#d47b28"),
        ):
            rows = [row for row in joint_rows if row["split"] == split]
            if rows:
                axis.scatter(
                    [row["command_deg"] for row in rows],
                    [row[f"{joint}_actual_deg"] for row in rows],
                    marker=marker,
                    color=colour,
                    label=split,
                )
        if joint_rows:
            x_values = np.asarray([row["command_deg"] for row in joint_rows])
            x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
            slope = fit_results[joint]["slope_joint_deg_per_command_deg"]
            intercept = fit_results[joint]["intercept_joint_deg"]
            axis.plot(x_line, slope * x_line + intercept, color="#333333")
        axis.set_title(joint)
        axis.set_xlabel("logical command / deg")
        axis.set_ylabel("observed joint angle / deg")
        axis.grid(alpha=0.25)
        axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def calibrate_joint_samples(
    *,
    samples_csv: Path,
    intrinsics_path: Path,
    plane_path: Path,
    marker_config_path: Path,
    output_json: Path,
    measured_csv: Path,
    plot_path: Path,
    annotated_dir: Path | None,
) -> dict[str, Any]:
    rows = _load_sample_rows(samples_csv)
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
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    observations_by_row: list[dict[str, Any]] = []
    base_dir = samples_csv.resolve().parent
    for row_index, row in enumerate(rows):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise CalibrationError(f"cannot read sample image: {image_path}")
        observations, annotated = observe_image(
            image,
            camera_matrix=camera_matrix,
            distortion=distortion,
            image_size=image_size,
            pixel_to_plane=pixel_to_plane,
            marker_config=marker_config,
        )
        if annotated_dir is not None:
            annotated_name = f"{row_index:03d}_{image_path.name}"
            cv2.imwrite(str(annotated_dir / annotated_name), annotated)
        observations_by_row.append(
            {
                "image_path": str(image_path),
                "sweep_joint": row["sweep_joint"].strip(),
                "command_deg": float(row["command_deg"] or 0.0),
                "split": row["split"].strip().lower(),
                "link_angles_deg": {
                    name: observation.angle_deg
                    for name, observation in observations.items()
                },
            }
        )

    reference_row = next(
        row
        for row in observations_by_row
        if row["sweep_joint"].lower() == "reference"
        or row["split"] == "reference"
    )
    reference_angles = reference_row["link_angles_deg"]
    measured_rows: list[dict[str, Any]] = []
    for row in observations_by_row:
        actual = joint_angles_from_link_angles(
            row["link_angles_deg"],
            reference_angles,
        )
        measured_rows.append({**row, **{f"{j}_actual_deg": actual[j] for j in JOINTS}})

    results: dict[str, dict[str, Any]] = {}
    for joint in JOINTS:
        joint_rows = [
            row for row in measured_rows if row["sweep_joint"].upper() == joint
        ]
        fit_rows = [row for row in joint_rows if row["split"] == "fit"]
        validation_rows = [
            row for row in joint_rows if row["split"] == "validation"
        ]
        commands = np.asarray([row["command_deg"] for row in fit_rows])
        actual = np.asarray([row[f"{joint}_actual_deg"] for row in fit_rows])
        slope, intercept = _fit_mapping(commands, actual)
        fit_predicted = slope * commands + intercept
        validation_commands = np.asarray(
            [row["command_deg"] for row in validation_rows]
        )
        validation_actual = np.asarray(
            [row[f"{joint}_actual_deg"] for row in validation_rows]
        )
        validation_predicted = slope * validation_commands + intercept
        for row in joint_rows:
            row["predicted_actual_deg"] = slope * row["command_deg"] + intercept
            row["residual_deg"] = (
                row[f"{joint}_actual_deg"] - row["predicted_actual_deg"]
            )
        results[joint] = {
            "urdf_joint_name": URDF_JOINT_NAMES[joint],
            "slope_joint_deg_per_command_deg": slope,
            "intercept_joint_deg": intercept,
            "servo_calibration": {
                "servo_zero_deg": -intercept / slope,
                "joint_zero_rad": 0.0,
                "direction": 1 if slope > 0.0 else -1,
                "joint_deg_per_servo_deg": abs(slope),
            },
            "fit_metrics": _error_metrics(actual, fit_predicted),
            "validation_metrics": _error_metrics(
                validation_actual,
                validation_predicted,
            ),
        }

    measured_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_path",
        "sweep_joint",
        "command_deg",
        "split",
        "upper_arm_angle_deg",
        "forearm_angle_deg",
        "tool_angle_deg",
        "J2_actual_deg",
        "J3_actual_deg",
        "J4_actual_deg",
        "predicted_actual_deg",
        "residual_deg",
    ]
    with measured_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in measured_rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "sweep_joint": row["sweep_joint"],
                    "command_deg": row["command_deg"],
                    "split": row["split"],
                    "upper_arm_angle_deg": row["link_angles_deg"]["upper_arm"],
                    "forearm_angle_deg": row["link_angles_deg"]["forearm"],
                    "tool_angle_deg": row["link_angles_deg"]["tool"],
                    "J2_actual_deg": row["J2_actual_deg"],
                    "J3_actual_deg": row["J3_actual_deg"],
                    "J4_actual_deg": row["J4_actual_deg"],
                    "predicted_actual_deg": row.get("predicted_actual_deg", ""),
                    "residual_deg": row.get("residual_deg", ""),
                }
            )

    output = {
        "schema_version": 1,
        "calibration_type": "static_logical_command_to_robot_joint_angle",
        "scope": {
            "calibrated_joints": list(JOINTS),
            "J1": "excluded; project owner reports manual calibration",
            "J5": "fixed at q=0 during acquisition and intended operation",
            "J2": (
                "one logical shoulder command; paired left/right servo transport "
                "remains a controller-adapter responsibility"
            ),
        },
        "inputs": {
            "samples_csv": str(samples_csv),
            "intrinsics": str(intrinsics_path),
            "side_plane": str(plane_path),
            "marker_config": str(marker_config_path),
        },
        "reference_link_angles_deg": reference_angles,
        "joints": results,
        "measured_samples_csv": str(measured_csv),
        "plot": str(plot_path),
    }
    save_json(output_json, output)
    _save_plot(plot_path, measured_rows, results)
    return output


def _build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--side-plane", type=Path, required=True)
    parser.add_argument(
        "--marker-config",
        type=Path,
        default=base / "yellow_tape.example.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "output" / "joint_calibration.json",
    )
    parser.add_argument(
        "--measured-csv",
        type=Path,
        default=base / "output" / "joint_samples_measured.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=base / "output" / "joint_calibration_plot.png",
    )
    parser.add_argument("--annotated-dir", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        output = calibrate_joint_samples(
            samples_csv=args.samples,
            intrinsics_path=args.intrinsics,
            plane_path=args.side_plane,
            marker_config_path=args.marker_config,
            output_json=args.output,
            measured_csv=args.measured_csv,
            plot_path=args.plot,
            annotated_dir=args.annotated_dir,
        )
    except CalibrationError as error:
        print(f"calibration failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"saved: {args.output}")
    for joint in JOINTS:
        result = output["joints"][joint]
        validation = result["validation_metrics"]
        validation_text = (
            "no validation samples"
            if validation is None
            else (
                f"validation RMSE={validation['rmse_deg']:.3f} deg, "
                f"max={validation['max_abs_error_deg']:.3f} deg"
            )
        )
        calibration = result["servo_calibration"]
        print(
            f"{joint}: zero={calibration['servo_zero_deg']:.4f} deg, "
            f"direction={calibration['direction']:+d}, "
            f"scale={calibration['joint_deg_per_servo_deg']:.6f}, "
            f"{validation_text}"
        )


if __name__ == "__main__":
    main()
