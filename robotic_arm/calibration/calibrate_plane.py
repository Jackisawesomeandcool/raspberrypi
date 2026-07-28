#!/usr/bin/env python3
"""Calibrate the fixed side-view image to the J2-J4 motion plane.

Place the printed checkerboard in the same physical plane as the link markers.
The output homography maps *undistorted image pixels* to board-plane
millimetres.  The board column direction is plane +X; image/board row direction
is plane +Y, so the joint-angle script uses -Y as physical upward Z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


from _shared import (
    CalibrationError,
    chessboard_object_points,
    detect_chessboard,
    load_intrinsics,
    require_image_size,
    save_json,
    transform_points_homography,
    undistort_image,
)


def calibrate_plane(
    image_bgr: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    pattern_cols: int,
    pattern_rows: int,
    square_mm: float,
) -> tuple[dict[str, object], np.ndarray]:
    require_image_size(image_bgr, image_size)
    undistorted = undistort_image(image_bgr, camera_matrix, distortion)
    found, corners = detect_chessboard(
        undistorted,
        pattern_cols,
        pattern_rows,
    )
    if not found or corners is None:
        raise CalibrationError("complete checkerboard was not detected")
    plane_points = chessboard_object_points(
        pattern_cols,
        pattern_rows,
        square_mm,
    )[:, :2]
    image_points = corners.reshape(-1, 2)
    homography, inlier_mask = cv2.findHomography(
        image_points,
        plane_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=max(0.25, 0.04 * square_mm),
    )
    if homography is None:
        raise CalibrationError("homography solve failed")
    homography = homography / homography[2, 2]
    projected = transform_points_homography(image_points, homography)
    residuals = np.linalg.norm(projected - plane_points, axis=1)
    inverse = np.linalg.inv(homography)
    inverse /= inverse[2, 2]
    mask = (
        np.ones(len(residuals), dtype=bool)
        if inlier_mask is None
        else inlier_mask.reshape(-1).astype(bool)
    )

    annotated = undistorted.copy()
    cv2.drawChessboardCorners(
        annotated,
        (pattern_cols, pattern_rows),
        corners,
        True,
    )
    origin_px = tuple(int(round(value)) for value in image_points[0])
    cv2.putText(
        annotated,
        "plane origin (0,0)",
        origin_px,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "calibration_type": "undistorted_pixel_to_side_plane",
        "image_size_px": list(image_size),
        "checkerboard": {
            "inner_corners": [pattern_cols, pattern_rows],
            "square_mm_measured": float(square_mm),
        },
        "pixel_to_plane_homography": homography.tolist(),
        "plane_to_pixel_homography": inverse.tolist(),
        "plane_axes": {
            "x": "checkerboard columns increasing to the right",
            "y": "checkerboard rows increasing downward",
            "physical_z_up": "-plane_y",
        },
        "inlier_count": int(np.count_nonzero(mask)),
        "corner_count": int(len(mask)),
        "rmse_mm": float(np.sqrt(np.mean(residuals[mask] ** 2))),
        "max_error_mm": float(np.max(residuals[mask])),
    }
    return result, annotated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        required=True,
    )
    parser.add_argument("--pattern-cols", type=int, default=7)
    parser.add_argument("--pattern-rows", type=int, default=5)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "side_plane.json",
    )
    parser.add_argument(
        "--annotated-image",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "side_plane_annotated.png",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    camera_matrix, distortion, image_size, intrinsics_payload = load_intrinsics(
        args.intrinsics
    )
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    result, annotated = calibrate_plane(
        image,
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        square_mm=args.square_mm,
    )
    result["intrinsics_file"] = str(args.intrinsics)
    result["intrinsics_rms_reprojection_error_px"] = intrinsics_payload[
        "rms_reprojection_error_px"
    ]
    result["source"] = str(args.image)
    save_json(args.output, result)
    args.annotated_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.annotated_image), annotated):
        raise CalibrationError(f"failed to save {args.annotated_image}")
    print(f"saved: {args.output}")
    print(f"annotated: {args.annotated_image}")
    print(f"plane RMSE: {result['rmse_mm']:.4f} mm")
    print(f"plane max error: {result['max_error_mm']:.4f} mm")


if __name__ == "__main__":
    main()
