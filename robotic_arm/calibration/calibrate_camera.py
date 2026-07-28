#!/usr/bin/env python3
"""Calibrate camera intrinsics from saved checkerboard images.

The default board is the project A4 sheet: 8x6 squares, therefore 7x5 inner
corners.  ``square_mm`` must be the measured printed square size, not blindly
the nominal 25 mm.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


from _shared import (
    CalibrationError,
    chessboard_object_points,
    detect_chessboard,
    save_json,
)


def robust_median_threshold(values: np.ndarray, scale: float = 3.5) -> float:
    """Return a median/MAD upper threshold for per-view reprojection errors."""

    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return median + max(1e-9, 0.05 * max(1.0, abs(median)))
    return median + scale * 1.4826 * mad


def _calibrate_subset(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]:
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    errors = []
    for object_view, image_view, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        delta = projected.reshape(-1, 2) - image_view.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return (
        float(rms),
        camera_matrix,
        distortion,
        list(rvecs),
        list(tvecs),
        np.asarray(errors),
    )


def calibrate_images(
    image_paths: Sequence[Path],
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_mm: float,
    annotated_dir: Path | None = None,
) -> dict[str, object]:
    object_template = chessboard_object_points(
        pattern_cols,
        pattern_rows,
        square_mm,
    )
    valid_paths: list[Path] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip unreadable image: {path}")
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(f"skip different resolution {current_size}: {path}")
            continue
        found, corners = detect_chessboard(image, pattern_cols, pattern_rows)
        if not found or corners is None:
            print(f"skip no complete checkerboard: {path}")
            continue
        valid_paths.append(path)
        object_points.append(object_template.copy())
        image_points.append(corners)
        if annotated_dir is not None:
            annotated = image.copy()
            cv2.drawChessboardCorners(
                annotated,
                (pattern_cols, pattern_rows),
                corners,
                True,
            )
            cv2.imwrite(str(annotated_dir / path.name), annotated)

    if image_size is None or len(valid_paths) < 8:
        raise CalibrationError(
            f"found {len(valid_paths)} valid views; at least 8 are required"
        )
    initial = _calibrate_subset(object_points, image_points, image_size)
    threshold = robust_median_threshold(initial[-1])
    keep = initial[-1] <= threshold
    if np.count_nonzero(keep) >= 8 and not np.all(keep):
        final_indices = np.flatnonzero(keep)
        final = _calibrate_subset(
            [object_points[index] for index in final_indices],
            [image_points[index] for index in final_indices],
            image_size,
        )
    else:
        final_indices = np.arange(len(valid_paths))
        final = initial

    rms, camera_matrix, distortion, rvecs, tvecs, errors = final
    used_paths = [valid_paths[index] for index in final_indices]
    rejected_paths = [
        valid_paths[index]
        for index in range(len(valid_paths))
        if index not in set(final_indices.tolist())
    ]
    return {
        "schema_version": 1,
        "calibration_type": "opencv_pinhole_intrinsics",
        "image_size_px": list(image_size),
        "checkerboard": {
            "inner_corners": [pattern_cols, pattern_rows],
            "square_mm_measured": float(square_mm),
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rms_reprojection_error_px": rms,
        "per_view_rmse_px": [
            {"image": str(path), "rmse_px": float(error)}
            for path, error in zip(used_paths, errors, strict=True)
        ],
        "used_images": [str(path) for path in used_paths],
        "rejected_images": [str(path) for path in rejected_paths],
        "rvecs": [np.asarray(value).reshape(-1).tolist() for value in rvecs],
        "tvecs_mm": [np.asarray(value).reshape(-1).tolist() for value in tvecs],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        required=True,
        help="glob for calibration images, for example data/intrinsics/*.png",
    )
    parser.add_argument("--pattern-cols", type=int, default=7)
    parser.add_argument("--pattern-rows", type=int, default=5)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "camera_intrinsics.json",
    )
    parser.add_argument("--annotated-dir", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = [Path(value) for value in sorted(glob.glob(args.images))]
    if not paths:
        raise SystemExit(f"no images matched: {args.images}")
    result = calibrate_images(
        paths,
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        square_mm=args.square_mm,
        annotated_dir=args.annotated_dir,
    )
    save_json(args.output, result)
    print(f"saved: {args.output}")
    print(f"used views: {len(result['used_images'])}")
    print(f"RMS reprojection error: {result['rms_reprojection_error_px']:.4f} px")


if __name__ == "__main__":
    main()
