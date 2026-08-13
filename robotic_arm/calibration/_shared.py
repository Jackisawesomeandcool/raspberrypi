#!/usr/bin/env python3
"""Shared helpers for the calibration command-line tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


class CalibrationError(RuntimeError):
    """Raised when a calibration input cannot produce a trustworthy result."""


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def chessboard_object_points(
    pattern_cols: int,
    pattern_rows: int,
    square_mm: float,
) -> np.ndarray:
    """Return row-major inner-corner coordinates on the Z=0 board plane."""

    if pattern_cols < 2 or pattern_rows < 2:
        raise ValueError("checkerboard must have at least 2x2 inner corners")
    if square_mm <= 0.0:
        raise ValueError("square_mm must be positive")
    points = np.zeros((pattern_cols * pattern_rows, 3), dtype=np.float32)
    grid = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    points[:, :2] = grid.astype(np.float32) * float(square_mm)
    return points


def detect_chessboard(
    image_bgr: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
) -> tuple[bool, np.ndarray | None]:
    """Detect a complete checkerboard with subpixel-quality corners.

    The SB detector is preferred, but a printed board under glare or with a
    crease can make its global check fail even when all corners are visible.
    In that case, retry on a contrast-normalised image and finally use the
    conventional detector with explicit subpixel refinement.  All successful
    paths return the same row-major corner layout.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    expected = pattern_cols * pattern_rows

    def valid(corners: np.ndarray | None) -> np.ndarray | None:
        if corners is None or corners.reshape(-1, 2).shape != (expected, 2):
            return None
        return corners.astype(np.float32)

    contrast_normalised = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    for candidate in (gray, contrast_normalised):
        found, corners = cv2.findChessboardCornersSB(
            candidate,
            (pattern_cols, pattern_rows),
            flags=sb_flags,
        )
        result = valid(corners if found else None)
        if result is not None:
            return True, result

    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    for candidate in (gray, contrast_normalised):
        found, corners = cv2.findChessboardCorners(
            candidate,
            (pattern_cols, pattern_rows),
            flags=classic_flags,
        )
        result = valid(corners if found else None)
        if result is None:
            continue
        cv2.cornerSubPix(
            candidate,
            result,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01,
            ),
        )
        return True, result
    return False, None


def load_intrinsics(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, Any]]:
    payload = load_json(path)
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=float)
    distortion = np.asarray(payload["distortion_coefficients"], dtype=float)
    image_size = tuple(int(value) for value in payload["image_size_px"])
    if camera_matrix.shape != (3, 3):
        raise CalibrationError("camera_matrix must be 3x3")
    if len(image_size) != 2:
        raise CalibrationError("image_size_px must be [width, height]")
    return camera_matrix, distortion, image_size, payload


def require_image_size(image: np.ndarray, expected_size: Sequence[int]) -> None:
    actual = (int(image.shape[1]), int(image.shape[0]))
    expected = tuple(int(value) for value in expected_size)
    if actual != expected:
        raise CalibrationError(
            f"image resolution {actual} does not match calibration {expected}"
        )


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(image, camera_matrix, distortion)


def transform_points_homography(
    points_xy: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)
