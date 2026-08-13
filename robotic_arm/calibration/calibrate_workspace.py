#!/usr/bin/env python3
"""Calibrate raw image pixels to the J1-centred workspace XY plane.

Full inner corners provide the global board pose.  The small red origin mark
selects the correct board orientation before black-square centres optionally
refine the homography with RANSAC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _shared import (
    CalibrationError,
    detect_chessboard,
    load_intrinsics,
    require_image_size,
    save_json,
    transform_points_homography,
    undistort_image,
)


BOARD_COLS = 16
BOARD_ROWS = 12
INNER_COLS = BOARD_COLS - 1
INNER_ROWS = BOARD_ROWS - 1
SQUARE_MM = 23.5
BOARD_ORIGIN_BASE_MM = np.asarray([236.668, 0.0, -70.0], dtype=float)
BOARD_ORIGIN_GRID = np.asarray([7.0, 1.0], dtype=np.float32)

# Preliminary direction convention for the installed board:
#   board row, away from J1 in the image -> base +X
#   board column, image right          -> base +Y
# Flip either vector after the first physical direction check if necessary.
BOARD_COLUMN_AXIS_BASE_XY = np.asarray([0.0, 1.0], dtype=float)
BOARD_ROW_AXIS_BASE_XY = np.asarray([1.0, 0.0], dtype=float)

# Four rough black-cell centre correspondences bootstrap automatic matching.
# Pixel coordinates are in the undistorted 1920x1080 formal calibration view.
ROUGH_SEED_IMAGE_PX = np.asarray(
    [
        [642.44, 250.69],
        [1416.67, 270.87],
        [590.27, 919.72],
        [1417.73, 922.60],
    ],
    dtype=np.float32,
)
ROUGH_SEED_BOARD_CELLS = np.asarray(
    [
        [13.5, 11.5],
        [1.5, 11.5],
        [14.5, 0.5],
        [0.5, 0.5],
    ],
    dtype=np.float32,
)

BLACK_THRESHOLD = 75
MIN_BLACK_AREA_PX = 600.0
MAX_BLACK_AREA_PX = 25000.0
MAX_BLACK_ASPECT_RATIO = 2.5
MAX_CELL_ASSIGNMENT_ERROR = 0.48
MIN_MATCHED_CELLS = 40
RED_MIN_AREA_PX = 3
RED_MAX_AREA_PX = 1500
RED_MAX_DIMENSION_PX = 120
MAX_ORIGIN_ANCHOR_ERROR_CELLS = 0.75


def _validate_axes() -> np.ndarray:
    axes = np.column_stack(
        (BOARD_COLUMN_AXIS_BASE_XY, BOARD_ROW_AXIS_BASE_XY)
    )
    if axes.shape != (2, 2) or not np.all(np.isfinite(axes)):
        raise CalibrationError("board axes must form a finite 2x2 matrix")
    if abs(float(np.linalg.det(axes))) < 1e-9:
        raise CalibrationError("board axes must not be parallel")
    return axes


def _detect_black_cell_centres(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < BLACK_THRESHOLD, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    centres: list[list[float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not MIN_BLACK_AREA_PX <= area <= MAX_BLACK_AREA_PX:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if y < 200 or width < 20 or height < 15:
            continue
        rect_width, rect_height = cv2.minAreaRect(contour)[1]
        if rect_width <= 0.0 or rect_height <= 0.0:
            continue
        aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
        if aspect > MAX_BLACK_ASPECT_RATIO:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-9:
            continue
        centres.append(
            [
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            ]
        )

    if len(centres) < MIN_MATCHED_CELLS:
        raise CalibrationError(
            f"only {len(centres)} black-cell candidates were detected"
        )
    return np.asarray(centres, dtype=np.float32)


def _snap_black_cells(
    image_points: np.ndarray,
    image_to_board: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimated = transform_points_homography(image_points, image_to_board)
    snapped = np.round(estimated - 0.5) + 0.5
    errors = np.linalg.norm(estimated - snapped, axis=1)

    cols = np.rint(snapped[:, 0] - 0.5).astype(int)
    rows = np.rint(snapped[:, 1] - 0.5).astype(int)
    valid = (
        (cols >= 0)
        & (cols < BOARD_COLS)
        & (rows >= 0)
        & (rows < BOARD_ROWS)
        & ((cols + rows) % 2 == 0)
        & (errors <= MAX_CELL_ASSIGNMENT_ERROR)
    )

    selected_indices: list[int] = []
    for cell in set(map(tuple, snapped[valid])):
        same_cell = valid & np.all(snapped == cell, axis=1)
        indices = np.flatnonzero(same_cell)
        selected_indices.append(int(indices[np.argmin(errors[indices])]))

    if len(selected_indices) < MIN_MATCHED_CELLS:
        raise CalibrationError(
            f"only {len(selected_indices)} black cells matched the board grid"
        )
    selected = np.asarray(selected_indices, dtype=int)
    return image_points[selected], snapped[selected].astype(np.float32)


def _fit_board_grid(
    cell_centres_px: np.ndarray,
    initial_image_to_board: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if initial_image_to_board is None:
        image_to_board = cv2.getPerspectiveTransform(
            ROUGH_SEED_IMAGE_PX,
            ROUGH_SEED_BOARD_CELLS,
        )
    else:
        image_to_board = np.asarray(
            initial_image_to_board,
            dtype=np.float64,
        )
        if image_to_board.shape != (3, 3):
            raise CalibrationError(
                "initial image-to-board homography must be 3x3"
            )
        if not np.all(np.isfinite(image_to_board)):
            raise CalibrationError(
                "initial image-to-board homography must be finite"
            )
        if abs(float(image_to_board[2, 2])) < 1e-12:
            raise CalibrationError(
                "initial image-to-board homography is singular"
            )
        image_to_board = image_to_board / image_to_board[2, 2]
    matched_pixels = np.empty((0, 2), dtype=np.float32)
    matched_cells = np.empty((0, 2), dtype=np.float32)
    for _ in range(3):
        matched_pixels, matched_cells = _snap_black_cells(
            cell_centres_px,
            image_to_board,
        )
        fitted, inliers = cv2.findHomography(
            matched_pixels,
            matched_cells,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.35,
        )
        if fitted is None or inliers is None:
            raise CalibrationError("image-to-board homography solve failed")
        image_to_board = fitted / fitted[2, 2]
    return image_to_board, matched_pixels, matched_cells


def _red_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 55, 35), (15, 255, 255)),
        cv2.inRange(hsv, (165, 55, 35), (179, 255, 255)),
    )


def _red_marker_candidates(
    image_bgr: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        _red_mask(image_bgr)
    )
    candidates: list[tuple[np.ndarray, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if not RED_MIN_AREA_PX <= area <= RED_MAX_AREA_PX:
            continue
        if width > RED_MAX_DIMENSION_PX or height > RED_MAX_DIMENSION_PX:
            continue
        candidates.append((
            np.asarray(centroids[index], dtype=np.float32),
            area,
        ))
    return candidates


def _inner_corner_board_points(
    *,
    flip_columns: bool,
    flip_rows: bool,
) -> np.ndarray:
    columns = np.arange(1, INNER_COLS + 1, dtype=np.float32)
    rows = np.arange(1, INNER_ROWS + 1, dtype=np.float32)
    if flip_columns:
        columns = columns[::-1]
    if flip_rows:
        rows = rows[::-1]
    col_grid, row_grid = np.meshgrid(columns, rows)
    return np.column_stack((col_grid.ravel(), row_grid.ravel()))


def _inner_corner_solution(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Return a globally oriented board fit from full inner corners.

    The black-square pattern repeats every cell, so the red pin is used here
    as an anchor before refinement to choose the correct mirror and grid offset.
    """

    found, corners = detect_chessboard(
        image_bgr,
        INNER_COLS,
        INNER_ROWS,
    )
    red_candidates = _red_marker_candidates(image_bgr)
    if not found or corners is None or not red_candidates:
        return None

    image_points = corners.reshape(-1, 2).astype(np.float32)
    best: tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
    ] | None = None
    for flip_columns in (False, True):
        for flip_rows in (False, True):
            board_points = _inner_corner_board_points(
                flip_columns=flip_columns,
                flip_rows=flip_rows,
            )
            image_to_board, _ = cv2.findHomography(
                image_points,
                board_points,
                method=0,
            )
            if image_to_board is None:
                continue
            image_to_board /= image_to_board[2, 2]
            for red_pixel, area in red_candidates:
                predicted = transform_points_homography(
                    red_pixel.reshape(1, 2),
                    image_to_board,
                )[0]
                error = float(np.linalg.norm(predicted - BOARD_ORIGIN_GRID))
                candidate = (
                    error,
                    image_to_board,
                    board_points,
                    red_pixel,
                    area,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

    if best is None or best[0] > MAX_ORIGIN_ANCHOR_ERROR_CELLS:
        return None
    _, image_to_board, board_points, red_pixel, area = best
    return image_to_board, image_points, board_points, red_pixel, area


def _find_red_origin(
    image_bgr: np.ndarray,
    image_to_board: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    candidates: list[tuple[float, np.ndarray, int]] = []
    for pixel, area in _red_marker_candidates(image_bgr):
        board = transform_points_homography(
            pixel.reshape(1, 2),
            image_to_board,
        )[0]
        error = float(np.linalg.norm(board - BOARD_ORIGIN_GRID))
        if error <= MAX_ORIGIN_ANCHOR_ERROR_CELLS:
            candidates.append((error, pixel, area))

    if not candidates:
        raise CalibrationError("red board-origin mark was not identified")
    _, pixel, area = min(
        candidates,
        key=lambda item: item[0],
    )
    return BOARD_ORIGIN_GRID.copy(), pixel, area


def _board_to_base_xy(
    board_points: np.ndarray,
    origin_board: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    delta_mm = (board_points - origin_board) * SQUARE_MM
    return BOARD_ORIGIN_BASE_MM[:2] + delta_mm @ axes.T


def pixel_to_base_xy(
    raw_pixel_xy: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    pixel_to_base_homography: np.ndarray,
) -> np.ndarray:
    """Convert raw image pixels to robot-base XY millimetres."""

    points = np.asarray(raw_pixel_xy, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        points,
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(-1, 2)
    return transform_points_homography(
        undistorted,
        pixel_to_base_homography,
    )


def calibrate_workspace(
    image_bgr: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    initial_image_to_board: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    axes = _validate_axes()
    require_image_size(image_bgr, image_size)
    undistorted = undistort_image(image_bgr, camera_matrix, distortion)

    corner_solution = _inner_corner_solution(undistorted)
    if corner_solution is not None:
        (
            corner_seed,
            corner_pixels,
            corner_board_points,
            origin_pixel,
            marker_area,
        ) = corner_solution
        try:
            candidates = _detect_black_cell_centres(undistorted)
            image_to_board, image_points, board_points = _fit_board_grid(
                candidates,
                initial_image_to_board=corner_seed,
            )
            origin_board, origin_pixel, marker_area = _find_red_origin(
                undistorted,
                image_to_board,
            )
            matching_method = "inner_corners_seeded_black_cells"
        except (CalibrationError, cv2.error):
            image_to_board = corner_seed
            image_points = corner_pixels
            board_points = corner_board_points
            origin_board = BOARD_ORIGIN_GRID.copy()
            matching_method = "inner_corners"
    else:
        candidates = _detect_black_cell_centres(undistorted)
        image_to_board, image_points, board_points = _fit_board_grid(
            candidates,
            initial_image_to_board=initial_image_to_board,
        )
        origin_board, origin_pixel, marker_area = _find_red_origin(
            undistorted,
            image_to_board,
        )
        matching_method = "black_cells_fallback"
    base_points = _board_to_base_xy(board_points, origin_board, axes)

    pixel_to_base, _ = cv2.findHomography(
        image_points,
        base_points,
        method=0,
    )
    if pixel_to_base is None:
        raise CalibrationError("pixel-to-base homography solve failed")
    pixel_to_base /= pixel_to_base[2, 2]
    base_to_pixel = np.linalg.inv(pixel_to_base)
    base_to_pixel /= base_to_pixel[2, 2]

    predicted = transform_points_homography(image_points, pixel_to_base)
    residuals = np.linalg.norm(predicted - base_points, axis=1)

    annotated = undistorted.copy()
    for point in image_points:
        cv2.circle(
            annotated,
            tuple(int(round(value)) for value in point),
            5,
            (0, 255, 0),
            2,
        )
    origin_xy = tuple(int(round(value)) for value in origin_pixel)
    cv2.circle(annotated, origin_xy, 13, (255, 0, 255), 3)
    cv2.putText(
        annotated,
        "workspace origin",
        (origin_xy[0] + 15, origin_xy[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    board_to_pixel = np.linalg.inv(image_to_board)
    board_to_pixel /= board_to_pixel[2, 2]
    axis_pixels = transform_points_homography(
        np.asarray(
            [
                origin_board,
                origin_board + [1.0, 0.0],
                origin_board + [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        board_to_pixel,
    )
    for endpoint, label, color in (
        (axis_pixels[1], "board column", (0, 0, 255)),
        (axis_pixels[2], "board row", (0, 255, 0)),
    ):
        endpoint_xy = tuple(int(round(value)) for value in endpoint)
        cv2.arrowedLine(
            annotated,
            origin_xy,
            endpoint_xy,
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.putText(
            annotated,
            label,
            endpoint_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "calibration_type": "undistorted_pixel_to_j1_workspace_xy",
        "image_size_px": list(image_size),
        "checkerboard": {
            "squares": [BOARD_COLS, BOARD_ROWS],
            "square_mm_measured": SQUARE_MM,
            "matched_black_cells": int(len(image_points)),
            "matching_method": matching_method,
            "marked_origin_board_grid": origin_board.tolist(),
            "marked_origin_pixel_undistorted": origin_pixel.tolist(),
            "red_marker_area_px": marker_area,
        },
        "board_origin_base_mm": BOARD_ORIGIN_BASE_MM.tolist(),
        "board_axes_in_base_xy": {
            "column": BOARD_COLUMN_AXIS_BASE_XY.tolist(),
            "row": BOARD_ROW_AXIS_BASE_XY.tolist(),
        },
        "pixel_to_board_grid_homography": image_to_board.tolist(),
        "pixel_to_base_xy_homography": pixel_to_base.tolist(),
        "base_xy_to_pixel_homography": base_to_pixel.tolist(),
        "rmse_mm": float(np.sqrt(np.mean(residuals**2))),
        "max_error_mm": float(np.max(residuals)),
    }
    return result, annotated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "camera_intrinsics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "workspace_plane.json",
    )
    parser.add_argument(
        "--annotated-image",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "workspace_plane_annotated.png",
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

    result, annotated = calibrate_workspace(
        image,
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
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
    print(
        f"matched black cells: {result['checkerboard']['matched_black_cells']}"
    )
    print(
        "marked origin grid: "
        f"{result['checkerboard']['marked_origin_board_grid']}"
    )
    print(f"workspace RMSE: {result['rmse_mm']:.3f} mm")
    print(f"workspace max error: {result['max_error_mm']:.3f} mm")


if __name__ == "__main__":
    main()
