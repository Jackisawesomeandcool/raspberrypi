#!/usr/bin/env python3
"""Convert an image pixel to a J1-centred workspace coordinate."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "calibration"
    / "output"
    / "workspace_plane.json"
)


@lru_cache(maxsize=4)
def _load_transform(
    calibration_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workspace_path = Path(calibration_path)
    with workspace_path.open(encoding="utf-8") as handle:
        workspace = json.load(handle)

    intrinsics_path = Path(workspace["intrinsics_file"])
    if not intrinsics_path.is_absolute():
        intrinsics_path = workspace_path.parent / intrinsics_path

    with intrinsics_path.open(encoding="utf-8") as handle:
        intrinsics = json.load(handle)

    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        intrinsics["distortion_coefficients"],
        dtype=np.float64,
    )
    pixel_to_base = np.asarray(
        workspace["pixel_to_base_xy_homography"],
        dtype=np.float64,
    )
    return camera_matrix, distortion, pixel_to_base


def pixel_to_robot(
    pixel_xy: tuple[float, float] | list[float] | np.ndarray,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
) -> list[float]:
    """Return ``[X, Y, 0]`` in millimetres for one raw-image pixel.

    ``pixel_xy`` must come from the same fixed camera pose and resolution used
    for workspace calibration. For an upright object, pass its ground-contact
    centre rather than the centre of its YOLO bounding box.
    """

    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("pixel_xy must contain exactly two finite values: [u, v]")

    path = str(Path(calibration_path).expanduser().resolve())
    camera_matrix, distortion, pixel_to_base = _load_transform(path)

    undistorted = cv2.undistortPoints(
        pixel.reshape(1, 1, 2),
        camera_matrix,
        distortion,
        P=camera_matrix,
    )
    base_xy = cv2.perspectiveTransform(
        undistorted,
        pixel_to_base,
    ).reshape(2)

    return [float(base_xy[0]), float(base_xy[1]), 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one raw image pixel to J1-centred [X, Y, 0] mm."
    )
    parser.add_argument("u", type=float, help="image x coordinate in pixels")
    parser.add_argument("v", type=float, help="image y coordinate in pixels")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="workspace_plane.json path",
    )
    args = parser.parse_args()

    coordinate = pixel_to_robot(
        [args.u, args.v],
        calibration_path=args.calibration,
    )
    print(json.dumps(coordinate))


if __name__ == "__main__":
    main()
