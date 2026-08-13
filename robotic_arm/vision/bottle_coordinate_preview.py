#!/usr/bin/env python3
"""Show live soda-can coordinates using the shared YOLO-World detector."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from can_yolo_world import (
    CanDetection,
    YoloWorldCanDetector,
    draw_can_base,
    draw_can_box,
)
from pixel_to_robot import DEFAULT_CALIBRATION, pixel_to_robot


VISION_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = VISION_DIR / "captures"
YOLO_PREVIEW_INTERVAL_S = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--reference",
        type=Path,
        help="retained for CLI compatibility; YOLO detection does not use it",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    return parser


def _expected_image_size(calibration_path: Path) -> tuple[int, int]:
    with calibration_path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    return tuple(int(value) for value in calibration["image_size_px"])


def _draw_status(
    image: np.ndarray,
    coordinate: list[float] | None,
    confidence: float | None,
) -> np.ndarray:
    display = image.copy()
    if coordinate is None:
        text = "container base: not detected"
        color = (0, 0, 255)
    else:
        text = (
            f"J1=({coordinate[0]:.1f}, "
            f"{coordinate[1]:.1f}, 0.0) mm"
        )
        color = (0, 255, 0)
    if confidence is not None:
        text += f"  confidence={confidence:.3f}"
    cv2.putText(
        display,
        text,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return display


def main() -> int:
    args = _parser().parse_args()
    expected_size = _expected_image_size(args.calibration)
    detector = YoloWorldCanDetector()
    print("正在加载 YOLO-World soda-can detector...")
    detector.load()

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit(
            f"无法打开 camera index {args.camera_index}；"
            "请关闭 Photo Booth 后重试。"
        )

    window_name = "iPhone container base locator"
    print(
        "backend=yolo_world_soda_can；"
        "实时显示瓶/罐底部中心和 J1 坐标；"
        "按 s 保存当前结果；按 q 或 Esc 退出。"
    )

    detection: CanDetection | None = None
    next_inference_s = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("读取相机画面失败。")
                return 1

            actual_size = (frame.shape[1], frame.shape[0])
            if actual_size != expected_size:
                raise RuntimeError(
                    f"相机画面为 {actual_size}，但标定对应 {expected_size}"
                )

            center: tuple[int, int] | None = None
            coordinate: list[float] | None = None
            now_s = time.monotonic()
            if now_s >= next_inference_s:
                detection = detector.detect(frame)
                next_inference_s = now_s + YOLO_PREVIEW_INTERVAL_S
            annotated = draw_can_box(frame, detection)
            if detection is not None and detection.base_center is not None:
                center = detection.base_center
                coordinate = pixel_to_robot(
                    center,
                    calibration_path=args.calibration,
                )

            confidence = (
                None if detection is None else detection.confidence
            )
            display = _draw_status(annotated, coordinate, confidence)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key != ord("s"):
                continue

            detection = detector.detect(frame)
            center = None
            coordinate = None
            if detection is not None:
                annotated = draw_can_base(frame, detection)
                center = detection.base_center
                if center is not None:
                    coordinate = pixel_to_robot(
                        center,
                        calibration_path=args.calibration,
                    )
                display = _draw_status(
                    annotated,
                    coordinate,
                    detection.confidence,
                )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            args.save_dir.mkdir(parents=True, exist_ok=True)
            raw_path = args.save_dir / f"bottle_{timestamp}.png"
            annotated_path = (
                args.save_dir / f"bottle_{timestamp}_detected.png"
            )
            mask_path = args.save_dir / f"bottle_{timestamp}_mask.png"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(str(annotated_path), display)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if detection is not None and detection.red_contour is not None:
                cv2.drawContours(
                    mask,
                    [detection.red_contour],
                    -1,
                    255,
                    -1,
                )
            cv2.imwrite(str(mask_path), mask)

            if center is None or coordinate is None:
                print(f"已保存当前帧，但没有检测到瓶/罐底部：{raw_path}")
                continue

            result = {
                "backend": "yolo_world_soda_can",
                "bottle_bottom_pixel": list(center),
                "j1_coordinate_mm": coordinate,
                "image": str(raw_path),
                "confidence": detection.confidence,
                "box_xyxy": list(detection.box_xyxy),
            }
            result_path = args.save_dir / f"bottle_{timestamp}.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"标注图：{annotated_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
