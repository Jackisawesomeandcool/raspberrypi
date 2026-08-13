#!/usr/bin/env python3
"""Detect the red soda can and estimate its base inside the YOLO box."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np


VISION_DIR = Path(__file__).resolve().parent
MODEL_DIR = VISION_DIR / "models"
DEFAULT_MODEL = MODEL_DIR / "yolov8s-world.pt"
LOCAL_CLIP_WEIGHTS_ROOT = MODEL_DIR / "weights"
CAN_PROMPT = "soda can"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 960
DEFAULT_DEVICE = "cpu"
MIN_RED_CONTOUR_AREA_PX = 1000.0
BASE_BAND_SHORT_AXIS_FRACTION = 0.30


class CanDetectionError(RuntimeError):
    """Raised when the can or its base cannot be located."""


def _load_yolo_model(model_path: str | Path):
    """Load local weights or let Ultralytics fetch the official checkpoint."""

    path = Path(model_path)
    if path.suffix.lower() != ".pt":
        raise ValueError("YOLO model must be a .pt weights file")

    import torch
    from ultralytics import YOLO

    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load
    try:
        source = str(path) if path.is_file() else path.name
        return YOLO(source)
    finally:
        torch.load = original_torch_load


@dataclass(frozen=True)
class CanDetection:
    label: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    base_center: tuple[int, int] | None
    base_axes: tuple[int, int] | None
    base_angle_deg: float | None
    red_contour: np.ndarray | None


class YoloWorldCanDetector:
    """Lazy single-model detector for the current red soda-can task."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self._model = None
        self._target = CAN_PROMPT
        self._model_lock = RLock()

    @property
    def target(self) -> str:
        with self._model_lock:
            return self._target

    @property
    def supports_grasp_geometry(self) -> bool:
        return self.target == CAN_PROMPT

    def load(self) -> None:
        with self._model_lock:
            if self._model is not None:
                return
            model = _load_yolo_model(self.model_path)
            local_clip = LOCAL_CLIP_WEIGHTS_ROOT / "clip" / "ViT-B-32.pt"
            if local_clip.is_file():
                import ultralytics.nn.text_model as text_model

                text_model.WEIGHTS_DIR = LOCAL_CLIP_WEIGHTS_ROOT
            model.set_classes([self._target])
            self._model = model

    def set_target(self, target: str) -> str:
        if not isinstance(target, str):
            raise ValueError("detection target must be a string")
        if "\n" in target or "\r" in target:
            raise ValueError("detection target must be one line")
        normalized = " ".join(target.strip().split())
        if not normalized:
            raise ValueError("detection target cannot be empty")
        if len(normalized) > 80:
            raise ValueError("detection target must be at most 80 characters")
        if not normalized.isascii():
            raise ValueError("detection target must be an English visual category")
        if not any(character.isalpha() for character in normalized):
            raise ValueError("detection target must contain English letters")

        with self._model_lock:
            self.load()
            if normalized == self._target:
                return self._target
            self._model.set_classes([normalized])
            self._target = normalized
            return self._target

    def detect(self, frame: np.ndarray) -> CanDetection | None:
        with self._model_lock:
            self.load()
            target = self._target
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_index = int(np.argmax(boxes.conf.detach().cpu().numpy()))
        confidence = float(boxes.conf[best_index].item())
        coordinates = boxes.xyxy[best_index].detach().cpu().numpy()
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(round(float(coordinates[0])))))
        y1 = max(0, min(height - 1, int(round(float(coordinates[1])))))
        x2 = max(x1 + 1, min(width, int(round(float(coordinates[2])))))
        y2 = max(y1 + 1, min(height, int(round(float(coordinates[3])))))
        box = (x1, y1, x2, y2)

        center = None
        axes = None
        angle_deg = None
        contour = None
        if target == CAN_PROMPT:
            try:
                center, axes, angle_deg, contour = _estimate_red_can_base(
                    frame,
                    box,
                )
            except CanDetectionError:
                pass
        return CanDetection(
            label=target,
            confidence=confidence,
            box_xyxy=box,
            base_center=center,
            base_axes=axes,
            base_angle_deg=angle_deg,
            red_contour=contour,
        )


def _estimate_red_can_base(
    frame: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], float, np.ndarray]:
    x1, y1, x2, y2 = box_xyxy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise CanDetectionError("YOLO can box is empty")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 70, 30), (15, 255, 255))
    red_high = cv2.inRange(hsv, (165, 70, 30), (179, 255, 255))
    mask = cv2.bitwise_or(red_low, red_high)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_RED_CONTOUR_AREA_PX
    ]
    if not candidates:
        raise CanDetectionError("no red can contour was found inside YOLO box")

    contour = max(candidates, key=cv2.contourArea).copy()
    contour[:, 0, 0] += x1
    contour[:, 0, 1] += y1
    rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
    edge_vectors = np.roll(rectangle, -1, axis=0) - rectangle
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    short_length = float(np.min(edge_lengths))
    if short_length <= 1.0:
        raise CanDetectionError("red can contour is too small")

    points = contour[:, 0, :].astype(np.float64)
    bottom_y = float(np.max(points[:, 1]))
    bottom_band = points[
        points[:, 1]
        >= bottom_y - BASE_BAND_SHORT_AXIS_FRACTION * short_length
    ]
    if len(bottom_band) < 5:
        raise CanDetectionError("not enough red lower-contour points")

    left = bottom_band[int(np.argmin(bottom_band[:, 0]))]
    right = bottom_band[int(np.argmax(bottom_band[:, 0]))]
    major_vector = right - left
    major_length = float(np.linalg.norm(major_vector))
    if major_length <= 1.0:
        raise CanDetectionError("red can base is too narrow")

    center = 0.5 * (left + right)
    major_axis = major_vector / major_length
    downward_normal = np.asarray(
        (-major_axis[1], major_axis[0]),
        dtype=np.float64,
    )
    if downward_normal[1] < 0.0:
        downward_normal *= -1.0
    lower_extent = float(
        np.max((bottom_band - center) @ downward_normal)
    )
    axes = (
        max(1, int(round(0.5 * major_length))),
        max(1, int(round(lower_extent))),
    )
    angle_deg = float(
        np.degrees(np.arctan2(major_vector[1], major_vector[0]))
    )
    return (
        (int(round(float(center[0]))), int(round(float(center[1])))),
        axes,
        angle_deg,
        contour,
    )


def draw_can_box(
    frame: np.ndarray,
    detection: CanDetection | None,
) -> np.ndarray:
    display = frame.copy()
    if detection is None:
        return display
    x1, y1, x2, y2 = detection.box_xyxy
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        display,
        f"{detection.label} {detection.confidence:.2f}",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return display


def draw_can_base(
    frame: np.ndarray,
    detection: CanDetection,
) -> np.ndarray:
    display = draw_can_box(frame, detection)
    if (
        detection.base_center is None
        or detection.base_axes is None
        or detection.base_angle_deg is None
    ):
        return display
    if detection.red_contour is not None:
        cv2.drawContours(
            display,
            [detection.red_contour],
            -1,
            (0, 255, 255),
            3,
        )
    cv2.ellipse(
        display,
        detection.base_center,
        detection.base_axes,
        detection.base_angle_deg,
        0.0,
        360.0,
        (255, 0, 255),
        3,
    )
    cv2.drawMarker(
        display,
        detection.base_center,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=3,
    )
    return display
