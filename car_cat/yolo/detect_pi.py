"""Lightweight YOLOv8 ONNX inference for Raspberry Pi using OpenCV DNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def detect(frame: np.ndarray, net: cv2.dnn.Net, size: int, threshold: float) -> np.ndarray:
    image, scale, pad_x, pad_y = letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (size, size), swapRB=True)
    net.setInput(blob)
    predictions = np.squeeze(net.forward())
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes, scores = [], []
    for prediction in predictions:
        cx, cy, width, height = prediction[:4]
        confidence = float(prediction[4 + 15])  # COCO class 15 = cat
        if confidence < threshold:
            continue
        x = int((cx - width / 2 - pad_x) / scale)
        y = int((cy - height / 2 - pad_y) / scale)
        box_w = int(width / scale)
        box_h = int(height / scale)
        boxes.append([max(0, x), max(0, y), box_w, box_h])
        scores.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.45)
    result = frame.copy()
    for index in indices:
        index = int(index)
        x, y, box_w, box_h = boxes[index]
        cv2.rectangle(result, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        cv2.putText(
            result,
            f"cat {scores[index] * 100:.1f}%",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n_opencv.onnx")
    parser.add_argument("--source", default="0")
    parser.add_argument("--output", default="runs/pi_camera.mp4")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    net = cv2.dnn.readNetFromONNX(args.model)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入源: {source}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = detect(frame, net, args.imgsz, args.conf)
        if is_image:
            cv2.imwrite(str(output), result)
            print(f"saved: {output}")
            break

        if writer is None:
            h, w = result.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(result)
        if args.show:
            cv2.imshow("cat detection on Raspberry Pi", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
        print(f"saved: {output}")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
