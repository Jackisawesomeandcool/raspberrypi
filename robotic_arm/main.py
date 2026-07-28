#!/usr/bin/env python3
"""Run YOLO-World vision, workspace calibration, and robot control."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
VISION_DIR = PROJECT_ROOT / "vision"
CONTROL_DIR = PROJECT_ROOT / "control"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

for module_dir in (VISION_DIR, CONTROL_DIR, CALIBRATION_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from can_yolo_world import (  # noqa: E402
    CAN_PROMPT,
    CanDetection,
    CanDetectionError,
    YoloWorldCanDetector,
    draw_can_base,
    draw_can_box,
)
from calibrate_workspace import calibrate_workspace  # noqa: E402
from pixel_to_robot import DEFAULT_CALIBRATION, pixel_to_robot  # noqa: E402
from _shared import (  # noqa: E402
    CalibrationError,
    load_intrinsics,
    load_json,
    save_json,
)
from pick_and_place import (  # noqa: E402
    DEFAULT_GRIPPER_CLOSED_DEG,
    DEFAULT_GRIPPER_OPEN_DEG,
    DEFAULT_LIFT_MM,
    DEFAULT_RATE_HZ,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_PORT,
    Frame,
    IKError,
    plan_pick_and_place,
    run_frames,
)


WINDOW_NAME = "Bottle pipeline"
DEFAULT_INTRINSICS = CALIBRATION_DIR / "output" / "camera_intrinsics.json"
DEFAULT_CALIBRATION_ANNOTATED = (
    CALIBRATION_DIR / "output" / "workspace_plane_annotated.png"
)
DEFAULT_REFERENCE_DIR = VISION_DIR / "captures"
YOLO_PREVIEW_INTERVAL_S = 1.0
CAN_DETECTOR = YoloWorldCanDetector()


def set_detection_target(target: str) -> str:
    """Switch the loaded YOLO-World model to one text category."""

    return CAN_DETECTOR.set_target(target)


def get_detection_target() -> str:
    return CAN_DETECTOR.target


def detection_target_supports_control() -> bool:
    return CAN_DETECTOR.supports_grasp_geometry


def detect_target(frame: cv2.typing.MatLike) -> CanDetection | None:
    """Return a generic box result for the active text category."""

    return CAN_DETECTOR.detect(frame)


def _validate_image_size(
    frame: cv2.typing.MatLike,
    expected_size: tuple[int, int],
) -> None:
    actual_size = (frame.shape[1], frame.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"输入画面为 {actual_size}，但标定对应 {expected_size}"
        )


def _synchronize_reference_and_calibration(
    frame: cv2.typing.MatLike,
    *,
    reference_path: Path,
    intrinsics_path: Path,
    calibration_path: Path,
    annotated_path: Path,
) -> tuple[int, int]:
    camera_matrix, distortion, image_size, intrinsics_payload = load_intrinsics(
        intrinsics_path
    )
    _validate_image_size(frame, image_size)
    initial_image_to_board = None
    if calibration_path.is_file():
        previous_calibration = load_json(calibration_path)
        initial_image_to_board = previous_calibration.get(
            "pixel_to_board_grid_homography"
        )
    result, annotated = calibrate_workspace(
        frame,
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
        initial_image_to_board=initial_image_to_board,
    )

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(reference_path), frame):
        raise CalibrationError(f"无法保存参考图：{reference_path}")

    result["intrinsics_file"] = str(intrinsics_path)
    result["intrinsics_rms_reprojection_error_px"] = intrinsics_payload[
        "rms_reprojection_error_px"
    ]
    result["source"] = str(reference_path)
    save_json(calibration_path, result)

    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(annotated_path), annotated):
        raise CalibrationError(f"无法保存标定标注图：{annotated_path}")

    print(
        json.dumps(
            {
                "reference": str(reference_path),
                "calibration": str(calibration_path),
                "calibration_annotated": str(annotated_path),
                "matching_method": result["checkerboard"][
                    "matching_method"
                ],
                "rmse_mm": result["rmse_mm"],
                "max_error_mm": result["max_error_mm"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return image_size


def _capture_reference_and_calibrate(
    camera: cv2.VideoCapture,
    args: argparse.Namespace,
) -> tuple[Path, tuple[int, int]]:
    print(
        "请保持相机和棋盘格固定、移走杯子；"
        "按 c 拍摄 workspace calibration 参考图。",
        flush=True,
    )
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("读取相机画面失败。")
        display = _draw_status(
            frame,
            (
                "Remove bottle; keep camera and checkerboard fixed.",
                "C: capture calibration/reference    Q/Esc: cancel",
            ),
            color=(255, 255, 255),
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key != ord("c"):
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        reference_path = (
            DEFAULT_REFERENCE_DIR / f"workspace_reference_{timestamp}.png"
        )
        try:
            image_size = _synchronize_reference_and_calibration(
                frame.copy(),
                reference_path=reference_path,
                intrinsics_path=args.intrinsics,
                calibration_path=args.calibration,
                annotated_path=args.calibration_annotated,
            )
            print(
                "参考图与工作区标定已同步；保持相机和棋盘格不动，"
                "现在可放入杯子。",
                flush=True,
            )
            return reference_path, image_size
        except (CalibrationError, ValueError, cv2.error) as error:
            print(f"参考图标定失败，请调整后重新按 c：{error}", flush=True)


def run_vision(
    frame: cv2.typing.MatLike,
    *,
    reference_path: Path,
) -> tuple[tuple[int, int], cv2.typing.MatLike, float]:
    """Locate the soda-can base with YOLO-World and box-constrained geometry."""

    del reference_path
    if not CAN_DETECTOR.supports_grasp_geometry:
        raise CanDetectionError(
            f"当前目标 {CAN_DETECTOR.target!r} 仅支持检测框；"
            f"只有 {CAN_PROMPT!r} 可进入坐标换算和控制"
        )
    detection = detect_target(frame)
    return _vision_result(frame, detection)


def _vision_result(
    frame: cv2.typing.MatLike,
    detection: CanDetection | None,
) -> tuple[tuple[int, int], cv2.typing.MatLike, float]:
    if detection is None:
        raise CanDetectionError("YOLO-World 未检测到 soda can")
    if detection.base_center is None:
        raise CanDetectionError(
            "YOLO-World 已检测到易拉罐，但框内未找到红色罐底轮廓"
        )
    return (
        detection.base_center,
        draw_can_base(frame, detection),
        detection.confidence,
    )


def run_calibration(
    bottle_bottom_pixel: tuple[int, int],
    *,
    calibration_path: Path,
    grasp_z_mm: float,
) -> list[float]:
    """Convert the detected bottle base pixel to the robot grasp point."""

    coordinate = pixel_to_robot(
        bottle_bottom_pixel,
        calibration_path=calibration_path,
    )
    coordinate[2] = float(grasp_z_mm)
    return coordinate


def plan_control(
    pick_xyz_mm: Sequence[float],
    place_xyz_mm: Sequence[float],
    current_control_deg: Sequence[float],
    *,
    lift_mm: float,
    rate_hz: float,
    speed_scale: float,
    gripper_open_deg: float,
    gripper_closed_deg: float,
) -> list[Frame]:
    """Plan the complete motion without sending any command."""

    frames = plan_pick_and_place(
        pick_xyz_mm,
        place_xyz_mm,
        current_control_deg,
        lift_mm=lift_mm,
        rate_hz=rate_hz,
        speed_scale=speed_scale,
        gripper_open_deg=gripper_open_deg,
        gripper_closed_deg=gripper_closed_deg,
    )
    print(
        json.dumps(
            {
                "control": {
                    "frame_count": len(frames),
                    "duration_s": round(frames[-1].time_s, 3),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return frames


def _draw_status(
    frame: cv2.typing.MatLike,
    lines: Sequence[str],
    *,
    color: tuple[int, int, int],
) -> cv2.typing.MatLike:
    display = frame.copy()
    line_height = 34
    panel_height = 20 + line_height * len(lines)
    cv2.rectangle(
        display,
        (0, 0),
        (display.shape[1], panel_height),
        (0, 0, 0),
        thickness=-1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (24, 32 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return display


def _prepare_motion(
    frame: cv2.typing.MatLike,
    args: argparse.Namespace,
    detection: CanDetection | None = None,
) -> tuple[list[Frame], cv2.typing.MatLike]:
    if detection is None:
        bottle_bottom_pixel, annotated, confidence = run_vision(
            frame,
            reference_path=args.reference,
        )
    else:
        bottle_bottom_pixel, annotated, confidence = _vision_result(
            frame,
            detection,
        )
    pick = run_calibration(
        bottle_bottom_pixel,
        calibration_path=args.calibration,
        grasp_z_mm=args.grasp_z_mm,
    )
    print(
        json.dumps(
            {
                "vision": {
                    "backend": "yolo_world_soda_can",
                    "confidence": round(confidence, 6),
                    "bottle_bottom_pixel": list(bottle_bottom_pixel),
                },
                "calibration": {
                    "pick_j1_xyz_mm": pick,
                    "place_j1_xyz_mm": args.place,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    planning_display = _draw_status(
        annotated,
        (
            f"J1 XYZ = ({pick[0]:.1f}, {pick[1]:.1f}, {pick[2]:.1f}) mm",
            "Detection ready; planning trajectory...",
        ),
        color=(0, 255, 255),
    )
    cv2.imshow(WINDOW_NAME, planning_display)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        raise KeyboardInterrupt

    frames = plan_control(
        pick,
        args.place,
        args.current_control,
        lift_mm=args.lift_mm,
        rate_hz=args.rate_hz,
        speed_scale=args.speed_scale,
        gripper_open_deg=args.gripper_open_deg,
        gripper_closed_deg=args.gripper_closed_deg,
    )
    mode = "SEND ENABLED" if args.send else "DRY RUN"
    display = _draw_status(
        annotated,
        (
            f"J1 XYZ = ({pick[0]:.1f}, {pick[1]:.1f}, {pick[2]:.1f}) mm",
            f"I: start {mode}    S: recapture    Q/Esc: cancel",
        ),
        color=(0, 255, 0),
    )
    return frames, display


def _interactive_capture(
    args: argparse.Namespace,
) -> list[Frame]:
    camera = None
    static_frame = None
    if args.image is not None:
        static_frame = cv2.imread(str(args.image))
        if static_frame is None:
            raise ValueError(f"无法读取图片：{args.image}")
    else:
        camera = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
        if not camera.isOpened():
            raise RuntimeError(
                f"无法打开 camera index {args.camera_index}；"
                "请关闭 Photo Booth 后重试。"
            )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    prepared_frames: list[Frame] | None = None
    prepared_display = None
    current_frame = static_frame
    inference_executor: ThreadPoolExecutor | None = None
    preview_future: Future[CanDetection | None] | None = None
    preview_detection: CanDetection | None = None
    next_preview_inference_s = 0.0
    try:
        if camera is not None:
            reference_path, expected_size = _capture_reference_and_calibrate(
                camera,
                args,
            )
        else:
            if args.reference is None:
                raise ValueError(
                    "使用 --image 时必须同时提供 --reference，"
                    "该参考图会在启动时同步重算工作区标定。"
                )
            reference_frame = cv2.imread(str(args.reference))
            if reference_frame is None:
                raise ValueError(f"无法读取参考图：{args.reference}")
            reference_path = args.reference
            expected_size = _synchronize_reference_and_calibration(
                reference_frame,
                reference_path=reference_path,
                intrinsics_path=args.intrinsics,
                calibration_path=args.calibration,
                annotated_path=args.calibration_annotated,
            )

        args.reference = reference_path
        if static_frame is not None:
            _validate_image_size(static_frame, expected_size)

        if current_frame is None:
            ok, current_frame = camera.read()
            if not ok or current_frame is None:
                raise RuntimeError("读取相机画面失败。")
            _validate_image_size(current_frame, expected_size)
        loading_display = _draw_status(
            current_frame,
            ("Loading YOLO-World soda-can detector...",),
            color=(0, 255, 255),
        )
        cv2.imshow(WINDOW_NAME, loading_display)
        cv2.waitKey(1)
        CAN_DETECTOR.load()
        inference_executor = ThreadPoolExecutor(max_workers=1)

        print(
            "YOLO-World 视觉定位已就绪：预览约每秒识别一次；"
            "按 s 抓取易拉罐并计算位置，"
            "定位成功后按 i 开始，按 q 或 Esc 退出。",
            flush=True,
        )
        while True:
            if prepared_frames is None:
                if camera is not None:
                    ok, current_frame = camera.read()
                    if not ok or current_frame is None:
                        raise RuntimeError("读取相机画面失败。")
                    _validate_image_size(current_frame, expected_size)
                if current_frame is None:
                    raise RuntimeError("没有取得输入画面。")
                if preview_future is not None and preview_future.done():
                    try:
                        preview_detection = preview_future.result()
                    except Exception as error:
                        preview_detection = None
                        print(f"YOLO 预览失败：{error}", flush=True)
                    preview_future = None
                now_s = time.monotonic()
                if (
                    preview_future is None
                    and now_s >= next_preview_inference_s
                ):
                    preview_future = inference_executor.submit(
                        CAN_DETECTOR.detect,
                        current_frame.copy(),
                    )
                    next_preview_inference_s = (
                        now_s + YOLO_PREVIEW_INTERVAL_S
                    )
                display = _draw_status(
                    current_frame,
                    ("S: capture soda can    Q/Esc: cancel",),
                    color=(255, 255, 255),
                )
                display = draw_can_box(display, preview_detection)
            else:
                display = prepared_display

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            if key == ord("i"):
                if prepared_frames is None:
                    print("请先按 s 抓取画面并计算位置。", flush=True)
                    continue
                return prepared_frames
            if key != ord("s"):
                continue
            if prepared_frames is not None:
                prepared_frames = None
                prepared_display = None
                print("已恢复实时画面，请再次按 s 抓取。", flush=True)
                continue

            try:
                captured_frame = current_frame.copy()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                capture_path = (
                    DEFAULT_REFERENCE_DIR
                    / f"bottle_capture_{timestamp}.png"
                )
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(capture_path), captured_frame):
                    raise RuntimeError(f"无法保存捕捉原图：{capture_path}")
                print(f"已保存捕捉原图：{capture_path}", flush=True)

                detecting_display = _draw_status(
                    captured_frame,
                    ("Captured; running fresh YOLO-World detection...",),
                    color=(0, 255, 255),
                )
                cv2.imshow(WINDOW_NAME, detecting_display)
                cv2.waitKey(1)
                fresh_detection = inference_executor.submit(
                    CAN_DETECTOR.detect,
                    captured_frame,
                ).result()
                preview_future = None
                preview_detection = fresh_detection
                next_preview_inference_s = (
                    time.monotonic() + YOLO_PREVIEW_INTERVAL_S
                )
                prepared_frames, prepared_display = _prepare_motion(
                    captured_frame,
                    args,
                    fresh_detection,
                )
                print(
                    "位置和轨迹已准备；确认画面与安全状态后按 i 开始。",
                    flush=True,
                )
            except (
                CanDetectionError,
                IKError,
                RuntimeError,
                ValueError,
                cv2.error,
            ) as error:
                print(f"定位或规划失败：{error}", flush=True)
    finally:
        if inference_executor is not None:
            inference_executor.shutdown(wait=True, cancel_futures=True)
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def run_control(
    frames: Sequence[Frame],
    *,
    send: bool,
    robot_ip: str,
    robot_port: int,
) -> None:
    """Run the prepared frames with immediate per-command angle logs."""

    mode = "实体发送" if send else "dry-run"
    print(f"开始{mode}，角度日志将按时间实时输出。", flush=True)
    run_frames(
        frames,
        send=send,
        robot_ip=robot_ip,
        robot_port=robot_port,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", nargs=3, type=float, required=True)
    parser.add_argument("--grasp-z-mm", type=float, required=True)
    parser.add_argument(
        "--current-control",
        nargs=5,
        type=float,
        required=True,
        metavar=("J1", "J2", "J3", "J4", "J6"),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "existing workspace-calibration reference for --image mode; "
            "live camera mode always captures a new calibration image"
        ),
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--calibration-annotated",
        type=Path,
        default=DEFAULT_CALIBRATION_ANNOTATED,
    )
    parser.add_argument("--lift-mm", type=float, default=DEFAULT_LIFT_MM)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--gripper-open-deg",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_DEG,
    )
    parser.add_argument(
        "--gripper-closed-deg",
        type=float,
        default=DEFAULT_GRIPPER_CLOSED_DEG,
    )
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
        frames = _interactive_capture(args)
        run_control(
            frames,
            send=args.send,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
        )
        return 0
    except KeyboardInterrupt:
        print("已取消，未发送机械臂命令。")
        return 130
    except (
        CanDetectionError,
        CalibrationError,
        IKError,
        OSError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
