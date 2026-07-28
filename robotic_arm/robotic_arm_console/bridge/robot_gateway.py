#!/usr/bin/env python3
"""Local HTTP bridge between the web console and advanced/main.py."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import ipaddress
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J6")
DEFAULT_INITIAL_CONTROL_DEG = [90.0, 60.0, 90.0, 90.0, 150.0]
ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://axis-maker-arm-console.ninthday.chatgpt.site",
}
CONTROL_LINE = re.compile(
    r"phase=(?P<phase>\S+)\s+control_deg=\s*"
    r"(?P<J1>-?\d+)\s+(?P<J2>-?\d+)\s+(?P<J3>-?\d+)\s+"
    r"(?P<J4>-?\d+)\s+(?P<J6>-?\d+)"
)
AVFOUNDATION_DEVICE = re.compile(r"\[(?P<index>\d+)\]\s+(?P<name>.+)$")


def list_avfoundation_cameras() -> list[tuple[int, str]]:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    cameras: list[tuple[int, str]] = []
    in_video_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue
        match = AVFOUNDATION_DEVICE.search(line)
        if match is None:
            continue
        name = match.group("name").strip()
        if name.lower().startswith("capture screen"):
            continue
        cameras.append((int(match.group("index")), name))
    return cameras


def discover_continuity_camera(
    cameras: list[tuple[int, str]],
) -> tuple[int, str] | None:
    """Return the AVFoundation index and name of an attached iPhone camera."""

    for _, name in cameras:
        lowered = name.lower()
        if "iphone" in lowered or "continuity" in lowered:
            # OpenCV exposes a different order from ffmpeg/AVFoundation on
            # this Mac. Index 1 is the iPhone only while the named device is
            # present; callers must not bypass this name check.
            return 1, name
    return None


def is_allowed_origin(origin: str | None) -> bool:
    if origin in ALLOWED_ORIGINS:
        return True
    if origin is None:
        return False
    try:
        parsed = urlparse(origin)
        if parsed.scheme != "http" or parsed.port != 3000:
            return False
        address = ipaddress.ip_address(parsed.hostname or "")
        return address.is_private
    except (ValueError, TypeError):
        return False


def load_pipeline(project_root: Path) -> ModuleType:
    main_path = project_root / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"main.py not found: {main_path}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("maker_arm_pipeline", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeState:
    def __init__(self, pipeline: ModuleType) -> None:
        self.pipeline = pipeline
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.operation_lock = threading.Lock()
        self.events: deque[dict[str, object]] = deque(maxlen=1200)
        self.logs: deque[dict[str, object]] = deque(maxlen=240)
        self.event_id = 0
        self.reference_path: Path | None = None
        self.expected_size: tuple[int, int] | None = None
        self.pick_xyz_mm: list[float] | None = None
        self.current_control_deg: list[float] | None = None
        self.planned_frames: list[object] | None = None
        self.planned_place_mm: list[float] | None = None
        self.planned_frame_count: int | None = None
        self.planned_duration_s: float | None = None
        self.detection_target = pipeline.get_detection_target()
        self.target_mode = (
            "control"
            if pipeline.detection_target_supports_control()
            else "detection_only"
        )
        self.last_detection: dict[str, object] | None = None
        self.camera_index: int | None = None
        self.camera_name: str | None = None
        self.camera_resolution: tuple[int, int] | None = None
        self.stage = "idle"
        self.moving = False

    def publish(self, event_type: str, data: dict[str, object]) -> None:
        with self.condition:
            self.event_id += 1
            event = {"id": self.event_id, "type": event_type, "data": data}
            self.events.append(event)
            self.condition.notify_all()

    def log(self, level: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        }
        with self.lock:
            self.logs.append(entry)
        self.publish("log", entry)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "stage": self.stage,
                "moving": self.moving,
                "referencePath": (
                    str(self.reference_path) if self.reference_path else None
                ),
                "pickMm": self.pick_xyz_mm,
                "currentControlDeg": self.current_control_deg,
                "plannedPlaceMm": self.planned_place_mm,
                "plannedFrameCount": self.planned_frame_count,
                "plannedDurationS": self.planned_duration_s,
                "detectionTarget": self.detection_target,
                "targetMode": self.target_mode,
                "lastDetection": self.last_detection,
                "logs": list(self.logs),
                "detector": f"YOLO-World {self.detection_target}",
                "send": self.moving,
                "requiresExplicitSend": True,
                "speedScale": 1.8,
                "cameraIndex": self.camera_index,
                "cameraName": self.camera_name,
                "cameraResolution": self.camera_resolution,
            }


class CameraHub:
    def __init__(self, cv2_module: ModuleType, state: RuntimeState) -> None:
        self.cv2 = cv2_module
        self.state = state
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.capture = None
        self.frame = None
        self.preview_detection = None
        self.preview_error: str | None = None
        self.camera_index: int | None = None
        self.start_error: str | None = None
        self.running = False

    def start(self, camera_index: int) -> None:
        with self.lock:
            if self.running and self.camera_index == camera_index:
                return
            if self.running:
                raise RuntimeError(
                    "camera is already active; restart the gateway to change index"
                )
            self.camera_index = camera_index
            with self.state.lock:
                self.state.camera_index = camera_index
            backend = getattr(self.cv2, "CAP_AVFOUNDATION", 0)
            capture = self.cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                self.start_error = (
                    "cannot open the named Continuity Camera; close Photo Booth "
                    "and reconnect the iPhone"
                )
                raise RuntimeError(self.start_error)
            capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture = capture
            self.frame = None
            self.preview_detection = None
            self.preview_error = None
            self.start_error = None
            self.running = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            threading.Thread(target=self._preview_loop, daemon=True).start()
        try:
            self.snapshot(camera_index, timeout_s=10.0)
        except RuntimeError:
            self.stop()
            self.start_error = "Continuity Camera opened but returned no frame"
            raise RuntimeError(self.start_error)
        self.state.log(
            "success",
            f"OpenCV camera {camera_index} connected",
        )

    def ensure_started(self, camera_index: int) -> None:
        with self.lock:
            if self.camera_index != camera_index:
                raise RuntimeError(
                    f"gateway started for camera index {self.camera_index}; "
                    f"restart with --camera-index {camera_index}"
                )
            if self.running:
                return
            raise RuntimeError(
                self.start_error
                or "camera is unavailable; restart the gateway"
            )

    def _read_loop(self) -> None:
        failures = 0
        while True:
            with self.lock:
                if not self.running or self.capture is None:
                    return
                capture = self.capture
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    break
                time.sleep(0.05)
                continue
            failures = 0
            publish_state = False
            with self.state.lock:
                if self.state.camera_resolution is None:
                    self.state.camera_resolution = (
                        int(frame.shape[1]),
                        int(frame.shape[0]),
                    )
                    publish_state = True
            with self.condition:
                self.frame = frame
                self.condition.notify_all()
            if publish_state:
                self.state.publish("state", self.state.snapshot())
        with self.lock:
            stopped_unexpectedly = self.running
            self.running = False
            if self.capture is capture:
                self.capture = None
        capture.release()
        if stopped_unexpectedly:
            self.state.log("error", "Continuity Camera stream stopped")

    def _preview_loop(self) -> None:
        interval_s = float(self.state.pipeline.YOLO_PREVIEW_INTERVAL_S)
        while True:
            started_s = time.monotonic()
            with self.condition:
                if not self.running:
                    return
                if self.frame is None:
                    self.condition.wait(timeout=0.1)
                    continue
                frame = self.frame.copy()
            try:
                detection = self.state.pipeline.CAN_DETECTOR.detect(frame)
                error_message = None
            except Exception as error:
                detection = None
                error_message = str(error)
            with self.lock:
                if not self.running:
                    return
                active_target = self.state.pipeline.get_detection_target()
                self.preview_detection = (
                    detection
                    if detection is not None
                    and detection.label == active_target
                    else None
                )
                previous_error = self.preview_error
                self.preview_error = error_message
            if error_message is not None and error_message != previous_error:
                self.state.log(
                    "error",
                    f"YOLO preview failed · {error_message}",
                )
            time.sleep(max(0.05, interval_s - (time.monotonic() - started_s)))

    def stop(self) -> None:
        with self.lock:
            self.running = False
            capture = self.capture
            self.capture = None
            self.preview_detection = None
            self.condition.notify_all()
        if capture is not None:
            capture.release()

    def clear_preview_detection(self) -> None:
        with self.lock:
            self.preview_detection = None
            self.preview_error = None

    def snapshot(self, camera_index: int, timeout_s: float = 4.0):
        self.ensure_started(camera_index)
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("camera opened but no frame was received")
                self.condition.wait(remaining)
            return self.frame.copy()

    def jpeg(self, camera_index: int) -> bytes:
        frame = self.snapshot(camera_index)
        with self.lock:
            detection = self.preview_detection
        frame = self.state.pipeline.draw_can_box(frame, detection)
        ok, encoded = self.cv2.imencode(
            ".jpg",
            frame,
            [int(self.cv2.IMWRITE_JPEG_QUALITY), 86],
        )
        if not ok:
            raise RuntimeError("failed to encode camera frame")
        return encoded.tobytes()


class ControlOutput(io.TextIOBase):
    def __init__(self, state: RuntimeState) -> None:
        self.state = state
        self.pending = ""
        self.last_phase = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._handle(line.strip())
        return len(text)

    def flush(self) -> None:
        if self.pending.strip():
            self._handle(self.pending.strip())
            self.pending = ""

    def _handle(self, line: str) -> None:
        if not line:
            return
        match = CONTROL_LINE.search(line)
        if match is None:
            self.state.log("info", line)
            return
        phase = match.group("phase")
        values = [float(match.group(name)) for name in JOINT_NAMES]
        control = dict(zip(JOINT_NAMES, values, strict=True))
        with self.state.lock:
            self.state.current_control_deg = values
        self.state.publish(
            "control",
            {
                "timestamp": time.time(),
                "phase": phase,
                "controlDeg": control,
            },
        )
        if phase != self.last_phase:
            self.state.log("info", f"control phase → {phase}")
            self.last_phase = phase


class Gateway:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.pipeline = load_pipeline(project_root)
        self.state = RuntimeState(self.pipeline)
        self.camera = CameraHub(self.pipeline.cv2, self.state)
        self.reference_dir = project_root / "calibration" / "data"
        self.intrinsics_path = (
            project_root / "calibration" / "output" / "camera_intrinsics.json"
        )
        self.calibration_path = self.pipeline.DEFAULT_CALIBRATION
        self.calibration_annotated_path = (
            project_root
            / "calibration"
            / "output"
            / "workspace_plane_annotated.png"
        )

    def _camera_index(self, payload: dict[str, object]) -> int:
        raw = payload.get("cameraIndex", "auto")
        if raw == "auto":
            value = self.camera.camera_index
            if value is None:
                raise RuntimeError(
                    "Apple Continuity Camera not found; "
                    "connect the iPhone and restart the gateway"
                )
            return value
        value = int(raw)
        if not 0 <= value <= 16:
            raise ValueError("cameraIndex must be between 0 and 16")
        return value

    @staticmethod
    def _finite_vector(
        payload: dict[str, object],
        key: str,
        size: int,
    ) -> list[float]:
        raw = payload.get(key)
        if not isinstance(raw, list) or len(raw) != size:
            raise ValueError(f"{key} must contain {size} numbers")
        values = [float(value) for value in raw]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{key} must contain finite numbers")
        return values

    def set_detection_target(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if not self.state.operation_lock.acquire(blocking=False):
            raise RuntimeError("another operation is already running")
        try:
            target = self.pipeline.set_detection_target(payload.get("target"))
            mode = (
                "control"
                if self.pipeline.detection_target_supports_control()
                else "detection_only"
            )
            self.camera.clear_preview_detection()
            with self.state.lock:
                self.state.detection_target = target
                self.state.target_mode = mode
                self.state.last_detection = None
                self.state.pick_xyz_mm = None
                self.state.planned_frames = None
                self.state.planned_place_mm = None
                self.state.planned_frame_count = None
                self.state.planned_duration_s = None
                self.state.stage = (
                    "calibrated"
                    if self.state.reference_path is not None
                    else "idle"
                )
            self.state.log(
                "warning" if mode == "detection_only" else "success",
                f"detection target → {target} · {mode}",
            )
            snapshot = self.state.snapshot()
            self.state.publish("state", snapshot)
            return {"ok": True, **snapshot}
        finally:
            self.state.operation_lock.release()

    def calibrate(self, payload: dict[str, object]) -> dict[str, object]:
        if not self.state.operation_lock.acquire(blocking=False):
            raise RuntimeError("another operation is already running")
        try:
            self.state.log("info", "calibration started")
            frame = self.camera.snapshot(self._camera_index(payload))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            reference_path = (
                self.reference_dir
                / f"workspace_reference_ui_{timestamp}.png"
            )
            image_size = self.pipeline._synchronize_reference_and_calibration(
                frame.copy(),
                reference_path=reference_path,
                intrinsics_path=self.intrinsics_path,
                calibration_path=self.calibration_path,
                annotated_path=self.calibration_annotated_path,
            )
            with self.state.lock:
                self.state.reference_path = reference_path
                self.state.expected_size = image_size
                self.state.pick_xyz_mm = None
                self.state.last_detection = None
                self.state.planned_frames = None
                self.state.planned_place_mm = None
                self.state.planned_frame_count = None
                self.state.planned_duration_s = None
                self.state.stage = "calibrated"
            self.state.log(
                "success",
                f"calibration complete · reference {reference_path.name}",
            )
            self.state.publish("state", self.state.snapshot())
            return {
                "ok": True,
                "stage": "calibrated",
                "resolution": list(image_size),
                "referencePath": str(reference_path),
            }
        finally:
            self.state.operation_lock.release()

    def capture_object(self, payload: dict[str, object]) -> dict[str, object]:
        if not self.state.operation_lock.acquire(blocking=False):
            raise RuntimeError("another operation is already running")
        try:
            with self.state.lock:
                reference = self.state.reference_path
                expected_size = self.state.expected_size
                target = self.state.detection_target
                target_mode = self.state.target_mode

            self.state.log("info", "capture started")
            frame = self.camera.snapshot(self._camera_index(payload))

            if target_mode == "detection_only":
                detection = self.pipeline.detect_target(frame)
                if detection is None:
                    raise RuntimeError(
                        f"YOLO-World did not detect {target!r}"
                    )
                result = {
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "box": list(detection.box_xyxy),
                }
                with self.state.lock:
                    self.state.last_detection = result
                    self.state.pick_xyz_mm = None
                    self.state.planned_frames = None
                    self.state.planned_place_mm = None
                    self.state.planned_frame_count = None
                    self.state.planned_duration_s = None
                    self.state.stage = (
                        "calibrated" if reference is not None else "idle"
                    )
                self.state.log(
                    "success",
                    f"detected {detection.label} · "
                    f"confidence {detection.confidence:.2f} · detection only",
                )
                self.state.publish("state", self.state.snapshot())
                return {
                    "ok": True,
                    "stage": self.state.stage,
                    "detectionTarget": target,
                    "targetMode": target_mode,
                    "lastDetection": result,
                    "controlReady": False,
                }

            grasp_z = float(payload.get("graspZMm"))
            if not math.isfinite(grasp_z):
                raise ValueError("graspZMm must be a finite number")
            if reference is None or expected_size is None:
                raise RuntimeError("calibration is required before capture")
            actual_size = (int(frame.shape[1]), int(frame.shape[0]))
            if actual_size != expected_size:
                raise ValueError(
                    f"camera frame is {actual_size}, calibration expects "
                    f"{expected_size}"
                )
            center, _, confidence = self.pipeline.run_vision(
                frame,
                reference_path=reference,
            )
            pick = self.pipeline.run_calibration(
                center,
                calibration_path=self.calibration_path,
                grasp_z_mm=grasp_z,
            )
            with self.state.lock:
                self.state.pick_xyz_mm = [float(value) for value in pick]
                self.state.last_detection = None
                self.state.planned_frames = None
                self.state.planned_place_mm = None
                self.state.planned_frame_count = None
                self.state.planned_duration_s = None
                self.state.stage = "captured"
            self.state.log(
                "success",
                "object captured · "
                f"pick [{pick[0]:.1f}, {pick[1]:.1f}, {pick[2]:.1f}] mm",
            )
            self.state.publish("state", self.state.snapshot())
            return {
                "ok": True,
                "stage": "captured",
                "pickMm": pick,
                "pixel": list(center),
                "confidence": confidence,
                "detector": f"YOLO-World {target}",
                "detectionTarget": target,
                "targetMode": target_mode,
                "controlReady": True,
            }
        finally:
            self.state.operation_lock.release()

    def move(self, payload: dict[str, object]) -> dict[str, object]:
        if not self.state.operation_lock.acquire(blocking=False):
            raise RuntimeError("another operation is already running")
        try:
            place = self._finite_vector(payload, "placeMm", 3)
            send = payload.get("send", False)
            if not isinstance(send, bool):
                raise ValueError("send must be a boolean")

            with self.state.lock:
                target_mode = self.state.target_mode
                pick = (
                    list(self.state.pick_xyz_mm)
                    if self.state.pick_xyz_mm is not None
                    else None
                )
                current = (
                    list(self.state.current_control_deg)
                    if self.state.current_control_deg is not None
                    else list(DEFAULT_INITIAL_CONTROL_DEG)
                )
                planned_frames = self.state.planned_frames
                planned_place = (
                    list(self.state.planned_place_mm)
                    if self.state.planned_place_mm is not None
                    else None
                )
            if target_mode != "control":
                raise RuntimeError(
                    "the active detection target does not support robot control"
                )
            if pick is None:
                raise RuntimeError("capture is required before move")

            if not send:
                frames = self.pipeline.plan_control(
                    pick,
                    place,
                    current,
                    lift_mm=self.pipeline.DEFAULT_LIFT_MM,
                    rate_hz=self.pipeline.DEFAULT_RATE_HZ,
                    speed_scale=1.8,
                    gripper_open_deg=self.pipeline.DEFAULT_GRIPPER_OPEN_DEG,
                    gripper_closed_deg=self.pipeline.DEFAULT_GRIPPER_CLOSED_DEG,
                )
                frame_count = len(frames)
                duration_s = float(frames[-1].time_s)
                with self.state.lock:
                    self.state.current_control_deg = current
                    self.state.planned_frames = list(frames)
                    self.state.planned_place_mm = place
                    self.state.planned_frame_count = frame_count
                    self.state.planned_duration_s = duration_s
                    self.state.stage = "planned"
                self.state.log(
                    "success",
                    f"motion planned · {frame_count} frames · "
                    f"{duration_s:.2f} s · physical send disabled",
                )
                self.state.publish("state", self.state.snapshot())
                self.state.operation_lock.release()
                return {
                    "ok": True,
                    "stage": "planned",
                    "pickMm": pick,
                    "placeMm": place,
                    "frameCount": frame_count,
                    "durationS": duration_s,
                    "send": False,
                    "speedScale": 1.8,
                }

            if planned_frames is None or planned_place is None:
                raise RuntimeError("motion planning is required before physical send")
            if any(
                not math.isclose(actual, expected, abs_tol=1e-9)
                for actual, expected in zip(place, planned_place, strict=True)
            ):
                raise RuntimeError(
                    "placeMm changed after planning; plan the motion again"
                )

            with self.state.lock:
                self.state.planned_frames = None
                self.state.planned_place_mm = None
                self.state.planned_frame_count = None
                self.state.planned_duration_s = None
                self.state.stage = "moving"
                self.state.moving = True
            self.state.log(
                "warning",
                "physical send accepted · speed scale 1.8",
            )
            self.state.publish("state", self.state.snapshot())
            threading.Thread(
                target=self._run_motion,
                args=(list(planned_frames),),
                daemon=True,
            ).start()
            return {
                "ok": True,
                "stage": "moving",
                "pickMm": pick,
                "placeMm": place,
                "send": True,
                "speedScale": 1.8,
            }
        except Exception:
            self.state.operation_lock.release()
            raise

    def _run_motion(self, frames: list[object]) -> None:
        relay = ControlOutput(self.state)
        try:
            with contextlib.redirect_stdout(relay):
                self.pipeline.run_control(
                    frames,
                    send=True,
                    robot_ip=self.pipeline.DEFAULT_ROBOT_IP,
                    robot_port=self.pipeline.DEFAULT_ROBOT_PORT,
                )
            relay.flush()
            with self.state.lock:
                self.state.stage = "complete"
            self.state.log(
                "success",
                "command sequence sent · hardware acknowledgement unavailable",
            )
        except Exception as error:
            with self.state.lock:
                self.state.stage = "fault"
            self.state.log("error", f"move failed · {error}")
        finally:
            with self.state.lock:
                self.state.moving = False
            self.state.publish("state", self.state.snapshot())
            self.state.operation_lock.release()


class GatewayHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MakerArmGateway/1.0"

        def log_message(self, format_string: str, *args: object) -> None:
            return

        def _cors(self) -> None:
            origin = self.headers.get("origin")
            if is_allowed_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Cache-Control", "no-store")

        def _json(
            self,
            payload: dict[str, object],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self) -> dict[str, object]:
            try:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError("request body must be valid JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json({"ok": True, **gateway.state.snapshot()})
                return
            if parsed.path == "/api/state":
                self._json({"ok": True, **gateway.state.snapshot()})
                return
            if parsed.path == "/api/events":
                self._events(parsed)
                return
            if parsed.path == "/api/video/stream":
                self._stream(parsed)
                return
            self._json(
                {"ok": False, "error": "not found"},
                HTTPStatus.NOT_FOUND,
            )

        def _events(self, parsed) -> None:
            try:
                query = parse_qs(parsed.query)
                last_id = int(
                    self.headers.get("last-event-id")
                    or query.get("after", ["0"])[0]
                )
                self.send_response(HTTPStatus.OK)
                self._cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    with gateway.state.condition:
                        pending = [
                            event
                            for event in gateway.state.events
                            if int(event["id"]) > last_id
                        ]
                        if not pending:
                            gateway.state.condition.wait(timeout=12)
                            pending = [
                                event
                                for event in gateway.state.events
                                if int(event["id"]) > last_id
                            ]
                    if not pending:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    for event in pending:
                        event_id = int(event["id"])
                        data = json.dumps(
                            event["data"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        packet = (
                            f"id: {event_id}\n"
                            f"event: {event['type']}\n"
                            f"data: {data}\n\n"
                        ).encode("utf-8")
                        self.wfile.write(packet)
                        self.wfile.flush()
                        last_id = event_id
            except (BrokenPipeError, ConnectionResetError):
                return

        def _stream(self, parsed) -> None:
            try:
                query = parse_qs(parsed.query)
                camera_index = gateway._camera_index(
                    {"cameraIndex": query.get("cameraIndex", ["auto"])[0]}
                )
                self.send_response(HTTPStatus.OK)
                self._cors()
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                while True:
                    jpeg = gateway.camera.jpeg(camera_index)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(1 / 24)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                gateway.state.log("error", f"video stream failed · {error}")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            routes = {
                "/api/detection-target": gateway.set_detection_target,
                "/api/calibration": gateway.calibrate,
                "/api/capture": gateway.capture_object,
                "/api/move": gateway.move,
            }
            action = routes.get(parsed.path)
            if action is None:
                self._json(
                    {"ok": False, "error": "not found"},
                    HTTPStatus.NOT_FOUND,
                )
                return
            try:
                result = action(self._body())
                status = (
                    HTTPStatus.ACCEPTED
                    if result.get("stage") == "moving"
                    else HTTPStatus.OK
                )
                self._json(result, status)
            except (ValueError, RuntimeError) as error:
                gateway.state.log("error", str(error))
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )
            except Exception as error:
                gateway.state.log("error", f"{parsed.path} failed · {error}")
                self._json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--camera-index",
        type=int,
        help="override the name-gated Continuity Camera OpenCV index",
        default=0
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gateway = Gateway(args.project_root.resolve())
    cameras = list_avfoundation_cameras()
    detected_camera = discover_continuity_camera(cameras)
    camera = (
        (
            args.camera_index,
            (
                detected_camera[1]
                if detected_camera is not None
                and args.camera_index == detected_camera[0]
                else f"OpenCV camera index {args.camera_index}"
            ),
        )
        if args.camera_index is not None
        else detected_camera
    )
    if camera is None:
        error = (
            "Apple Continuity Camera not found; connect the iPhone, "
            "enable Continuity Camera, then restart the gateway"
        )
        gateway.state.log("error", error)
        print(f"Camera unavailable: {error}", file=sys.stderr, flush=True)
    else:
        camera_index, camera_name = camera
        with gateway.state.lock:
            gateway.state.camera_name = camera_name
        try:
            gateway.camera.start(camera_index)
            gateway.state.log(
                "success",
                f"video source · {camera_name}",
            )
        except RuntimeError as error:
            gateway.state.log("error", str(error))
            print(f"Camera unavailable: {error}", file=sys.stderr, flush=True)
    server = GatewayHTTPServer(
        (args.host, args.port),
        make_handler(gateway),
    )
    print(
        f"Maker Arm gateway listening at http://{args.host}:{args.port}",
        flush=True,
    )
    print(f"Control source: {args.project_root / 'main.py'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Gateway stopped.", flush=True)
    finally:
        gateway.camera.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
