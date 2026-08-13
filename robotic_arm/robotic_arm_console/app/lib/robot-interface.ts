/**
 * Frontend integration contract.
 *
 * The current UI runs against a local simulator. When the Python gateway is
 * ready, keep this shape and replace the reserved URLs with the real service.
 */
export type JointName = "J1" | "J2" | "J3" | "J4" | "J6";

export interface RobotTelemetry {
  timestamp: number;
  phase: string;
  controlDeg: Record<JointName, number>;
  tcpMm: { x: number; y: number; z: number };
  mode: "idle" | "planning" | "executing" | "fault";
}

/**
 * A frame emitted by the future Python control gateway.
 * The UI intentionally does not interpolate or animate by itself: every model
 * and chart update is driven by one of these control frames.
 */
export interface ControlInputFrame {
  timestamp?: number;
  phase?: string;
  controlDeg: Record<JointName, number>;
}

export const CONTROL_INPUT_EVENT = "maker-arm:control-frame";

export function publishControlInput(frame: ControlInputFrame) {
  window.dispatchEvent(
    new CustomEvent<ControlInputFrame>(CONTROL_INPUT_EVENT, { detail: frame }),
  );
}

export interface PickAndPlaceRequest {
  pickMm: [number, number, number];
  placeMm: [number, number, number];
  liftMm?: number;
  rateHz?: number;
  speedScale?: number;
  dryRun: boolean;
}

export interface VideoStreamDescriptor {
  transport: "webrtc" | "hls" | "mjpeg";
  endpoint: string;
  signalingEndpoint?: string;
  cameraId: string;
}

export const ROBOT_ENDPOINTS = {
  state: "/api/state",
  events: "/api/events",
  detectionTarget: "/api/detection-target",
  calibration: "/api/calibration",
  capture: "/api/capture",
  move: "/api/move",
  video: "/api/video/stream",
} as const;

export function createRobotGateway(baseUrl = "") {
  return {
    async state() {
      return fetch(`${baseUrl}${ROBOT_ENDPOINTS.state}`);
    },
    async setDetectionTarget(target: string) {
      return fetch(`${baseUrl}${ROBOT_ENDPOINTS.detectionTarget}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ target }),
      });
    },
    async calibrate(cameraIndex: number | "auto") {
      return fetch(`${baseUrl}${ROBOT_ENDPOINTS.calibration}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cameraIndex }),
      });
    },
    async capture(cameraIndex: number | "auto", graspZMm: number) {
      return fetch(`${baseUrl}${ROBOT_ENDPOINTS.capture}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cameraIndex, graspZMm }),
      });
    },
    async move(request: {
      placeMm: [number, number, number];
      send: boolean;
    }) {
      return fetch(`${baseUrl}${ROBOT_ENDPOINTS.move}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
      });
    },
    eventSource() {
      return new EventSource(`${baseUrl}${ROBOT_ENDPOINTS.events}`);
    },
    videoUrl(cameraIndex: number | "auto") {
      return `${baseUrl}${ROBOT_ENDPOINTS.video}?cameraIndex=${cameraIndex}`;
    },
  };
}
