"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  LoadingManager,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
} from "three";
import {
  CONTROL_INPUT_EVENT,
  createRobotGateway,
  publishControlInput,
  type ControlInputFrame,
} from "./lib/robot-interface";

type JointKey = "J1" | "J2" | "J3" | "J4" | "J6";
type JointState = Record<JointKey, number>;
type PipelineStage =
  | "idle"
  | "calibrated"
  | "captured"
  | "planned"
  | "moving"
  | "complete"
  | "fault";
type LogEntry = {
  timestamp: string;
  level: "info" | "success" | "warning" | "error";
  message: string;
};
type TargetMode = "control" | "detection_only";
type DetectionSummary = {
  label: string;
  confidence: number;
  box: [number, number, number, number];
};

const JOINTS: Array<{
  key: JointKey;
  label: string;
  detail: string;
  min: number;
  max: number;
  color: string;
}> = [
  { key: "J1", label: "BASE YAW", detail: "J1_base_yaw", min: 0, max: 180, color: "#77f7ff" },
  { key: "J2", label: "SHOULDER", detail: "J2_shoulder_pitch", min: 0, max: 180, color: "#9b7bff" },
  { key: "J3", label: "ELBOW", detail: "J3_elbow_pitch", min: 0, max: 180, color: "#ff5ab7" },
  { key: "J4", label: "WRIST", detail: "J4_wrist_pitch", min: 0, max: 180, color: "#8dff80" },
  { key: "J6", label: "GRIPPER", detail: "J6_gripper", min: 100, max: 170, color: "#ffc65c" },
];

const INITIAL_JOINTS: JointState = { J1: 90, J2: 60, J3: 90, J4: 90, J6: 150 };
const TASK_STEPS = ["Approach", "Pick", "Lift", "Transfer", "Place"];

function phaseToTaskStep(phase: string) {
  if (["START", "OPEN", "MOVE_PICK_ABOVE"].includes(phase)) return 0;
  if (["DESCEND_PICK", "CLOSE", "HOLD_GRASP"].includes(phase)) return 1;
  if (phase === "LIFT") return 2;
  if (phase === "TRANSFER") return 3;
  if (["LOWER_PLACE", "HOLD_RELEASE", "RETREAT"].includes(phase)) return 4;
  return -1;
}

function toUrdfAngles(control: JointState) {
  const deg = Math.PI / 180;
  return {
    J1_base_yaw: (90 - control.J1) * deg,
    J2_shoulder_pitch: (control.J2 - 60) * deg,
    J3_elbow_pitch: -(control.J3 - 90) * deg,
    J4_wrist_pitch: -(control.J4 - 90) * deg,
    J5_wrist_roll: 0,
    J6_gripper: -0.6 + ((control.J6 - 115) / 35) * 0.86,
  };
}

function RobotViewer({ joints }: { joints: JointState }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const robotRef = useRef<any>(null);
  const jointValues = useMemo(() => toUrdfAngles(joints), [joints]);
  const latestJointValuesRef = useRef(jointValues);
  const [assetError, setAssetError] = useState(false);

  useEffect(() => {
    latestJointValuesRef.current = jointValues;
  }, [jointValues]);

  useEffect(() => {
    let disposed = false;
    let frame = 0;
    let cleanup = () => {};

    async function mount() {
      const host = mountRef.current;
      if (!host) return;

      const THREE = await import("three");
      const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
      const { STLLoader } = await import("three/examples/jsm/loaders/STLLoader.js");
      const { default: URDFLoader } = await import("urdf-loader");
      if (disposed) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 50);
      camera.position.set(0.64, -0.85, 0.52);
      camera.up.set(0, 0, 1);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.35;
      host.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.07;
      controls.target.set(0.12, 0, 0.16);
      controls.minDistance = 0.35;
      controls.maxDistance = 1.8;

      scene.add(new THREE.HemisphereLight(0xb9faff, 0x07101f, 2.1));
      const keyLight = new THREE.DirectionalLight(0x8fefff, 4.5);
      keyLight.position.set(-0.4, -0.5, 1.2);
      scene.add(keyLight);
      const rimLight = new THREE.PointLight(0x8b5cff, 18, 4);
      rimLight.position.set(0.5, 0.4, 0.7);
      scene.add(rimLight);

      const grid = new THREE.GridHelper(1.6, 24, 0x236b83, 0x122331);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = -0.002;
      const gridMaterial = grid.material as Material;
      gridMaterial.opacity = 0.42;
      gridMaterial.transparent = true;
      scene.add(grid);

      const platform = new THREE.Mesh(
        new THREE.CylinderGeometry(0.115, 0.135, 0.018, 64),
        new THREE.MeshStandardMaterial({
          color: 0x102b38,
          emissive: 0x0b5866,
          emissiveIntensity: 0.7,
          metalness: 0.82,
          roughness: 0.28,
        }),
      );
      platform.rotation.x = Math.PI / 2;
      platform.position.z = 0.008;
      scene.add(platform);

      const manager = new THREE.LoadingManager();
      const loader = new URDFLoader(manager);
      loader.loadMeshCb = (
        path: string,
        loadManager: LoadingManager,
        urdfMaterial: Material,
        done: (mesh: Object3D, error?: Error) => void,
      ) => {
        const stl = new STLLoader(loadManager);
        stl.load(
          path,
          (geometry) => {
            geometry.computeVertexNormals();
            const isWrist = /Bilek|El|Parmak/i.test(path);
            const baseColor =
              "color" in urdfMaterial
                ? (urdfMaterial as MeshStandardMaterial).color
                : new THREE.Color(isWrist ? 0xf0ad6b : 0xa7eaff);
            const material = new THREE.MeshStandardMaterial({
              color: baseColor,
              emissive: isWrist ? 0x6b3214 : 0x0a6170,
              emissiveIntensity: isWrist ? 0.24 : 0.42,
              metalness: 0.34,
              roughness: 0.34,
            });
            const mesh = new THREE.Mesh(geometry, material);
            done(mesh);
          },
          undefined,
          (error) => {
            setAssetError(true);
            console.warn("STL viewer:", error);
          },
        );
      };
      loader.load(
        "/robot/robotic_arm_v0_6.urdf",
        (robot) => {
          if (disposed) return;
          robotRef.current = robot;
          setAssetError(false);
          robot.rotation.z = Math.PI;
          scene.add(robot);
          Object.entries(latestJointValuesRef.current).forEach(([name, value]) => {
            robot.joints[name]?.setJointValue(value);
          });
        },
        undefined,
        (error) => {
          setAssetError(true);
          console.warn("URDF viewer:", error);
        },
      );

      const resize = () => {
        const width = Math.max(host.clientWidth, 1);
        const height = Math.max(host.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      };
      const observer = new ResizeObserver(resize);
      observer.observe(host);
      resize();

      const animate = () => {
        controls.update();
        renderer.render(scene, camera);
        frame = requestAnimationFrame(animate);
      };
      animate();

      cleanup = () => {
        observer.disconnect();
        cancelAnimationFrame(frame);
        controls.dispose();
        renderer.dispose();
        renderer.domElement.remove();
        scene.traverse((node) => {
          const mesh = node as Mesh;
          mesh.geometry?.dispose?.();
          const material = mesh.material;
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material?.dispose?.();
        });
      };
    }

    mount();
    return () => {
      disposed = true;
      robotRef.current = null;
      cleanup();
    };
  }, []);

  useEffect(() => {
    const robot = robotRef.current;
    if (!robot) return;
    Object.entries(jointValues).forEach(([name, value]) => {
      robot.joints[name]?.setJointValue(value);
    });
  }, [jointValues]);

  return (
    <div className="robot-stage" ref={mountRef} aria-label="URDF v0.6 交互式三维机械臂模型">
      {assetError ? (
        <p className="robot-asset-notice">
          3D assets are not included in the public repository. See README.
        </p>
      ) : null}
    </div>
  );
}

function JointChart({ history }: { history: JointState[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const draw = () => {
      const ratio = Math.min(window.devicePixelRatio, 2);
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const left = 42;
      const right = width - 18;
      const top = 18;
      const bottom = height - 28;
      context.font = "12px var(--font-geist-mono)";
      context.fillStyle = "#64778b";
      context.strokeStyle = "rgba(112, 178, 203, .13)";
      context.lineWidth = 1;

      for (let line = 0; line <= 4; line += 1) {
        const y = top + ((bottom - top) * line) / 4;
        context.beginPath();
        context.moveTo(left, y);
        context.lineTo(right, y);
        context.stroke();
        context.fillText(`${180 - line * 45}°`, 4, y + 3);
      }
      for (let line = 0; line <= 8; line += 1) {
        const x = left + ((right - left) * line) / 8;
        context.beginPath();
        context.moveTo(x, top);
        context.lineTo(x, bottom);
        context.stroke();
      }

      JOINTS.forEach((joint) => {
        context.beginPath();
        history.forEach((sample, index) => {
          const x = left + ((right - left) * index) / Math.max(history.length - 1, 1);
          const normalized = Math.max(0, Math.min(1, sample[joint.key] / 180));
          const y = bottom - normalized * (bottom - top);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.strokeStyle = joint.color;
        context.lineWidth = 1.8;
        context.shadowBlur = 8;
        context.shadowColor = joint.color;
        context.stroke();
        context.shadowBlur = 0;
      });

      context.fillStyle = "#64778b";
      context.fillText("−12s", left, height - 7);
      context.fillText("NOW", right - 22, height - 7);
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [history]);

  return <canvas ref={canvasRef} className="telemetry-canvas" aria-label="最近十二秒的关节角度曲线" />;
}

export default function Home() {
  const [joints, setJoints] = useState<JointState>(INITIAL_JOINTS);
  const [history, setHistory] = useState<JointState[]>(() => Array.from({ length: 72 }, () => INITIAL_JOINTS));
  const [lastPhase, setLastPhase] = useState("STANDBY");
  const [hasControlInput, setHasControlInput] = useState(false);
  const [gatewayUrl, setGatewayUrl] = useState("http://127.0.0.1:8765");
  const [gatewayStatus, setGatewayStatus] = useState<"offline" | "online" | "secure-required">("offline");
  const [videoReady, setVideoReady] = useState(false);
  const [resolution, setResolution] = useState("—");
  const [cameraName, setCameraName] = useState("Apple Continuity Camera");
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>("idle");
  const [busyAction, setBusyAction] = useState<"target" | "calibration" | "capture" | "plan" | "move" | null>(null);
  const [targetDraft, setTargetDraft] = useState("soda can");
  const [activeTarget, setActiveTarget] = useState("soda can");
  const [targetMode, setTargetMode] = useState<TargetMode>("control");
  const [lastDetection, setLastDetection] = useState<DetectionSummary | null>(null);
  const [graspZ, setGraspZ] = useState("80");
  const [place, setPlace] = useState({ x: "245", y: "120", z: "0" });
  const [pick, setPick] = useState<[number, number, number] | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  const normalizedGatewayUrl = useMemo(
    () => gatewayUrl.trim().replace(/\/+$/, ""),
    [gatewayUrl],
  );
  const gateway = useMemo(
    () => createRobotGateway(normalizedGatewayUrl),
    [normalizedGatewayUrl],
  );

  const appendLog = useCallback((entry: LogEntry) => {
    setLogs((items) => [...items.slice(-199), entry]);
  }, []);

  const setStageFromGateway = useCallback((payload: any) => {
    if (
      ["idle", "calibrated", "captured", "planned", "moving", "complete", "fault"].includes(
        payload?.stage,
      )
    ) {
      setPipelineStage(payload.stage);
    }
    if (Array.isArray(payload?.pickMm) && payload.pickMm.length === 3) {
      setPick(payload.pickMm.map(Number) as [number, number, number]);
    } else if (payload?.pickMm === null) {
      setPick(null);
    }
    if (typeof payload?.detectionTarget === "string") {
      const nextTarget = payload.detectionTarget;
      setActiveTarget((previousTarget) => {
        setTargetDraft((draft) =>
          draft === previousTarget ? nextTarget : draft,
        );
        return nextTarget;
      });
    }
    if (["control", "detection_only"].includes(payload?.targetMode)) {
      setTargetMode(payload.targetMode);
    }
    if (
      typeof payload?.lastDetection?.label === "string" &&
      Number.isFinite(Number(payload.lastDetection.confidence)) &&
      Array.isArray(payload.lastDetection.box) &&
      payload.lastDetection.box.length === 4
    ) {
      setLastDetection({
        label: payload.lastDetection.label,
        confidence: Number(payload.lastDetection.confidence),
        box: payload.lastDetection.box.map(Number) as [
          number,
          number,
          number,
          number,
        ],
      });
    } else if (payload?.lastDetection === null) {
      setLastDetection(null);
    }
    if (
      Array.isArray(payload?.plannedPlaceMm) &&
      payload.plannedPlaceMm.length === 3
    ) {
      const [x, y, z] = payload.plannedPlaceMm.map(Number);
      if ([x, y, z].every(Number.isFinite)) {
        setPlace({ x: String(x), y: String(y), z: String(z) });
      }
    }
    if (typeof payload?.cameraName === "string" && payload.cameraName) {
      setCameraName(payload.cameraName);
    }
    if (Number.isInteger(payload?.cameraIndex)) {
      setVideoReady(true);
    }
    if (
      Array.isArray(payload?.cameraResolution) &&
      payload.cameraResolution.length === 2
    ) {
      setResolution(
        `${payload.cameraResolution[0]} × ${payload.cameraResolution[1]}`,
      );
    }
  }, []);

  useEffect(() => {
    const handleControlInput = (event: Event) => {
      const frame = (event as CustomEvent<ControlInputFrame>).detail;
      if (!frame?.controlDeg) return;

      const next = Object.fromEntries(
        JOINTS.map((joint) => [joint.key, Number(frame.controlDeg[joint.key])]),
      ) as JointState;
      const isValid = JOINTS.every(
        (joint) =>
          Number.isFinite(next[joint.key]) &&
          next[joint.key] >= joint.min &&
          next[joint.key] <= joint.max,
      );
      if (!isValid) return;

      setJoints(next);
      setHistory((items) => [...items.slice(-71), next]);
      setLastPhase(frame.phase ?? "CONTROL_UPDATE");
      setHasControlInput(true);
    };

    window.addEventListener(CONTROL_INPUT_EVENT, handleControlInput);
    return () => window.removeEventListener(CONTROL_INPUT_EVENT, handleControlInput);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "nearest" });
  }, [logs]);

  useEffect(() => {
    const host = window.location.hostname;
    if (
      window.location.protocol === "http:" &&
      host !== "localhost" &&
      host !== "127.0.0.1"
    ) {
      setGatewayUrl(`http://${host}:8765`);
    }
  }, []);

  useEffect(() => {
    if (!normalizedGatewayUrl) return;
    setVideoReady(false);

    if (
      typeof window !== "undefined" &&
      window.location.protocol === "https:" &&
      normalizedGatewayUrl.startsWith("http:")
    ) {
      setGatewayStatus("secure-required");
      appendLog({
        timestamp: new Date().toISOString(),
        level: "warning",
        message:
          "HTTPS page requires an HTTPS gateway URL; use the local UI or a secure tunnel",
      });
      return;
    }

    let disposed = false;
    gateway
      .state()
      .then(async (response) => {
        if (!response.ok) throw new Error(`gateway returned ${response.status}`);
        const payload = await response.json();
        if (disposed) return;
        setGatewayStatus("online");
        setStageFromGateway(payload);
        if (Array.isArray(payload.logs)) setLogs(payload.logs.slice(-200));
      })
      .catch(() => {
        if (!disposed) setGatewayStatus("offline");
      });

    const events = gateway.eventSource();
    events.onopen = () => {
      if (!disposed) setGatewayStatus("online");
    };
    events.onerror = () => {
      if (!disposed) setGatewayStatus("offline");
    };
    events.addEventListener("log", (event) => {
      if (!disposed) appendLog(JSON.parse((event as MessageEvent).data));
    });
    events.addEventListener("state", (event) => {
      if (!disposed) setStageFromGateway(JSON.parse((event as MessageEvent).data));
    });
    events.addEventListener("control", (event) => {
      if (disposed) return;
      const frame = JSON.parse((event as MessageEvent).data) as ControlInputFrame;
      publishControlInput(frame);
    });
    return () => {
      disposed = true;
      events.close();
    };
  }, [
    appendLog,
    gateway,
    normalizedGatewayUrl,
    setStageFromGateway,
  ]);

  const graspZNumber = Number(graspZ);
  const placeNumbers = [Number(place.x), Number(place.y), Number(place.z)] as [
    number,
    number,
    number,
  ];
  const graspValid = graspZ.trim() !== "" && Number.isFinite(graspZNumber);
  const placeValid = Object.values(place).every(
    (value) => value.trim() !== "" && Number.isFinite(Number(value)),
  );

  const normalizedTarget = targetDraft.trim().replace(/\s+/g, " ");
  const targetValid =
    normalizedTarget.length > 0 &&
    normalizedTarget.length <= 80 &&
    /^[\x20-\x7E]+$/.test(normalizedTarget) &&
    /[A-Za-z]/.test(normalizedTarget);

  const applyDetectionTarget = async () => {
    if (gatewayStatus !== "online") {
      appendLog({
        timestamp: new Date().toISOString(),
        level: "error",
        message: "Local Python gateway is not connected",
      });
      return;
    }
    if (
      busyAction !== null ||
      pipelineStage === "moving" ||
      normalizedTarget === activeTarget
    ) {
      return;
    }
    if (!targetValid) {
      appendLog({
        timestamp: new Date().toISOString(),
        level: "error",
        message: "Detection target must be an English category up to 80 characters",
      });
      return;
    }
    setBusyAction("target");
    try {
      const response = await gateway.setDetectionTarget(normalizedTarget);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error ?? "target update failed");
      }
      setTargetDraft(payload.detectionTarget);
      setStageFromGateway(payload);
    } catch (error) {
      appendLog({
        timestamp: new Date().toISOString(),
        level: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusyAction(null);
    }
  };

  const requestAction = async (
    action: "calibration" | "capture" | "plan" | "move",
  ) => {
    if (gatewayStatus !== "online") {
      appendLog({
        timestamp: new Date().toISOString(),
        level: "error",
        message: "Local Python gateway is not connected",
      });
      return;
    }
    if (
      action === "move" &&
      !window.confirm(
        "Send the planned motion to the physical robot at 1.8× speed?",
      )
    ) {
      return;
    }
    setBusyAction(action);
    appendLog({
      timestamp: new Date().toISOString(),
      level: "info",
      message: `${action} requested`,
    });
    try {
      let response: Response;
      if (action === "calibration") {
        response = await gateway.calibrate("auto");
      } else if (action === "capture") {
        response = await gateway.capture("auto", graspZNumber);
      } else {
        response = await gateway.move({
          placeMm: placeNumbers,
          send: action === "move",
        });
      }
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? `${action} failed`);
      setStageFromGateway(payload);
      if (action === "calibration") {
        setPick(null);
        if (Array.isArray(payload.resolution)) {
          setResolution(`${payload.resolution[0]} × ${payload.resolution[1]}`);
        }
      }
    } catch (error) {
      appendLog({
        timestamp: new Date().toISOString(),
        level: "error",
        message: error instanceof Error ? error.message : String(error),
      });
      if (action === "plan") setPipelineStage("captured");
      else if (action === "move") setPipelineStage("planned");
      else setPipelineStage("fault");
    } finally {
      setBusyAction(null);
    }
  };

  const activeTaskStep = hasControlInput ? phaseToTaskStep(lastPhase) : -1;
  const streamUrl =
    gatewayStatus === "secure-required" ? "" : gateway.videoUrl("auto");
  const stageLabel = pipelineStage.replaceAll("_", " ");

  return (
    <main className="console-shell">
      <section className="workspace">
        <section className="dashboard-grid">
          <article className="panel video-panel primary-video">
            <div className="panel-head video-head">
              <h2>Live camera</h2>
              <div className="stream-info-inline">
                <span><small>Resolution</small><b>{resolution}</b></span>
                <span><small>Detector</small><b>{activeTarget}</b></span>
                <span className={`status-chip ${videoReady ? "online" : "waiting"}`}>
                  <i /> {videoReady ? "Live" : "Waiting for source"}
                </span>
              </div>
            </div>
            <div className="video-placeholder">
              {streamUrl ? (
                <img
                  className={`camera-stream ${videoReady ? "ready" : ""}`}
                  src={streamUrl}
                  alt="iPhone camera live stream"
                  onLoad={(event) => {
                    setVideoReady(true);
                    const image = event.currentTarget;
                    if (image.naturalWidth && image.naturalHeight) {
                      setResolution(
                        `${image.naturalWidth} × ${image.naturalHeight}`,
                      );
                    }
                  }}
                  onError={() => setVideoReady(false)}
                />
              ) : null}
              {!videoReady ? (
                <>
                  <div className="video-noise" />
                  <div className="feed-message">
                    <div className="camera-symbol">●</div>
                    <b>Camera stream ready to connect</b>
                    <span>
                      {gatewayStatus === "secure-required"
                        ? "Secure gateway URL required"
                        : "Waiting for camera source"}
                    </span>
                  </div>
                </>
              ) : null}
              <div className="stream-label">
                CAMERA FEED
              </div>
              <div className="stream-timecode">{videoReady ? "LIVE" : "—"}</div>
            </div>
          </article>

          <article className="panel model-panel">
            <div className="panel-head">
              <div>
                <p className="panel-kicker">DIGITAL TWIN</p>
                <h2>3D arm model</h2>
              </div>
              <span className={`status-chip ${hasControlInput ? "online" : "waiting"}`}>
                <i /> {hasControlInput ? "Control received" : "Awaiting joint input"}
              </span>
            </div>
            <div className="model-viewport">
              <RobotViewer joints={joints} />
              <span className="model-badge"><i /> Three.js · {hasControlInput ? "control pose" : "holding pose"}</span>
            </div>
          </article>

          <article className="panel chart-panel">
            <div className="panel-head">
              <div>
                <p className="panel-kicker">TELEMETRY</p>
                <h2>Joint angles <span>{hasControlInput ? "Control input history" : "Waiting for control input"}</span></h2>
              </div>
              <div className="live-joint-readings">
                {JOINTS.map((joint) => (
                  <span key={joint.key}>
                    <i style={{ background: joint.color }} />
                    <b>{joint.key}</b>
                    <output>{joints[joint.key].toFixed(1)}°</output>
                  </span>
                ))}
              </div>
            </div>
            <JointChart history={history} />
          </article>

          <article className="panel mission-panel">
            <div className="panel-head">
              <div>
                <p className="panel-kicker">CONTROL TASK</p>
                <h2>Pick & place pipeline</h2>
              </div>
              <span className={`status-chip ${hasControlInput ? "online" : "waiting"}`}>
                <i /> {pipelineStage === "moving" ? "Sending" : stageLabel}
              </span>
            </div>

            <div className="phase-readout">
              <div className="phase-number">{hasControlInput ? String(Math.max(activeTaskStep + 1, 1)).padStart(2, "0") : "—"}</div>
              <div>
                <small>Control status</small>
                <b>{hasControlInput ? lastPhase.replaceAll("_", " ") : "WAITING FOR INPUT"}</b>
                <span>{hasControlInput ? "Pose updated from control function" : "Robot holds its current pose"}</span>
              </div>
            </div>

            <div className="trajectory">
              {TASK_STEPS.map((step, index) => (
                <div key={step} className={index < activeTaskStep ? "done" : index === activeTaskStep ? "current" : ""}>
                  <i /><span>{step}</span>
                </div>
              ))}
            </div>

            <div className="place-task-control">
              <div className="place-task-copy">
                <small>Place destination / mm</small>
                <b>{pick ? `Pick ready · ${pick.map((value) => value.toFixed(1)).join(", ")}` : "Enter X · Y · Z"}</b>
              </div>
              <div className="place-axis-inputs">
                {(["x", "y", "z"] as const).map((axis) => (
                  <label key={axis}>
                    <span>{axis.toUpperCase()}</span>
                    <input
                      type="text"
                      inputMode="decimal"
                      value={place[axis]}
                      aria-label={`Place ${axis.toUpperCase()} in millimetres`}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(event) => {
                        if (pipelineStage === "planned") {
                          setPipelineStage("captured");
                        }
                        setPlace((value) => ({
                          ...value,
                          [axis]: event.target.value,
                        }));
                      }}
                    />
                    <i>mm</i>
                  </label>
                ))}
              </div>
            </div>

            <p className="safety-note">No autonomous UI animation. Model and telemetry update only when a valid control frame arrives.</p>
          </article>
        </section>

        <article className="panel hardware-panel">
          <div className="panel-head hardware-head">
            <div>
              <p className="panel-kicker">LOCAL CONTROL</p>
              <h2>Vision → calibration → motion</h2>
            </div>
            <span className={`status-chip ${gatewayStatus === "online" ? "online" : "waiting"}`}>
              <i />
              {gatewayStatus === "online"
                ? "Python gateway online"
                : gatewayStatus === "secure-required"
                  ? "HTTPS gateway required"
                  : "Gateway offline"}
            </span>
          </div>

          <div className="gateway-config">
            <label>
              <span>Gateway URL</span>
              <input
                value={gatewayUrl}
                onChange={(event) => setGatewayUrl(event.target.value)}
                spellCheck={false}
              />
            </label>
            <span className="fixed-setting">
              <small>Video source</small>
              <b>{cameraName}</b>
            </span>
            <span className="fixed-setting">
              <small>Execution</small>
              <b>Plan → confirm send · 1.8×</b>
            </span>
          </div>

          <div className="target-config">
            <label>
              <span>Recognition target</span>
              <input
                value={targetDraft}
                maxLength={80}
                placeholder="English visual category"
                spellCheck={false}
                disabled={busyAction !== null || pipelineStage === "moving"}
                onChange={(event) => setTargetDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void applyDetectionTarget();
                  }
                }}
              />
              <button
                type="button"
                disabled={
                  !targetValid ||
                  normalizedTarget === activeTarget ||
                  gatewayStatus !== "online" ||
                  busyAction !== null ||
                  pipelineStage === "moving"
                }
                onClick={() => void applyDetectionTarget()}
              >
                {busyAction === "target" ? "Applying…" : "Apply target"}
              </button>
            </label>
            <span className="fixed-setting">
              <small>Target mode</small>
              <b>{targetMode === "control" ? "Control target" : "Detection only"}</b>
            </span>
            <span className="fixed-setting">
              <small>Last detection</small>
              <b>
                {lastDetection
                  ? `${lastDetection.confidence.toFixed(2)} · [${lastDetection.box.join(", ")}]`
                  : "—"}
              </b>
            </span>
          </div>

          <div className="hardware-workflow">
            <section className="action-stack">
              <button
                className="workflow-button"
                disabled={
                  gatewayStatus !== "online" ||
                  busyAction !== null ||
                  pipelineStage === "moving"
                }
                onClick={() => requestAction("calibration")}
              >
                <i>01</i>
                <span>
                  <b>
                    {busyAction === "calibration"
                      ? "Calibrating…"
                      : "Calibration"}
                  </b>
                  <small>Save reference + workspace plane</small>
                </span>
              </button>
              <button
                className="workflow-button"
                disabled={
                  gatewayStatus !== "online" ||
                  (targetMode === "control" &&
                    (!graspValid ||
                      !["calibrated", "captured", "planned", "complete"].includes(
                        pipelineStage,
                      ))) ||
                  pipelineStage === "moving" ||
                  busyAction !== null
                }
                onClick={() => requestAction("capture")}
              >
                <i>02</i>
                <span>
                  <b>{busyAction === "capture" ? "Capturing…" : "Capture"}</b>
                  <small>
                    {targetMode === "control"
                      ? "Locate object and resolve pick XYZ"
                      : "Return label, confidence and box only"}
                  </small>
                </span>
              </button>
              <button
                className="workflow-button move-button"
                disabled={
                  !placeValid ||
                  targetMode !== "control" ||
                  pipelineStage !== "captured" ||
                  busyAction !== null
                }
                onClick={() => requestAction("plan")}
              >
                <i>03</i>
                <span>
                  <b>{busyAction === "plan" ? "Planning…" : "Plan motion"}</b>
                  <small>Compute frames without physical send</small>
                </span>
              </button>
              <button
                className="workflow-button move-button"
                disabled={
                  pipelineStage !== "planned" ||
                  targetMode !== "control" ||
                  busyAction !== null
                }
                onClick={() => requestAction("move")}
              >
                <i>04</i>
                <span>
                  <b>{busyAction === "move" ? "Sending…" : "Send motion"}</b>
                  <small>Confirm and send the planned frames</small>
                </span>
              </button>
            </section>

            <section className="parameter-stack">
              <div className="parameter-group">
                <div className="parameter-title">
                  <span>Capture height / mm</span>
                  <small>Grasp Z defaults to 80</small>
                </div>
                <div className="grasp-input">
                  <label>
                    <span>Grasp Z</span>
                    <input
                      type="number"
                      value={graspZ}
                      placeholder="required"
                      onChange={(event) => setGraspZ(event.target.value)}
                    />
                  </label>
                </div>
              </div>
            </section>
          </div>

          <section className="log-panel">
            <div className="log-head">
              <span>LOG OUTPUT</span>
              <button onClick={() => setLogs([])}>Clear</button>
            </div>
            <div className="log-output" role="log" aria-live="polite">
              {logs.length ? (
                logs.map((entry, index) => (
                  <div
                    key={`${entry.timestamp}-${index}`}
                    className={`log-line ${entry.level}`}
                  >
                    <time>
                      {new Date(entry.timestamp).toLocaleTimeString([], {
                        hour12: false,
                      })}
                    </time>
                    <b>{entry.level.toUpperCase()}</b>
                    <span>{entry.message}</span>
                  </div>
                ))
              ) : (
                <div className="log-empty">No log entries</div>
              )}
              <div ref={logEndRef} />
            </div>
          </section>
        </article>
      </section>
    </main>
  );
}
