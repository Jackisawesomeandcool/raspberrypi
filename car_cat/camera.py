import json
import os
import socket
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import cv2

from prediction import predict, models
from detect_pi import load_model, detect_cat_crop, detect

# --- Tune these for quality vs bandwidth/latency ---
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30
JPEG_QUALITY = 80
PORT = 5000

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Shared frame buffer: intentionally stores only one, newest frame 
latest_frame = None
latest_annotated_frame = None
frame_lock = threading.Condition()

MODEL_PATH = "/home/jack/Desktop/Project/final_model.keras"


def capture_loop():
    """Capture MJPEG frames and replace the shared buffer with each newest one."""
    global latest_frame

    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--codec", "mjpeg",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FRAMERATE),
        "--quality", str(JPEG_QUALITY),
        "--inline",
        "--flush",
        "-o", "-",
        "-n",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    buffer = b""

    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
            break

        buffer += chunk

        start = buffer.find(JPEG_START)
        end = buffer.find(JPEG_END, start + 2) if start != -1 else -1

        while start != -1 and end != -1:
            frame = buffer[start:end + 2]

            # Replaces any unshown frame with the newest one.
            with frame_lock:
                latest_frame = frame
                frame_lock.notify_all()

            buffer = buffer[end + 2:]
            start = buffer.find(JPEG_START)
            end = buffer.find(JPEG_END, start + 2) if start != -1 else -1


def detection_loop():
    """generate yolo frame"""
    global latest_annotated_frame

    processed_frame = None

    while True:
        with frame_lock:
            frame_lock.wait_for(
                lambda: (
                    latest_frame is not None
                    and latest_frame is not processed_frame
                )
            )
            frame = latest_frame

        
        processed_frame = frame

        try:
            image = cv2.imdecode(
                np.frombuffer(frame, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                continue

            annotated_image = detect(
                image,
                model_yolo,
                size=320,
                threshold=0.15,
            )

            encoded_ok, encoded_frame = cv2.imencode(
                ".jpg",
                annotated_image,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if not encoded_ok:
                continue

            with frame_lock:
                latest_annotated_frame = encoded_frame.tobytes()
                frame_lock.notify_all()

        except Exception as error:
            print(f"YOLO display error: {error}")

INDEX_HTML = b"""<!doctype html>
<html>
<head>
  <title>Pi Camera Stream</title>
</head>
<body style="margin:0; background:#111; display:flex; flex-direction:column;
             justify-content:center; align-items:center; height:100vh;
             font-family:sans-serif;">

  <img id="camera" style="max-width:100%; max-height:85vh;">

  <button onclick="capture()"
          style="margin-top:15px; padding:12px 24px; font-size:16px; cursor:pointer;">
    Make Prediction
  </button>

  <div id="status" style="color:white; margin-top:8px;"></div>

  <script>
    const camera = document.getElementById("camera");

    // Fetch one frame at a time. This avoids a slow connection building
    // a backlog of old MJPEG frames.
    async function showLatestFrame() {
      try {
        const response = await fetch("/frame?t=" + Date.now(), {
          cache: "no-store"
        });

        if (!response.ok) {
          throw new Error("Camera unavailable");
        }

        const blob = await response.blob();
        const previousUrl = camera.src;

        camera.src = URL.createObjectURL(blob);

        if (previousUrl.startsWith("blob:")) {
          URL.revokeObjectURL(previousUrl);
        }
      } catch (err) {
        console.error("Frame error:", err);
      } finally {
        // Start the next request only after this one is complete.
        setTimeout(showLatestFrame, 67);
      }
    }

    async function capture() {
      const status = document.getElementById("status");
      status.innerText = "Running prediction...";

      try {
        const response = await fetch("/capture", { cache: "no-store" });
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Prediction failed");
        }

        status.innerText = "Photo saved! | Prediction: " + data.prediction + " | Confidence: " + (data.confidence * 100).toFixed(2) + "%";
      } catch (err) {
        status.innerText = "Error: " + err.message;
      }
    }

    showLatestFrame();
  </script>
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    disable_nagle_algorithm = True

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)

        elif self.path.startswith("/frame"):
            # Return only the current newest frame.
            with frame_lock:
                frame_lock.wait_for(lambda: latest_frame is not None)
                frame = latest_annotated_frame or latest_frame 

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(frame)

        elif self.path == "/capture":
            # Copy the newest frame at the instant the button is clicked.
            with frame_lock:
                frame = latest_frame

            if frame is None:
                response = json.dumps({"error": "No frame available yet"})
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(SNAPSHOT_DIR, f"frame_{timestamp}.jpg")

            with open(filename, "wb") as file:
                file.write(frame)

            image_path = os.path.abspath(filename)

            try:
                image = cv2.imread(image_path)
                image = detect_cat_crop(
                    image,
                    model_yolo,
                    size=320,
                    threshold=0.15
                )
                if image is None:
                    raise ValueError("No cat detected in the image.")
                
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )
                
                prediction, confidence = predict(image, model)
                response = {
                    "filename": image_path,
                    "prediction": str(prediction),
                    "confidence": float(confidence)
                }
                status_code = 200
            except Exception as error:
                response = {"error": str(error)}
                status_code = 500

            response_bytes = json.dumps(response).encode()

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # Keep this if your prediction module requires the model to be loaded here.
    model = models.load_model(MODEL_PATH)
    model_yolo = load_model("yolov8n_opencv.onnx")

    detection_thread = threading.Thread(
    target=detection_loop,
    daemon=True,
    )
    detection_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), StreamHandler)

    print(f"Serving on http://{get_local_ip()}:{PORT}  (Ctrl+C to quit)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
