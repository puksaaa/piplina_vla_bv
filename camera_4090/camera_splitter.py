import argparse
import base64
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import zmq
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response


@dataclass
class EncodedFrame:
    name: str
    jpeg_base64: str
    timestamp: float
    width: int
    height: int


class LatestFrameStore:
    def __init__(self) -> None:
        self._frames: dict[str, EncodedFrame] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def set(self, frame: EncodedFrame) -> None:
        with self._lock:
            self._frames[frame.name] = frame
            self._errors.pop(frame.name, None)

    def set_error(self, name: str, message: str) -> None:
        with self._lock:
            self._errors[name] = message

    def get(self, name: str) -> EncodedFrame:
        with self._lock:
            frame = self._frames.get(name)
        if frame is None:
            raise KeyError(name)
        return frame

    def get_all(self) -> dict[str, EncodedFrame]:
        with self._lock:
            return dict(self._frames)

    def get_errors(self) -> dict[str, str]:
        with self._lock:
            return dict(self._errors)


class UsbCameraReader(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        device: int | str,
        store: LatestFrameStore,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
    ) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.device = device
        self.store = store
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.device)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.store.set_error(self.name, f"Could not open camera on device {self.device}")
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                self.store.set_error(self.name, "Could not read frame")
                time.sleep(0.05)
                continue

            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            height, width = frame.shape[:2]
            self.store.set(
                EncodedFrame(
                    name=self.name,
                    jpeg_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                    timestamp=time.time(),
                    width=width,
                    height=height,
                )
            )

        cap.release()


class SyntheticCameraReader(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        store: LatestFrameStore,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
    ) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.store = store
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        interval = 1 / max(self.fps, 1)
        frame_index = 0

        while not self._stop_event.is_set():
            frame = self._make_frame(frame_index)
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                height, width = frame.shape[:2]
                self.store.set(
                    EncodedFrame(
                        name=self.name,
                        jpeg_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                        timestamp=time.time(),
                        width=width,
                        height=height,
                    )
                )
            frame_index += 1
            time.sleep(interval)

    def _make_frame(self, frame_index: int) -> np.ndarray:
        width = max(self.width, 320)
        height = max(self.height, 240)
        frame = np.full((height, width, 3), (246, 251, 255), dtype=np.uint8)

        phase = frame_index % 120
        cx = int(width * 0.28 + phase * width * 0.003)
        cy = int(height * 0.55)
        box_x = int(width * 0.63)
        box_y = int(height * 0.42)

        cv2.rectangle(frame, (box_x, box_y), (box_x + 130, box_y + 90), (205, 235, 255), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + 130, box_y + 90), (92, 150, 190), 2)
        cv2.circle(frame, (cx, cy), 42, (255, 210, 235), -1)
        cv2.circle(frame, (cx, cy), 42, (126, 104, 214), 2)
        cv2.line(frame, (cx + 38, cy), (box_x, box_y + 45), (130, 180, 220), 2)

        cv2.putText(frame, f"{self.name} mock camera", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 72, 96), 2)
        cv2.putText(frame, "cup left of box", (24, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (91, 117, 148), 2)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (width - 145, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (91, 117, 148), 2)

        return frame


class ZmqPublisher(threading.Thread):
    def __init__(
        self,
        *,
        store: LatestFrameStore,
        bind: str,
        names: list[str],
        fps: int,
    ) -> None:
        super().__init__(daemon=True)
        self.store = store
        self.bind = bind
        self.names = names
        self.fps = fps
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.bind(self.bind)

        interval = 1 / max(self.fps, 1)
        while not self._stop_event.is_set():
            frames = self.store.get_all()
            available_names = [name for name in self.names if name in frames]
            if available_names:
                payload = {
                    "timestamps": {name: frames[name].timestamp for name in available_names},
                    "images": {name: frames[name].jpeg_base64 for name in available_names},
                    "missing": [name for name in self.names if name not in frames],
                }
                socket.send_json(payload)
            time.sleep(interval)

        socket.close(linger=0)


def frame_to_public_dict(frame: EncodedFrame) -> dict[str, Any]:
    return {
        "timestamp": frame.timestamp,
        "age_ms": int((time.time() - frame.timestamp) * 1000),
        "width": frame.width,
        "height": frame.height,
        "image": f"data:image/jpeg;base64,{frame.jpeg_base64}",
    }


def frame_to_jpeg_bytes(frame: EncodedFrame) -> bytes:
    return base64.b64decode(frame.jpeg_base64)


def build_app(store: LatestFrameStore, names: list[str]) -> FastAPI:
    app = FastAPI(title="USB Camera Splitter")

    @app.get("/health")
    def health() -> dict[str, Any]:
        frames = store.get_all()
        errors = store.get_errors()
        return {
            "status": "ok",
            "cameras": {
                name: {
                    "ready": name in frames,
                    "age_ms": int((time.time() - frames[name].timestamp) * 1000) if name in frames else None,
                    "error": errors.get(name),
                }
                for name in names
            },
        }

    @app.get("/snapshot/all")
    def snapshot_all() -> JSONResponse:
        frames = store.get_all()
        errors = store.get_errors()
        missing = [name for name in names if name not in frames]
        if not frames:
            details = [f"{name}: {errors.get(name, 'not ready')}" for name in missing]
            raise HTTPException(status_code=503, detail=f"Frames are not ready: {'; '.join(details)}")

        available_names = [name for name in names if name in frames]
        return JSONResponse(
            {
                "timestamp": time.time(),
                "complete": not missing,
                "missing": missing,
                "errors": {name: errors.get(name) for name in missing if errors.get(name)},
                "frames": {name: frame_to_public_dict(frames[name]) for name in available_names},
            }
        )

    @app.get("/snapshot/{camera_name}")
    def snapshot_one(camera_name: str) -> dict[str, Any]:
        try:
            frame = store.get(camera_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown or not ready camera: {camera_name}") from exc
        return frame_to_public_dict(frame)

    @app.get("/frame/{camera_name}.jpg")
    def frame_jpeg(camera_name: str) -> Response:
        try:
            frame = store.get(camera_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown or not ready camera: {camera_name}") from exc
        return Response(content=frame_to_jpeg_bytes(frame), media_type="image/jpeg")

    @app.get("/preview")
    def preview() -> HTMLResponse:
        image_tags = "\n".join(
            f'<section><h2>{name}</h2><img src="/frame/{name}.jpg?ts={int(time.time())}" alt="{name}" /></section>'
            for name in names
        )
        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="refresh" content="1" />
    <title>Camera Splitter Preview</title>
    <style>
      body {{ margin: 0; font-family: system-ui, sans-serif; background: #eef8ff; color: #263247; }}
      main {{ width: min(1200px, calc(100vw - 32px)); margin: 16px auto; }}
      .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
      section {{ padding: 14px; border-radius: 16px; background: rgba(255,255,255,.72); box-shadow: 0 12px 34px rgba(92,135,190,.18); }}
      h1, h2 {{ margin: 0 0 12px; }}
      img {{ width: 100%; height: auto; display: block; border-radius: 12px; background: #fff; }}
      code {{ background: rgba(255,255,255,.8); padding: 2px 6px; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <main>
      <h1>Camera Splitter Preview</h1>
      <p>Health: <code>/health</code>, snapshots: <code>/snapshot/all</code></p>
      <div class="grid">{image_tags}</div>
    </main>
  </body>
</html>
"""
        return HTMLResponse(html)

    return app


def parse_camera(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def probe_one_camera(index: int, width: int, height: int, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "opened": False,
        "read_ok": False,
        "width": None,
        "height": None,
        "error": None,
    }

    def worker() -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            result["opened"] = cap.isOpened()
            if result["opened"]:
                ok, frame = cap.read()
                result["read_ok"] = bool(ok and frame is not None)
                if ok and frame is not None:
                    actual_height, actual_width = frame.shape[:2]
                    result["width"] = actual_width
                    result["height"] = actual_height
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            if cap is not None:
                cap.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        result["error"] = f"probe timed out after {timeout}s"
    return result


def probe_cameras(max_index: int, width: int, height: int, timeout: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(max_index + 1):
        results.append(probe_one_camera(index, width, height, timeout))
    return results


def print_startup_summary(args: argparse.Namespace, names: list[str]) -> None:
    print("Camera splitter starting", flush=True)
    print(f"  mode: {'mock' if args.mock else 'usb'}", flush=True)
    print(f"  cameras: {names[0]}={args.front_device}, {names[1]}={args.wrist_device}", flush=True)
    print(f"  size/fps/jpeg: {args.width}x{args.height} @ {args.fps}fps, quality={args.jpeg_quality}", flush=True)
    print(f"  http: http://{args.http_host}:{args.http_port}", flush=True)
    print(f"  preview: http://{args.http_host}:{args.http_port}/preview", flush=True)
    if args.enable_zmq:
        print(f"  zmq: {args.zmq_bind} @ {args.zmq_fps}fps", flush=True)
    else:
        print("  zmq: disabled", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="USB camera splitter for LeRobot ZMQCamera and VLM snapshots.")
    parser.add_argument("--front-device", default=os.getenv("CAMERA_FRONT_DEVICE", "0"))
    parser.add_argument("--wrist-device", default=os.getenv("CAMERA_WRIST_DEVICE", "1"))
    parser.add_argument("--front-name", default=os.getenv("CAMERA_FRONT_NAME", "front"))
    parser.add_argument("--wrist-name", default=os.getenv("CAMERA_WRIST_NAME", "wrist"))
    parser.add_argument("--width", type=int, default=int(os.getenv("CAMERA_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("CAMERA_FPS", "15")))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.getenv("CAMERA_JPEG_QUALITY", "82")))
    parser.add_argument("--mock", action="store_true", default=os.getenv("CAMERA_MOCK", "0") == "1")
    parser.add_argument("--http-host", default=os.getenv("CAMERA_HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--http-port", type=int, default=int(os.getenv("CAMERA_HTTP_PORT", "8090")))
    parser.add_argument("--enable-zmq", action="store_true", default=os.getenv("CAMERA_ENABLE_ZMQ", "0") == "1")
    parser.add_argument("--zmq-bind", default=os.getenv("CAMERA_ZMQ_BIND", "tcp://0.0.0.0:5555"))
    parser.add_argument("--zmq-fps", type=int, default=int(os.getenv("CAMERA_ZMQ_FPS", "15")))
    parser.add_argument("--list-cameras", action="store_true", help="Probe local OpenCV camera indexes and exit.")
    parser.add_argument("--probe-count", type=int, default=int(os.getenv("CAMERA_PROBE_COUNT", "8")))
    parser.add_argument("--probe-timeout", type=float, default=float(os.getenv("CAMERA_PROBE_TIMEOUT", "2")))
    args = parser.parse_args()

    if args.list_cameras:
        for result in probe_cameras(args.probe_count - 1, args.width, args.height, args.probe_timeout):
            status = "ok" if result["read_ok"] else "unavailable"
            size = f'{result["width"]}x{result["height"]}' if result["width"] and result["height"] else "no frame"
            suffix = f' error={result["error"]}' if result["error"] else ""
            print(f'{result["index"]}: {status} ({size}){suffix}')
        return

    names = [args.front_name, args.wrist_name]
    print_startup_summary(args, names)

    store = LatestFrameStore()
    reader_class = SyntheticCameraReader if args.mock else UsbCameraReader

    if args.mock:
        readers = [
            reader_class(
                name=args.front_name,
                store=store,
                width=args.width,
                height=args.height,
                fps=args.fps,
                jpeg_quality=args.jpeg_quality,
            ),
            reader_class(
                name=args.wrist_name,
                store=store,
                width=args.width,
                height=args.height,
                fps=args.fps,
                jpeg_quality=args.jpeg_quality,
            ),
        ]
    else:
        readers = [
            reader_class(
                name=args.front_name,
                device=parse_camera(args.front_device),
                store=store,
                width=args.width,
                height=args.height,
                fps=args.fps,
                jpeg_quality=args.jpeg_quality,
            ),
            reader_class(
                name=args.wrist_name,
                device=parse_camera(args.wrist_device),
                store=store,
                width=args.width,
                height=args.height,
                fps=args.fps,
                jpeg_quality=args.jpeg_quality,
            ),
        ]

    for reader in readers:
        reader.start()

    if args.enable_zmq:
        ZmqPublisher(store=store, bind=args.zmq_bind, names=names, fps=args.zmq_fps).start()

    import uvicorn

    uvicorn.run(build_app(store, names), host=args.http_host, port=args.http_port)


if __name__ == "__main__":
    main()
