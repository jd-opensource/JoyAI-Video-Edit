from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
import uvicorn

from xvideo.serving.pe import DEFAULT_MODEL as DEFAULT_PE_MODEL
from xvideo.serving.joyomni_streaming import (
    JoyOmniRuntime,
    StreamingSettings,
    _module_device,
)

DEFAULT_DIT_CKPT = ""
DEFAULT_FACE_DETECTOR_ONNX = str(REPO_ROOT / "deps" / "checkpoints" / "face_detection_yunet_2023mar.onnx")
DEFAULT_PERSON_DETECTOR_ONNX = str(REPO_ROOT / "deps" / "checkpoints" / "yolov8n.onnx")
FACE_DETECTOR_DOWNSAMPLE = 1.5


class SessionGate:
    def __init__(self) -> None:
        self._order: list[int] = []
        self._events: dict[int, asyncio.Event] = {}
        self._next = 0

    def enqueue(self) -> int:
        self._next += 1
        ticket = self._next
        self._order.append(ticket)
        self._events[ticket] = asyncio.Event()
        return ticket

    def position(self, ticket: int) -> int:
        return self._order.index(ticket) if ticket in self._order else -1

    def is_holder(self, ticket: int) -> bool:
        return bool(self._order) and self._order[0] == ticket

    async def wait(self, ticket: int) -> None:
        ev = self._events.get(ticket)
        if ev is None:
            return
        await ev.wait()
        ev.clear()

    def release(self, ticket: int) -> None:
        self._events.pop(ticket, None)
        if ticket in self._order:
            self._order.remove(ticket)
        for ev in self._events.values():
            ev.set()


WS_SEND_TIMEOUT_S = 10.0

REF_IMAGE_DIR = REPO_ROOT / "rv2v_reference"
REF_IMAGE_FILES = {
    "hat": "4e481f7a-2443-4935-a841-af6113cc4236.png",
    "scarf": "5e178546-3ebf-40df-bb86-01613dd96c3b.png",
    "pink_tee": "1c182f2f-32cf-4825-904e-64c69aed2e31.png",
    "orange_glasses": "486b9561-e73d-45ca-bb2d-2a47998a0a73.png",
    "nailong": "nailong.png",
}

def _load_ref_images() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, fname in REF_IMAGE_FILES.items():
        path = REF_IMAGE_DIR / fname
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"#####[STREAM] ref image {name!r} unavailable ({path}): {exc!r}")
            continue
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        out[name] = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return out


_REF_IMAGES_CACHE: dict[str, str] | None = None

def _ref_images_cached() -> dict[str, str]:
    global _REF_IMAGES_CACHE
    if _REF_IMAGES_CACHE is None:
        _REF_IMAGES_CACHE = _load_ref_images()
    return _REF_IMAGES_CACHE


_INDEX_HTML_PATH = Path(__file__).resolve().parents[2] / "static" / "index.html"

def _load_index_html() -> str:
    return _INDEX_HTML_PATH.read_text(encoding="utf-8")

def _decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")

def _decode_ref_image(value: str | None) -> Image.Image | None:
    if not value:
        return None
    data = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return _decode_image(base64.b64decode(data))


def _snap_to_align(value: int, align: int) -> int:
    return max(align, (value + align // 2) // align * align)


class _H264Stream:
    def __init__(self, quality: int) -> None:
        self._lock = threading.Lock()
        self._enc = None
        self._size: tuple[int, int] | None = None
        self._crf = self.crf_for_quality(quality)
        self._want_crf = self._crf
        self._want_reset = False

    @staticmethod
    def crf_for_quality(quality: int) -> int:
        return max(15, min(30, round((35 - int(quality) / 4) / 5) * 5))

    def set_quality(self, quality: int) -> None:
        self._want_crf = self.crf_for_quality(quality)

    def reset(self) -> None:
        self._want_reset = True

    def encode(self, frames: list) -> list[tuple[bytes, bool]]:
        with self._lock:
            return self._encode_locked(frames)

    def _encode_locked(self, frames: list) -> list[tuple[bytes, bool]]:
        import av
        import cv2

        h, w = frames[0].shape[:2]
        want_crf = int(self._want_crf)
        if (
            self._enc is None
            or self._size != (w, h)
            or self._want_reset
            or want_crf != self._crf
        ):
            self._want_reset = False
            self._crf = want_crf
            self._enc = None
            enc = av.CodecContext.create("libx264", "w")
            enc.width = w
            enc.height = h
            enc.pix_fmt = "yuv420p"
            enc.options = {
                "preset": "veryfast",
                "tune": "zerolatency",
                "crf": str(self._crf),
                "g": "16",
                "x264-params": "scenecut=0:repeat-headers=1",
            }
            enc.open()
            self._enc = enc
            self._size = (w, h)
        out: list[tuple[bytes, bool]] = []
        for rgb in frames:
            i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
            vframe = av.VideoFrame.from_ndarray(i420, format="yuv420p")
            for pkt in self._enc.encode(vframe):
                out.append((bytes(pkt), bool(pkt.is_keyframe)))
        return out


class _H264Ingest:
    def __init__(self) -> None:
        import av

        self._dec = av.CodecContext.create("h264", "r")

    def decode_one(self, au: bytes) -> Image.Image | None:
        import av

        try:
            frames = self._dec.decode(av.Packet(au))
        except av.error.InvalidDataError:
            return None
        if not frames:
            return None
        return Image.fromarray(frames[-1].to_ndarray(format="rgb24"), mode="RGB")


_FACE_DETECTOR: dict[str, Any] = {}

def _get_face_detector(onnx_path: str, score_thresh: float):
    if not onnx_path:
        return None
    if onnx_path in _FACE_DETECTOR:
        return _FACE_DETECTOR[onnx_path] or None
    det = False
    try:
        if not os.path.exists(onnx_path):
            print(f"#####[FACE-GATE] YuNet weight not found: {onnx_path} -> gate DISABLED", flush=True)
        else:
            import cv2
            det = cv2.FaceDetectorYN.create(onnx_path, "", (320, 320), float(score_thresh), 0.3, 5000)
            print(f"#####[FACE-GATE] YuNet loaded: {onnx_path} (score>={score_thresh})", flush=True)
    except Exception as exc:
        print(f"#####[FACE-GATE] failed to load YuNet ({exc!r}) -> gate DISABLED", flush=True)
        det = False
    _FACE_DETECTOR[onnx_path] = det
    return det or None

def _check_face_gate(image: Image.Image, *, onnx_path: str,
                     score_thresh: float, min_below_ratio: float,
                     count_min_ratio: float = 0.45):
    det = _get_face_detector(onnx_path, score_thresh)
    if det is None:
        return (None, None, 0)
    import cv2
    import numpy as np
    rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    h, w = bgr.shape[:2]
    if FACE_DETECTOR_DOWNSAMPLE > 1.0:
        _s = 1.0 / FACE_DETECTOR_DOWNSAMPLE
        bgr = cv2.resize(bgr, (max(2, int(round(w * _s))), max(2, int(round(h * _s)))), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    det.setScoreThreshold(float(score_thresh))
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    frame_min = float(min(w, h))

    best = None
    best_area = -1.0
    conf_shorts = []
    if faces is not None:
        for f in faces:
            if float(f[-1]) < float(score_thresh):
                continue
            fw, fh = float(f[2]), float(f[3])
            conf_shorts.append(min(fw, fh))
            if fw * fh > best_area:
                best_area = fw * fh
                best = (float(f[0]), float(f[1]), fw, fh)
    if best is None:
        return ("no_face", None, 0)
    fx, fy, fw, fh = best

    _best_short = min(fw, fh)
    _count_thr = max(0.05 * frame_min, float(count_min_ratio) * _best_short)
    n_faces = sum(1 for s in conf_shorts if s >= _count_thr)
    cx = (fx + fw / 2.0) / float(w)
    cy = (fy + fh / 2.0) / float(h)
    room_below = 1.0 - (fy + fh) / float(h)
    if room_below < float(min_below_ratio):
        return ("too_close", None, n_faces)
    return (None, (cx, cy, min(fw, fh) / frame_min), n_faces)


def _detect_gate_faces(image: Image.Image, *, onnx_path: str, score_thresh: float):
    det = _get_face_detector(onnx_path, score_thresh)
    if det is None:
        return None, 0.0, 0.0
    import cv2
    import numpy as np
    rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    h, w = bgr.shape[:2]
    if FACE_DETECTOR_DOWNSAMPLE > 1.0:
        _s = 1.0 / FACE_DETECTOR_DOWNSAMPLE
        bgr = cv2.resize(bgr, (max(2, int(round(w * _s))), max(2, int(round(h * _s)))), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    det.setScoreThreshold(float(score_thresh))
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    if faces is not None:
        for f in faces:
            out.append((float(f[0]), float(f[1]), float(f[2]), float(f[3])))
    return out, float(w), float(h)


def _face_present_from(faces, w, h, *, min_ratio: float = 0.0, edge_margin: float = 0.0) -> bool:
    if faces is None:
        return True
    _min_side = float(min_ratio) * float(min(w, h))
    _mx = float(edge_margin) * float(w)
    _my = float(edge_margin) * float(h)
    for fx, fy, fw, fh in faces:
        if _min_side > 0.0 and min(fw, fh) < _min_side:
            continue
        if edge_margin > 0.0 and (fx < _mx or fy < _my or fx + fw > w - _mx or fy + fh > h - _my):
            continue
        return True
    return False


def _count_faces_from(faces, w, h, *, count_min_ratio: float) -> int:
    if not faces:
        return 0
    frame_min = float(min(w, h))
    best = None
    best_area = -1.0
    shorts = []
    for fx, fy, fw, fh in faces:
        shorts.append(min(fw, fh))
        if fw * fh > best_area:
            best_area = fw * fh
            best = (fw, fh)
    thr = max(0.05 * frame_min, float(count_min_ratio) * min(best))
    return sum(1 for sh in shorts if sh >= thr)


_PERSON_NET: dict[str, Any] = {}

def _get_person_net(onnx_path: str):
    if not onnx_path:
        return None
    if onnx_path in _PERSON_NET:
        return _PERSON_NET[onnx_path] or None
    net = False
    try:
        if not os.path.exists(onnx_path):
            print(f"#####[PERSON-GATE] YOLO weight not found: {onnx_path} -> presence passthrough DISABLED", flush=True)
        else:
            import cv2
            net = cv2.dnn.readNetFromONNX(onnx_path)
            print(f"#####[PERSON-GATE] YOLO loaded: {onnx_path}", flush=True)
    except Exception as exc:
        print(f"#####[PERSON-GATE] failed to load YOLO ({exc!r}) -> DISABLED", flush=True)
        net = False
    _PERSON_NET[onnx_path] = net
    return net or None

def _person_present(image: Image.Image, *, onnx_path: str, conf: float) -> bool:
    net = _get_person_net(onnx_path)
    if net is None:
        return True
    import numpy as np
    import cv2
    rgb = np.ascontiguousarray(np.asarray(image if image.mode == "RGB" else image.convert("RGB")))
    blob = cv2.dnn.blobFromImage(rgb, 1.0 / 255.0, (320, 320), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    return float(out[0, 4, :].max()) >= float(conf)

def _enhance_prompt_sync(
    *,
    raw_prompt: str,
    ref_image: Image.Image | None,
    pe_frame: Image.Image | None,
    pe_model: str | None,
) -> dict[str, Any]:
    started = time.time()
    task_type = "v2v"
    enhanced_prompt = raw_prompt
    error = None
    model = pe_model or DEFAULT_PE_MODEL
    try:
        from xvideo.serving.pe import PromptEnhancer

        enhancer = PromptEnhancer(model=pe_model)
        model = enhancer.model or model
        enhanced = enhancer(
            task_type,
            raw_prompt,
            video=[pe_frame] if pe_frame is not None else None,
            images=[ref_image] if ref_image is not None else None,
        )
        if isinstance(enhanced, str) and enhanced.strip():
            enhanced_prompt = enhanced.strip()
    except Exception as exc:
        error = repr(exc)
    report = {
        "enabled": True,
        "task_type": task_type,
        "model": model,
        "raw_prompt": raw_prompt,
        "enhanced_prompt": enhanced_prompt,
        "elapsed_s": time.time() - started,
        "fallback": enhanced_prompt == raw_prompt,
        "cached": False,
    }
    if error is not None:
        report["error"] = error
    return report

class _SegmentedRecorder:
    def __init__(
        self,
        *,
        prefix: Path,
        codec: str,
        bitrate: int,
        segment_seconds: int,
        queue_max: int = 64,
        lossless: bool = False,
    ) -> None:
        self._prefix = prefix
        self._codec = codec
        self._bitrate = int(bitrate)
        self._lossless = bool(lossless)
        self._segment_ms = max(1, int(segment_seconds)) * 1000
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, queue_max))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rec-{prefix.name[:12]}", daemon=True)
        self._width: int | None = None
        self._height: int | None = None

        self.frames_written = 0
        self.frames_dropped_recording = 0
        self.segments = 0
        self.last_error: str | None = None
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def submit(self, item: Any, t_capture_ms: float) -> None:
        if not self._started or self._stop.is_set():
            return
        try:
            self._q.put_nowait((item, float(t_capture_ms)))
        except queue.Full:
            self.frames_dropped_recording += 1

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        if self._stop.is_set():
            self._thread.join(timeout=timeout)
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _to_image(self, item: Any) -> "Image.Image | None":
        if isinstance(item, Image.Image):
            return item
        if isinstance(item, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(item))
                img.load()
                return img.convert("RGB") if img.mode != "RGB" else img
            except Exception:
                return None

        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return Image.fromarray(np.ascontiguousarray(item), mode="RGB")
        except Exception:
            return None
        return None

    def _open_segment(self):
        import av
        path = self._prefix.parent / f"{self._prefix.name}_{self.segments:04d}.mp4"
        output = av.open(str(path), mode="w")
        stream = output.add_stream(self._codec)
        stream.width = int(self._width)
        stream.height = int(self._height)
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        if self._lossless:
            stream.codec_context.options = {"crf": "8", "preset": "medium"}
        else:
            stream.bit_rate = self._bitrate
            stream.codec_context.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }
        self.segments += 1
        return output, stream

    def _close_segment(self, output, stream) -> None:
        if output is None or stream is None:
            return
        try:
            for packet in stream.encode():
                output.mux(packet)
        except Exception:
            pass
        try:
            output.close()
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import av
        except Exception as exc:
            self.last_error = f"pyav import failed: {exc!r}"
            return
        output = stream = None
        time_base = Fraction(1, 1000)
        seg_t0 = 0.0
        last_pts = -1
        try:
            while True:
                try:
                    got = self._q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if got is None:
                    break
                item, t_capture_ms = got
                image = self._to_image(item)
                if image is None:
                    continue
                if self._width is None:
                    self._width, self._height = int(image.width), int(image.height)
                if output is None:
                    output, stream = self._open_segment()
                    seg_t0 = t_capture_ms
                    last_pts = -1
                pts = round(t_capture_ms - seg_t0)
                if pts <= last_pts:
                    pts = last_pts + 1
                try:
                    frame = av.VideoFrame.from_image(image).reformat(format="yuv420p")
                    frame.pts = pts
                    frame.time_base = time_base
                    for packet in stream.encode(frame):
                        output.mux(packet)
                    self.frames_written += 1
                    last_pts = pts
                except Exception:
                    pass
                if pts >= self._segment_ms:
                    self._close_segment(output, stream)
                    output = stream = None
        finally:
            self._close_segment(output, stream)

def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}.")
    return parsed

def _preload_gate_detectors(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        det = _get_face_detector(args.face_detector_onnx, float(args.face_gate_score))
        if det is not None:
            gw = max(2, int(round(args.width / FACE_DETECTOR_DOWNSAMPLE)))
            gh = max(2, int(round(args.height / FACE_DETECTOR_DOWNSAMPLE)))
            det.setInputSize((gw, gh))
            det.detect(np.zeros((gh, gw, 3), dtype=np.uint8))
        net = _get_person_net(args.person_detector_onnx)
        if net is not None:
            import cv2
            blob = cv2.dnn.blobFromImage(np.zeros((320, 320, 3), dtype=np.uint8),
                                         1.0 / 255.0, (320, 320), swapRB=False, crop=False)
            net.setInput(blob)
            net.forward()
        print("#####[GATE] detectors preloaded", flush=True)
    except Exception as exc:
        print(f"#####[GATE] detector preload skipped: {exc!r}", flush=True)

def create_app(args: argparse.Namespace) -> FastAPI:
    def get_runtime() -> JoyOmniRuntime:
        if app.state.runtime is not None:
            return app.state.runtime
        with app.state.runtime_lock:
            if app.state.runtime is None:
                _preload_gate_detectors(args)
                app.state.runtime = JoyOmniRuntime.load(
                    args.dit_ckpt,
                    vae_ckpt=args.vae_ckpt,
                    text_encoder_ckpt=args.text_encoder_ckpt,
                    device=args.device,
                    vae_device=args.vae_device,
                    vae_encode_device=args.vae_encode_device,
                    vae_decode_device=args.vae_decode_device,
                    vae_pseudo_device=args.vae_pseudo_device,
                    postprocess_device=args.postprocess_device,
                    seed=args.seed,
                    warmup_height=args.height,
                    warmup_width=args.width,
                )
        return app.state.runtime

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if args.preload:
            get_runtime()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.state.runtime = None
    app.state.runtime_lock = threading.Lock()
    app.state.inference_lock = threading.Lock()
    app.state.active_session = None
    app.state.ws_debug = {}

    app.state.session_gate = SessionGate()

    app.state.last_recording_dir = None

    @app.get("/")
    def index() -> HTMLResponse:
        server_defaults = {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "kv_reset_frames": args.kv_reset_frames,
            "output_quality": args.output_quality,
            "online_gate": args.online_gate,
            "static_diff_thresh": args.static_diff_thresh,
            "freeze_kv_on_static": args.freeze_kv_on_static,
            "profile_timings": args.profile_timings,
            "use_pe": args.use_pe,
            "pe_available": bool(os.environ.get("OPENAI_API_KEY")),
            "max_temporal_ids": args.max_temporal_ids,
            "record_enabled": args.record_dir is not None,
        }
        html = _load_index_html().replace("__SERVER_DEFAULTS__", json.dumps(server_defaults))
        return HTMLResponse(html)

    @app.get("/ref-images")
    def ref_images() -> JSONResponse:
        return JSONResponse(_ref_images_cached())

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "runtime_loaded": app.state.runtime is not None,
                "dit_ckpt": args.dit_ckpt,
                "device": str(app.state.runtime.device) if app.state.runtime is not None else args.device,
                "vae_device": str(_module_device(app.state.runtime.pipeline.vae)) if app.state.runtime is not None else args.vae_device,
                "vae_encode_device": (
                    str(_module_device(app.state.runtime.pipeline.vae))
                    if app.state.runtime is not None
                    else (args.vae_encode_device or args.vae_device)
                ),
                "vae_decode_device": (
                    str(_module_device(app.state.runtime.decode_vae))
                    if app.state.runtime is not None
                    else (args.vae_decode_device or args.vae_device)
                ),
                "vae_pseudo_device": (
                    str(_module_device(app.state.runtime.pseudo_encode_vae))
                    if app.state.runtime is not None
                    else (args.vae_pseudo_device or args.vae_decode_device or args.vae_device)
                ),
                "postprocess_device": (
                    str(app.state.runtime.postprocess_device)
                    if app.state.runtime is not None
                    else (args.postprocess_device or args.vae_pseudo_device or args.vae_decode_device or args.vae_device)
                ),
                "use_pe": args.use_pe,
                "pe_model": args.pe_model or DEFAULT_PE_MODEL,
                "kv_reset_frames": args.kv_reset_frames,
                "max_temporal_ids": args.max_temporal_ids,
                "freeze_kv_on_static": args.freeze_kv_on_static,
                "static_diff_thresh": args.static_diff_thresh,
            }
        )

    @app.get("/debug")
    def debug() -> JSONResponse:
        session = getattr(app.state, "active_session", None)
        session_debug = session.debug_snapshot() if session is not None else None
        return JSONResponse(
            {
                "ok": True,
                "ts": time.time(),
                "runtime_loaded": app.state.runtime is not None,
                "ws": getattr(app.state, "ws_debug", {}),
                "session": session_debug,
            }
        )

    @app.post("/load")
    def load() -> JSONResponse:
        started = time.time()
        get_runtime()
        return JSONResponse({"ok": True, "elapsed": time.time() - started})

    @app.get("/download_last")
    async def download_last() -> Response:
        if args.record_dir is None:
            return JSONResponse({"error": "Recording is not enabled (--record-dir is unset)."}, status_code=404)
        rec_dir = getattr(app.state, "last_recording_dir", None)
        if not rec_dir:
            return JSONResponse({"error": "No downloadable result yet. Send an edit first."}, status_code=404)
        base = Path(rec_dir)
        segments = sorted(base.glob("output_*.mp4"))
        if not segments:
            return JSONResponse({"error": "Recording file has not been generated yet. Try again later."}, status_code=404)
        download_name = f"joyomni_{base.name}.mp4"

        if len(segments) == 1:
            return FileResponse(
                str(segments[0]), media_type="video/mp4", filename=download_name
            )
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            return JSONResponse({"error": f"ffmpeg unavailable: {exc!r}"}, status_code=500)
        list_path = out_path = None
        try:
            fd_list, list_path = tempfile.mkstemp(suffix=".txt", prefix="rv2v_cat_")
            with os.fdopen(fd_list, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"file '{seg.as_posix()}'\n")
            fd_out, out_path = tempfile.mkstemp(suffix=".mp4", prefix="rv2v_dl_")
            os.close(fd_out)
            enc_args = ["-c", "copy"]
            proc = await asyncio.create_subprocess_exec(
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                *enc_args, "-movflags", "+faststart", out_path,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                tail = (stderr or b"").decode("utf-8", "replace")[-800:]
                return JSONResponse({"error": f"ffmpeg failed: {tail}"}, status_code=500)
            with open(out_path, "rb") as f:
                data = f.read()
            return Response(
                content=data, media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
            )
        except Exception as exc:
            return JSONResponse({"error": f"download encode error: {exc!r}"}, status_code=500)
        finally:
            for p in (list_path, out_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime = get_runtime()
        session = None
        ticket = None
        frames_in = 0
        frames_out = 0
        session_prompt = args.prompt
        session_settings: StreamingSettings | None = None
        ref_image: Image.Image | None = None
        face_gate_pending = False
        pe_defer = False
        pe_task: asyncio.Task | None = None
        session_max_inflight = max(0, int(args.max_inflight_chunks or 0))

        gate_state = {"count": 0, "cx": None, "cy": None, "absent": 0,
                      "absent_hold": False, "present": 0, "person_check_i": 0, "person_last": True,
                      "subject_count": None, "cand": None, "cand_n": 0, "recount": False}
        kv_reset_frames = max(0, int(args.kv_reset_frames or 0))

        output_quality = 60 if args.output_quality == "auto" else int(args.output_quality)
        output_codec = "mjpeg"
        h264_stream: _H264Stream | None = None
        input_codec = "mjpeg"
        h264_ingest: _H264Ingest | None = None
        input_sniffed = False
        max_temporal_ids = args.max_temporal_ids
        freeze_kv_on_static = args.freeze_kv_on_static
        static_diff_thresh = args.static_diff_thresh
        frames_since_session_reset = 0
        reset_count = 0
        next_frame_meta: dict[str, Any] | None = None
        send_lock = asyncio.Lock()
        stop_output_pump = asyncio.Event()
        output_task: asyncio.Task[None] | None = None

        flow = {"recv": None, "rtt": None, "at": 0.0, "dropped": 0, "congested": False,
                "base": 0, "has_ack": False, "probe": 0, "clamped": False,
                "skew_min": None, "skew_at": 0.0, "up_ms": 0.0,
                "consec": 0}


        rec_input: _SegmentedRecorder | None = None
        rec_output: _SegmentedRecorder | None = None
        rec_seq = 0
        rec_base: Path | None = None
        lossless_mode = False
        ws_debug: dict[str, Any] = {
            "connected_at": time.time(),
            "frames_in": 0,
            "frames_out": 0,
            "output_bytes": 0,
            "chunk_results_sent": 0,
            "last_receive_at": None,
            "last_send_at": None,
            "last_message_type": None,
            "kv_reset_frames": kv_reset_frames,
            "max_temporal_ids": max_temporal_ids,
            "freeze_kv_on_static": freeze_kv_on_static,
            "static_diff_thresh": static_diff_thresh,
            "kv_reset_count": reset_count,
            "frames_since_session_reset": frames_since_session_reset,
            "send_state": "idle",
        }

        async def _ws_send_json(payload: dict[str, Any]) -> None:
            try:
                await asyncio.wait_for(websocket.send_json(payload), timeout=WS_SEND_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise WebSocketDisconnect()

        async def _ws_send_bytes(data: bytes) -> None:
            try:
                await asyncio.wait_for(websocket.send_bytes(data), timeout=WS_SEND_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise WebSocketDisconnect()

        async def _send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                ws_debug["send_state"] = f"json:{payload.get('type')}"
                await _ws_send_json(payload)
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

        async def _send_encoded_frames(
            encoded_frames: list[bytes],
            source_metas: list[dict[str, Any]],
            profile: dict[str, Any],
            server_elapsed: float,
        ) -> int:
            nonlocal frames_out
            if not encoded_frames:
                return 0

            if gate_state.get("absent_hold"):
                return 0
            count = len(encoded_frames)
            if not source_metas:
                source_metas = [{} for _ in range(count)]

            _wire_chunk = h264_stream is not None or (
                encoded_frames and isinstance(encoded_frames[0], (bytes, bytearray))
            )
            _fps = float(getattr(session_settings, "fps", None) or args.fps or 24.0)
            _outstanding = None
            if _wire_chunk and flow["recv"] is not None and (time.time() - flow["at"]) < 5.0:
                _outstanding = max(0, (frames_out - int(flow.get("base") or 0)) - int(flow["recv"]))
            if _outstanding is not None:
                _report_age = time.time() - flow["at"]
                _slack = 0 if flow.get("has_ack") else int(_report_age * _fps)
                _up_credit = int(min(float(flow.get("up_ms") or 0.0), 3000.0) * _fps / 1000.0)
                _adj = max(0, _outstanding - _slack - _up_credit)
                _hi = max(2 * count + 4, int(_fps * 0.8))
                _lo = max(count, int(_fps * 0.35))
                _drop = False
                if flow["congested"]:
                    if _adj <= _lo:
                        flow["congested"] = False
                        flow["probe"] = 8
                        flow["consec"] = 0
                    else:
                        _drop = True
                elif _adj >= _hi:
                    flow["congested"] = True
                    _drop = True
                    if h264_stream is not None and not flow.get("clamped"):
                        flow["clamped"] = True
                        h264_stream.set_quality(min(int(output_quality), 20))
                if not _drop and flow.get("probe", 0) > 0:
                    if _adj > max(2, count // 2):
                        _drop = True
                    else:
                        flow["probe"] -= 1
                        if flow["probe"] <= 0 and flow.get("clamped") and h264_stream is not None:
                            flow["clamped"] = False
                            h264_stream.set_quality(int(output_quality))
                if _drop:
                    if flow.get("consec", 0) >= 3:
                        _drop = False
                        flow["consec"] = 0
                    else:
                        flow["consec"] = flow.get("consec", 0) + 1
                else:
                    flow["consec"] = 0
                ws_debug["flow_outstanding"] = _outstanding
                ws_debug["flow_adj"] = _adj
                ws_debug["flow_up_ms"] = round(float(flow.get("up_ms") or 0.0), 1)
                ws_debug["flow_congested"] = bool(flow["congested"])
                if _drop:
                    flow["dropped"] += 1
                    ws_debug["chunks_dropped_congestion"] = flow["dropped"]
                    if h264_stream is not None:
                        h264_stream.reset()
                    _rec_o = rec_output
                    if _rec_o is not None and encoded_frames:
                        for _i, _enc in enumerate(encoded_frames):
                            _m = source_metas[min(_i, len(source_metas) - 1)]
                            _rec_o.submit(_enc, _m.get("t_capture_ms"))
                            ws_debug["rec_out_written"] = _rec_o.frames_written
                            ws_debug["rec_out_dropped"] = _rec_o.frames_dropped_recording
                    async with send_lock:
                        await _ws_send_json({
                            "type": "flow_drop",
                            "count": count,
                            "outstanding": _outstanding,
                            "dropped_total": flow["dropped"],
                        })
                    return 0

            _prof = bool(profile.get("profile_timings"))
            _send_t0 = time.perf_counter() if _prof else 0.0

            if _prof:
                _seen: set[int] = set()
                _dec_ms = 0.0
                for _m in source_metas:
                    if id(_m) in _seen:
                        continue
                    _seen.add(id(_m))
                    _dec_ms += float(_m.get("jpeg_decode_ms", 0.0) or 0.0)
                if _dec_ms > 0.0:
                    profile["jpeg_decode_ms"] = _dec_ms

            h264_packets: list[tuple[bytes, bool]] | None = None
            if h264_stream is not None and encoded_frames and not isinstance(encoded_frames[0], bytes):
                _enc_t0 = time.perf_counter()
                h264_packets = await asyncio.to_thread(h264_stream.encode, encoded_frames)
                if _prof:
                    profile["h264_encode_s"] = time.perf_counter() - _enc_t0

            async with send_lock:
                ws_debug["send_state"] = f"chunk_start:{profile.get('chunk_idx')}"
                await _ws_send_json(
                    {
                        "type": "chunk_start",
                        "count": count,
                        "elapsed": server_elapsed,
                        "frames_in": frames_in,
                        "source_seq_start": source_metas[0].get("seq"),
                        "source_seq_end": source_metas[-1].get("seq"),
                    }
                )
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

            for idx, encoded in enumerate(encoded_frames):
                source_meta = source_metas[min(idx, len(source_metas) - 1)]
                if h264_packets is not None:
                    wire, is_key = h264_packets[idx]
                elif not isinstance(encoded, bytes):
                    frames_out += 1
                    ws_debug["frames_out"] = frames_out
                    _rec_o = rec_output
                    if _rec_o is not None:
                        _rec_o.submit(encoded, source_meta.get("t_capture_ms"))
                        ws_debug["rec_out_written"] = _rec_o.frames_written
                        ws_debug["rec_out_dropped"] = _rec_o.frames_dropped_recording
                    continue
                else:
                    wire, is_key = encoded, False
                async with send_lock:
                    ws_debug["send_state"] = f"chunk_frame:{profile.get('chunk_idx')}:{idx}"
                    await _ws_send_json(
                        {
                            "type": "output_frame",
                            "index": idx,
                            "count": count,
                            "source_seq": source_meta.get("seq"),
                            "t_capture_ms": source_meta.get("t_capture_ms"),
                            "server_elapsed": server_elapsed,
                            "profile": profile,

                            "key": is_key,
                        }
                    )
                    await _ws_send_bytes(wire)
                    frames_out += 1
                    ws_debug["frames_out"] = frames_out
                    ws_debug["output_bytes"] = int(ws_debug.get("output_bytes", 0)) + len(wire)
                    ws_debug["last_send_at"] = time.time()
                    ws_debug["send_state"] = "idle"

                    _rec_o = rec_output
                    if _rec_o is not None:
                        _rec_o.submit(encoded, source_meta.get("t_capture_ms"))
                        ws_debug["rec_out_written"] = _rec_o.frames_written
                        ws_debug["rec_out_dropped"] = _rec_o.frames_dropped_recording

            next_chunk_needs = (
                session.frames_per_next_chunk - len(session.pending_frames)
                if session is not None
                else 0
            )

            if _prof:
                profile["ws_send_s"] = time.perf_counter() - _send_t0

                _recv_ms = source_metas[-1].get("t_server_recv_ms")
                if _recv_ms:
                    profile["server_residence_s"] = max(
                        0.0, time.time() - float(_recv_ms) / 1000.0
                    )

            _chunk_done_msg = {
                "type": "chunk_done",
                "count": count,
                "frames_in": frames_in,
                "frames_out": frames_out,
                "next_chunk_needs": next_chunk_needs,
            }
            if _prof:
                _chunk_done_msg["ws_send_s"] = profile.get("ws_send_s")
                _chunk_done_msg["server_residence_s"] = profile.get("server_residence_s")
                _chunk_done_msg["chunk_idx"] = profile.get("chunk_idx")
            async with send_lock:
                ws_debug["send_state"] = f"chunk_done:{profile.get('chunk_idx')}"
                await _ws_send_json(_chunk_done_msg)
                ws_debug["chunk_results_sent"] = int(ws_debug.get("chunk_results_sent", 0)) + 1
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

            if _prof:
                print(
                    f"#####[SERVER] chunk={profile.get('chunk_idx')} count={count} "
                    f"jpeg_dec={float(profile.get('jpeg_decode_ms', 0.0)):.0f}ms "
                    f"jpeg_enc={float(profile.get('jpeg_encode_s', 0.0)) * 1000:.0f}ms "
                    f"ws_send={float(profile.get('ws_send_s', 0.0)) * 1000:.0f}ms "
                    f"residence={float(profile.get('server_residence_s', 0.0)) * 1000:.0f}ms",
                    flush=True,
                )
            return count

        async def _send_chunk_result(
            result,
            *,
            fallback_meta: dict[str, Any] | None = None,
            fallback_elapsed: float = 0.0,
        ) -> int:
            if not result.jpegs:
                return 0
            jpegs = result.jpegs
            if result.source_metas:
                source_metas = result.source_metas
            elif fallback_meta is not None:
                source_metas = [fallback_meta] * len(jpegs)
            else:
                source_metas = [{} for _ in jpegs]
            if result.valid_count is not None:
                jpegs = jpegs[: result.valid_count]
                source_metas = source_metas[: result.valid_count]
            server_elapsed = float(result.elapsed or fallback_elapsed)
            return await _send_encoded_frames(
                jpegs, source_metas, result.profile, server_elapsed,
            )

        async def _output_pump(session_ref) -> None:
            while not stop_output_pump.is_set():
                try:
                    result = await asyncio.to_thread(session_ref.wait_async_result, 0.05)
                except Exception as exc:
                    await _send_json({"type": "error", "message": repr(exc)})
                    break
                if result is None:
                    continue
                if result.jpegs:
                    jpegs = result.jpegs
                    source_metas = result.source_metas or [{} for _ in jpegs]
                    if result.valid_count is not None:
                        jpegs = jpegs[: result.valid_count]
                        source_metas = source_metas[: result.valid_count]
                    await _send_encoded_frames(
                        jpegs, source_metas, result.profile, float(result.elapsed or 0.0)
                    )

        def _create_session():
            if session_settings is None:
                raise RuntimeError("streaming settings are not initialized")
            return runtime.create_v2v_session(
                prompt=session_prompt,
                settings=session_settings,
                ref_image=ref_image,
            )

        def _session_health_error(session_ref) -> str | None:
            if session_ref is None:
                return None
            try:
                snapshot = session_ref.debug_snapshot()
            except Exception:
                return None

            workers = snapshot.get("workers") or {}
            errored = []
            dead = []
            for name, info in workers.items():
                state = info.get("state") or {}
                state_name = state.get("state")
                if state_name == "error":
                    errored.append(f"{name}@chunk={state.get('chunk_idx')}")
                if info.get("alive") is False:
                    dead.append(f"{name}:{state_name or 'unknown'}")
            if errored:
                return "streaming pipeline worker error: " + ", ".join(errored)

            queues = snapshot.get("queues") or {}
            queue_maxsize = snapshot.get("queue_maxsize") or {}
            encode_depth = queues.get("encode")
            encode_max = queue_maxsize.get("encode")
            encode_full = (
                isinstance(encode_depth, int) and
                isinstance(encode_max, int) and
                encode_max > 0 and
                encode_depth >= encode_max
            )
            if encode_full and dead:
                return (
                    f"streaming pipeline stuck: encode queue full "
                    f"({encode_depth}/{encode_max}) and workers not alive: " +
                    ", ".join(dead)
                )
            return None

        def _close_session_sync(session_ref) -> None:
            with app.state.inference_lock:
                session_ref.close()

        async def _close_session_safely(session_ref, reason: str) -> None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(_close_session_sync, session_ref),
                    timeout=max(0.1, float(args.session_close_timeout_s)),
                )
            except asyncio.TimeoutError:
                ws_debug["close_timeout"] = reason
                print(
                    f"#####[WS-GUARD] session close timed out after "
                    f"{args.session_close_timeout_s:.1f}s reason={reason}",
                    flush=True,
                )
            except Exception as exc:
                ws_debug["close_error"] = repr(exc)
                print(f"#####[WS-GUARD] session close failed reason={reason}: {exc!r}", flush=True)

        async def _stop_output_task() -> None:
            nonlocal output_task
            stop_output_pump.set()
            if output_task is not None:
                try:
                    await asyncio.wait_for(output_task, timeout=1.0)
                except asyncio.TimeoutError:
                    output_task.cancel()
                except Exception:
                    pass
            output_task = None

        def _start_recorders() -> None:
            nonlocal rec_input, rec_output, rec_seq, rec_base
            if args.record_dir is None:
                return
            _stop_recorders()
            try:
                rec_seq += 1
                base = Path(args.record_dir) / f"{int(time.time())}_{rec_seq}"
                base.mkdir(parents=True, exist_ok=True)
                common = dict(
                    codec=str(args.record_codec),
                    bitrate=int(args.record_bitrate),
                    segment_seconds=int(args.record_segment_seconds),
                )
                rec_input = _SegmentedRecorder(prefix=base / "input", **common)
                rec_output = _SegmentedRecorder(prefix=base / "output", lossless=lossless_mode, **common)
                rec_input.start()
                rec_output.start()
                rec_base = base
                ws_debug["rec_dir"] = str(base)
                print(f"#####[REC] recording -> {base}", flush=True)

                _write_prompt_sidecar()
            except Exception as exc:
                ws_debug["rec_error"] = repr(exc)
                print(f"#####[REC] start failed: {exc!r} -> recording OFF", flush=True)
                rec_input = None
                rec_output = None
                rec_base = None

        def _stop_recorders() -> None:
            nonlocal rec_input, rec_output, rec_base
            for _rec in (rec_input, rec_output):
                if _rec is not None:
                    try:
                        _rec.stop()
                    except Exception as exc:
                        ws_debug["rec_error"] = f"stop: {exc!r}"
            rec_input = None
            rec_output = None
            rec_base = None

        def _write_prompt_sidecar() -> None:
            if rec_base is None:
                return
            try:
                report = pe_report if isinstance(pe_report, dict) else None
                enhanced = report.get("enhanced_prompt") if report else None
                doc = {"raw_prompt": raw_session_prompt}

                if enhanced and enhanced != raw_session_prompt:
                    doc["enhanced_prompt"] = enhanced
                (rec_base / "prompts.json").write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                ws_debug["rec_error"] = f"prompts: {exc!r}"

        async def _cancel_pe() -> None:
            nonlocal pe_task
            if pe_task is not None:
                pe_task.cancel()
                try:
                    await pe_task
                except BaseException:
                    pass
                pe_task = None

        def _start_output_task(session_ref) -> None:
            nonlocal output_task
            if session_settings is not None:
                stop_output_pump.clear()
                output_task = asyncio.create_task(_output_pump(session_ref))


        async def _reset_session(reason: str) -> None:
            nonlocal session, frames_since_session_reset, reset_count
            if session is None:
                return
            print(f"#####[STREAM] session reset ({reason})", flush=True)

            await _stop_output_task()
            await _close_session_safely(session, "kv_reset")
            reset_count += 1
            session = _create_session()
            app.state.active_session = session
            frames_since_session_reset = 0
            ws_debug["kv_reset_count"] = reset_count
            ws_debug["frames_since_session_reset"] = frames_since_session_reset
            await _send_json(
                {
                    "type": "session_reset",
                    "reason": reason,
                    "reset_count": reset_count,
                    "frames_in": frames_in,
                    "kv_reset_frames": kv_reset_frames,
                    "max_temporal_ids": max_temporal_ids,
                    "freeze_kv_on_static": freeze_kv_on_static,
                    "static_diff_thresh": static_diff_thresh,
                    "frames_per_next_chunk": session.frames_per_next_chunk,
                }
            )
            _start_output_task(session)

        try:
            gate = app.state.session_gate
            ticket = gate.enqueue()
            while not gate.is_holder(ticket):
                pos = gate.position(ticket)
                await _send_json({"type": "queue_position", "position": pos, "ahead": pos})
                try:
                    await asyncio.wait_for(gate.wait(ticket), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            await _send_json({"type": "session_granted"})

            app.state.ws_debug = ws_debug

            HOLDER_IDLE_TIMEOUT_S = 10.0
            last_activity = time.monotonic()
            last_frames_out = frames_out
            while True:
                if frames_out != last_frames_out or pe_task is not None:
                    last_frames_out = frames_out
                    last_activity = time.monotonic()
                if time.monotonic() - last_activity >= HOLDER_IDLE_TIMEOUT_S:
                    try:
                        await _send_json(
                            {
                                "type": "session_timeout",
                                "message": f"released after {HOLDER_IDLE_TIMEOUT_S:.0f}s idle; reconnect to re-queue",
                            }
                        )
                    except Exception:
                        pass
                    break
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if "text" in message and message["text"] is not None:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")
                    if msg_type == "start":
                        print(f"#####[RESTART] 'start' received (session {'live' if session is not None else 'none'})", flush=True)
                        last_activity = time.monotonic()
                        await _cancel_pe()
                        if session is not None:
                            print("#####[RESTART] tearing down prior session: stop_output_task", flush=True)
                            await _stop_output_task()
                            print("#####[RESTART] close_session", flush=True)
                            await _close_session_safely(session, "restart")
                            print("#####[RESTART] prior session closed OK", flush=True)
                            if getattr(app.state, "active_session", None) is session:
                                app.state.active_session = None
                            session = None

                        raw_session_prompt = str(payload.get("prompt", args.prompt))
                        session_prompt = raw_session_prompt
                        try:
                            ref_image = _decode_ref_image(payload.get("ref_image"))
                        except Exception as exc:
                            await _send_json({"type": "error", "message": f"failed to decode ref image: {exc!r}"})
                            continue
                        kv_reset_frames = max(0, int(payload.get("kv_reset_frames", args.kv_reset_frames)))
                        output_quality = max(1, min(100, int(payload.get("output_quality", output_quality))))
                        lossless_mode = str(payload.get("source", "")) == "file"
                        _up_allow = args.uplink_codec == "auto" and not lossless_mode
                        _dn_allow = args.downlink_codec == "auto" and not lossless_mode
                        output_codec = "h264" if payload.get("output_codec", "h264") == "h264" and _dn_allow else "mjpeg"
                        h264_stream = _H264Stream(output_quality) if output_codec == "h264" else None
                        input_codec = "h264" if payload.get("input_codec", "h264") == "h264" and _up_allow else "mjpeg"
                        h264_ingest = _H264Ingest() if input_codec == "h264" else None
                        input_sniffed = False
                        if app.state.runtime is not None:
                            app.state.runtime.output_quality = output_quality
                            app.state.runtime.lossless_output = lossless_mode
                        try:
                            max_temporal_ids = _optional_positive_int(
                                payload.get("max_temporal_ids", args.max_temporal_ids),
                                name="max_temporal_ids",
                            )
                        except Exception as exc:
                            await _send_json({"type": "error", "message": f"invalid max temporal ids: {exc!r}"})
                            continue

                        freeze_kv_on_static = bool(
                            payload.get("freeze_kv_on_static", args.freeze_kv_on_static)
                        )
                        static_diff_thresh = float(
                            payload.get("static_diff_thresh", args.static_diff_thresh)
                        )
                        session_max_inflight = max(0, int(
                            payload.get("max_inflight_chunks", args.max_inflight_chunks) or 0
                        ))
                        use_pe = bool(payload.get("use_pe", args.use_pe)) and bool(os.environ.get("OPENAI_API_KEY"))

                        entry_gate = bool(payload.get("gate_enabled", True))
                        face_gate_pending = entry_gate
                        flow["recv"] = None
                        flow["at"] = 0.0
                        flow["congested"] = False
                        flow["probe"] = 0
                        flow["clamped"] = False
                        flow["consec"] = 0
                        flow["base"] = frames_out

                        gate_on = bool(args.online_gate) and entry_gate

                        fg_score = float(payload.get("fg_score", args.face_gate_score))
                        fg_min_below = float(payload.get("fg_min_below_ratio", args.face_gate_min_below_ratio))
                        fg_stable = max(1, int(payload.get("fg_stable_frames", args.face_gate_stable_frames)))
                        fg_absent = max(1, int(args.presence_absent_frames))
                        fg_return = max(1, int(args.presence_return_frames))
                        _gate_fps = float(payload.get("fps") or args.fps or 24.0)
                        _fscale = _gate_fps / 24.0
                        fg_stable = max(1, int(round(fg_stable * _fscale)))
                        fg_absent = max(1, int(round(fg_absent * _fscale)))
                        fg_return = max(1, int(round(fg_return * _fscale)))
                        count_change_frames = max(1, int(round(int(args.person_count_change_frames) * _fscale)))
                        body_flip_frames = max(1, int(round(int(args.person_body_flip_frames) * _fscale)))
                        person_stride = max(1, int(round(int(args.person_check_stride) * _fscale)))
                        gate_move_eps = float(args.face_gate_move_eps) / _fscale
                        gate_state["count"] = 0
                        gate_state["cx"] = None
                        gate_state["cy"] = None
                        gate_state["absent"] = 0
                        gate_state["absent_hold"] = False
                        gate_state["present"] = 0
                        gate_state["person_check_i"] = 0
                        gate_state["person_last"] = True
                        gate_state["face_last"] = True
                        gate_state["hold_reason"] = "no_person"
                        gate_state["body_miss"] = 0
                        gate_state["subject_count"] = None
                        gate_state["cand"] = None
                        gate_state["cand_n"] = 0
                        gate_state["recount"] = False
                        gate_state["pe_anchor"] = None
                        gate_state["settle_ax"] = None
                        gate_state["settle_ay"] = None
                        if face_gate_pending:
                            print(f"#####[FACE-GATE] armed (score={fg_score}, min_below={fg_min_below}, stable={fg_stable}, absent={fg_absent})", flush=True)
                        pe_report = None
                        pe_defer = False
                        if use_pe:
                            cached_enhanced_prompt = str(payload.get("enhanced_prompt") or "").strip()
                            if cached_enhanced_prompt:
                                session_prompt = cached_enhanced_prompt
                                pe_report = {
                                    "enabled": True,
                                    "task_type": "rv2v" if ref_image is not None else "v2v",
                                    "model": args.pe_model or DEFAULT_PE_MODEL,
                                    "raw_prompt": raw_session_prompt,
                                    "enhanced_prompt": session_prompt,
                                    "elapsed_s": 0.0,
                                    "fallback": session_prompt == raw_session_prompt,
                                    "cached": True,
                                }
                                ws_debug["pe_report"] = pe_report
                                await _send_json({"type": "prompt_enhanced", **pe_report})
                            else:
                                pe_defer = True
                        _align = runtime.pipeline.vae.stem.stride * 8
                        _sh = _snap_to_align(int(payload.get("height", args.height)), _align)
                        _sw = _snap_to_align(int(payload.get("width", args.width)), _align)
                        session_settings = StreamingSettings(
                            height=_sh,
                            width=_sw,
                            num_inference_steps=int(payload.get("num_inference_steps", args.num_inference_steps)),
                            seed=int(payload.get("seed", args.seed)),
                            max_temporal_ids=max_temporal_ids,
                            freeze_kv_on_static=freeze_kv_on_static,
                            static_diff_thresh=static_diff_thresh,
                            profile_timings=bool(payload.get("profile_timings", args.profile_timings)),
                            output_codec=output_codec,
                        )

                        print("#####[RESTART] creating new session", flush=True)
                        session = _create_session()
                        print("#####[RESTART] new session created; sending 'started'", flush=True)
                        app.state.active_session = session
                        frames_since_session_reset = 0
                        reset_count = 0
                        ws_debug["kv_reset_frames"] = kv_reset_frames
                        ws_debug["use_pe"] = use_pe
                        ws_debug["pe_model"] = args.pe_model or DEFAULT_PE_MODEL
                        ws_debug["max_temporal_ids"] = max_temporal_ids
                        ws_debug["freeze_kv_on_static"] = freeze_kv_on_static
                        ws_debug["static_diff_thresh"] = static_diff_thresh
                        ws_debug["kv_reset_count"] = reset_count
                        ws_debug["frames_since_session_reset"] = frames_since_session_reset
                        ws_debug["has_ref_image"] = ref_image is not None
                        ws_debug["last_message_type"] = "start"
                        await _send_json(
                            {
                                "type": "started",
                                "frames_per_next_chunk": session.frames_per_next_chunk,
                                "height": session_settings.height,
                                "width": session_settings.width,
                                "output_codec": output_codec,
                                "input_codec": input_codec,
                                "ref_image": ref_image is not None,
                                "kv_reset_frames": kv_reset_frames,
                                "use_pe": use_pe,
                                "pe_model": args.pe_model or DEFAULT_PE_MODEL,
                                "max_temporal_ids": max_temporal_ids,
                                "freeze_kv_on_static": freeze_kv_on_static,
                                "static_diff_thresh": static_diff_thresh,
                            }
                        )
                        _start_recorders()
                        _start_output_task(session)
                        if not pe_defer:
                            def _prebake(_p=session_prompt, _s=session_settings, _r=ref_image):
                                with app.state.inference_lock:
                                    runtime.prebake_graph(_p, settings=_s, ref_image=_r)
                            asyncio.create_task(asyncio.to_thread(_prebake))
                    elif msg_type == "stop":
                        ws_debug["last_message_type"] = "stop"

                        await _cancel_pe()
                        await _stop_output_task()
                        await asyncio.to_thread(_stop_recorders)
                        break
                    elif msg_type == "finalize_recording":
                        ws_debug["last_message_type"] = "finalize_recording"
                        if args.record_dir is None:
                            await _send_json({"type": "recording_finalized", "ok": False,
                                              "message": "Recording is not enabled (--record-dir is unset)."})
                            continue

                        if session is not None:
                            await asyncio.to_thread(session.flush_pending)
                            _flush_deadline = time.monotonic() + 20.0
                            while frames_out < frames_in and time.monotonic() < _flush_deadline:
                                await asyncio.sleep(0.05)
                        last_activity = time.monotonic()
                        await _stop_output_task()
                        finalized = rec_base
                        await asyncio.to_thread(_stop_recorders)
                        if finalized is not None:
                            app.state.last_recording_dir = str(finalized)
                        await _send_json({
                            "type": "recording_finalized",
                            "ok": finalized is not None,
                            "message": None if finalized is not None
                            else "No downloadable result yet. Send an edit first.",
                        })
                        continue
                    elif msg_type == "frame_meta":
                        ws_debug["last_message_type"] = "frame_meta"
                        next_frame_meta = {
                            "seq": int(payload.get("seq", frames_in + 1)),
                            "t_capture_ms": float(payload.get("t_capture_ms", time.time() * 1000.0)),
                        }
                    elif msg_type == "ack":
                        _recv = payload.get("recv")
                        if _recv is not None:
                            try:
                                flow["recv"] = int(_recv)
                                flow["at"] = time.time()
                                flow["has_ack"] = True
                            except (TypeError, ValueError):
                                pass
                        continue
                    elif msg_type == "ping":
                        _recv = payload.get("recv")
                        if _recv is not None:
                            try:
                                flow["recv"] = int(_recv)
                                flow["at"] = time.time()
                            except (TypeError, ValueError):
                                pass
                        _rtt = payload.get("rtt")
                        if _rtt is not None:
                            try:
                                flow["rtt"] = float(_rtt)
                            except (TypeError, ValueError):
                                pass
                        _up_ms = None
                        _t = payload.get("t")
                        if _t is not None:
                            try:
                                _now = time.time()
                                _skew = _now * 1000.0 - float(_t)
                                _smin = flow.get("skew_min")
                                if _smin is not None:
                                    _smin += 0.5 * max(0.0, _now - float(flow.get("skew_at") or _now))
                                if _smin is None or _skew < _smin:
                                    _smin = _skew
                                flow["skew_min"] = _smin
                                flow["skew_at"] = _now
                                _up_ms = max(0.0, _skew - _smin)
                                flow["up_ms"] = _up_ms
                            except (TypeError, ValueError):
                                pass
                        await _send_json({
                            "type": "pong",
                            "t": payload.get("t"),
                            "ts": time.time() * 1000.0,
                            "up_ms": None if _up_ms is None else round(_up_ms, 1),
                        })
                        continue
                    elif msg_type == "set_output_quality":
                        try:
                            output_quality = max(1, min(100, int(payload.get("value", output_quality))))
                            if app.state.runtime is not None:
                                app.state.runtime.output_quality = output_quality
                            if h264_stream is not None:
                                if flow.get("clamped"):
                                    h264_stream.set_quality(min(int(output_quality), 20))
                                else:
                                    h264_stream.set_quality(output_quality)
                        except (TypeError, ValueError):
                            pass
                        continue
                    else:
                        await _send_json({"type": "error", "message": f"unknown message type: {msg_type}"})
                    continue

                if "bytes" not in message or message["bytes"] is None:
                    continue
                if session is None:
                    await _send_json({"type": "error", "message": "send start JSON before frames"})
                    continue

                frame_bytes = message["bytes"]
                frames_in += 1
                last_activity = time.monotonic()
                ws_debug["frames_in"] = frames_in
                ws_debug["last_receive_at"] = time.time()
                ws_debug["last_message_type"] = "frame_bytes"

                frame_meta = next_frame_meta or {
                    "seq": frames_in,
                    "t_capture_ms": time.time() * 1000.0,
                }
                frame_meta["t_server_recv_ms"] = time.time() * 1000.0
                next_frame_meta = None

                if not input_sniffed:
                    input_sniffed = True
                    _jpeg_magic = frame_bytes[:3] == b"\xff\xd8\xff"
                    _annexb_magic = frame_bytes[:4] == b"\x00\x00\x00\x01" or frame_bytes[:3] == b"\x00\x00\x01"
                    if _jpeg_magic and h264_ingest is not None:
                        h264_ingest = None
                        input_codec = "mjpeg"
                        print("#####[STREAM] uplink sniffed JPEG -> input_codec=mjpeg", flush=True)
                    elif _annexb_magic and h264_ingest is None and _up_allow:
                        h264_ingest = _H264Ingest()
                        input_codec = "h264"
                        print("#####[STREAM] uplink sniffed H.264 -> input_codec=h264", flush=True)

                uplink_frame: Image.Image | None = None
                if h264_ingest is not None:
                    _dec_t0 = time.perf_counter()
                    uplink_frame = await asyncio.to_thread(h264_ingest.decode_one, frame_bytes)
                    if session_settings is not None and session_settings.profile_timings:
                        frame_meta["jpeg_decode_ms"] = (time.perf_counter() - _dec_t0) * 1000.0
                    if uplink_frame is None:
                        continue

                if kv_reset_frames > 0 and frames_since_session_reset >= kv_reset_frames:
                    face_gate_pending = entry_gate
                    gate_state["count"] = 0
                    gate_state["cx"] = None
                    gate_state["cy"] = None
                    gate_state["pe_anchor"] = None
                    gate_state["settle_ax"] = None
                    gate_state["settle_ay"] = None
                    gate_state["subject_count"] = None
                    gate_state["cand"] = None
                    gate_state["cand_n"] = 0

                    if pe_report and pe_report.get("enhanced_prompt"):
                        session_prompt = pe_report["enhanced_prompt"]
                    await _reset_session("kv_reset_frames")
                    continue

                _prof_on = session_settings is not None and session_settings.profile_timings

                def _run_frame():
                    if uplink_frame is not None:
                        frame = uplink_frame
                    elif _prof_on:
                        _dec_t0 = time.perf_counter()
                        frame = _decode_image(frame_bytes)
                        frame_meta["jpeg_decode_ms"] = (time.perf_counter() - _dec_t0) * 1000.0
                    else:
                        frame = _decode_image(frame_bytes)

                    if face_gate_pending:
                        _reason, _center, _nf = _check_face_gate(
                            frame,
                            onnx_path=args.face_detector_onnx,
                            score_thresh=fg_score,
                            min_below_ratio=fg_min_below,
                        )
                        if _reason is not None:
                            gate_state["count"] = 0
                            gate_state["settle_ax"] = None
                            gate_state["settle_ay"] = None
                            return _reason
                        if _center is not None:
                            _cx, _cy, _csz = _center

                            if _nf < 2 and abs(_cx - 0.5) > float(args.face_gate_center_margin):
                                gate_state["count"] = 0
                                gate_state["settle_ax"] = None
                                gate_state["settle_ay"] = None
                                gate_state["settle_asz"] = None
                                gate_state["cx"] = _cx
                                gate_state["cy"] = _cy
                                gate_state["csz"] = _csz
                                return "off_center"
                            _pcx, _pcy, _pcsz = gate_state["cx"], gate_state["cy"], gate_state.get("csz")
                            _eps = gate_move_eps
                            _cap = float(args.face_gate_settle_drift)

                            _szeps = _eps * 0.5

                            _still = (_pcx is not None and abs(_cx - _pcx) <= _eps and abs(_cy - _pcy) <= _eps and
                                      _pcsz is not None and abs(_csz - _pcsz) <= _szeps)
                            _ax, _ay, _asz = gate_state.get("settle_ax"), gate_state.get("settle_ay"), gate_state.get("settle_asz")
                            if (_still and _ax is not None and abs(_cx - _ax) <= _cap and abs(_cy - _ay) <= _cap and
                                    _asz is not None and abs(_csz - _asz) <= _cap):
                                gate_state["count"] += 1
                            else:
                                gate_state["count"] = 1
                                gate_state["settle_ax"] = _cx
                                gate_state["settle_ay"] = _cy
                                gate_state["settle_asz"] = _csz
                            gate_state["cx"] = _cx
                            gate_state["cy"] = _cy
                            gate_state["csz"] = _csz
                            if gate_state["count"] < fg_stable:
                                return "settling"

                        if pe_defer:
                            gate_state["pe_anchor"] = frame
                            return "__gate_pe__"

                    if gate_on and not face_gate_pending:
                        _tick = gate_state.get("person_check_i", 0)
                        _gfaces, _gfw, _gfh = _detect_gate_faces(frame, onnx_path=args.face_detector_onnx, score_thresh=fg_score)
                        if _tick == 0 or gate_state.get("absent_hold"):
                            gate_state["person_last"] = _person_present(
                                frame, onnx_path=args.person_detector_onnx, conf=float(args.person_gate_conf))

                            if gate_state["person_last"]:
                                gate_state["face_last"] = _face_present_from(
                                    _gfaces, _gfw, _gfh,
                                    min_ratio=float(args.face_present_min_ratio),
                                    edge_margin=float(args.face_present_edge_margin))
                            else:
                                gate_state["face_last"] = True
                        gate_state["person_check_i"] = (_tick + 1) % person_stride
                        _body_here = bool(gate_state["person_last"])
                        _face_here = bool(gate_state.get("face_last", True))
                        _present = _body_here and _face_here
                        _reason_now = "no_person" if not _body_here else ("no_face" if not _face_here else "")

                        body_flip = body_flip_frames
                        if _body_here:
                            gate_state["body_miss"] = 0
                        else:
                            gate_state["body_miss"] = gate_state.get("body_miss", 0) + 1
                        if _present:
                            gate_state["absent"] = 0
                            if gate_state.get("absent_hold"):
                                gate_state["present"] = gate_state.get("present", 0) + 1
                                if gate_state["present"] >= fg_return:
                                    gate_state["absent_hold"] = False
                                    gate_state["present"] = 0
                                    print("#####[PERSON-GATE] subject returned (stable) -> re-run startup gate (reset)", flush=True)
                                    return ("__person_returned__",)

                        else:
                            gate_state["present"] = 0
                            gate_state["absent"] += 1

                            if _reason_now == "no_person" and gate_state.get("body_miss", 0) < body_flip:
                                _reason_now = "no_face"
                            gate_state["hold_reason"] = _reason_now or gate_state.get("hold_reason") or "no_person"
                            if not gate_state.get("absent_hold") and gate_state["absent"] >= fg_absent:
                                gate_state["absent_hold"] = True

                                try:
                                    if session is not None:
                                        session.pending_frames.clear()
                                        session.pending_metas.clear()
                                except Exception:
                                    pass
                                print(f"#####[PERSON-GATE] {gate_state['hold_reason']} for {gate_state['absent']} frames -> black-hold", flush=True)
                        if gate_state.get("absent_hold"):
                            return ("__no_person__", gate_state.get("hold_reason", "no_person"))

                        _n = _count_faces_from(
                            _gfaces, _gfw, _gfh,
                            count_min_ratio=float(args.count_face_min_ratio))
                        if gate_state["subject_count"] is None:
                            gate_state["subject_count"] = _n
                            gate_state["cand"] = None
                            gate_state["cand_n"] = 0
                        elif _n > gate_state["subject_count"]:
                            if _n == gate_state["cand"]:
                                gate_state["cand_n"] += 1
                            else:
                                gate_state["cand"] = _n
                                gate_state["cand_n"] = 1
                            if gate_state["cand_n"] >= count_change_frames:
                                gate_state["recount"] = True
                                gate_state["subject_count"] = _n
                                gate_state["cand"] = None
                                gate_state["cand_n"] = 0
                        elif _n < gate_state["subject_count"]:
                            if _n == gate_state["cand"]:
                                gate_state["cand_n"] += 1
                            else:
                                gate_state["cand"] = _n
                                gate_state["cand_n"] = 1
                            if gate_state["cand_n"] >= count_change_frames:
                                gate_state["subject_count"] = _n
                                gate_state["cand"] = None
                                gate_state["cand_n"] = 0
                        else:
                            gate_state["cand"] = None
                            gate_state["cand_n"] = 0

                    if pe_defer and not face_gate_pending:
                        gate_state["pe_anchor"] = frame
                        return "__gate_pe__"

                    health_error = _session_health_error(session)
                    if health_error:
                        raise RuntimeError(health_error)

                    acquired = app.state.inference_lock.acquire(
                        timeout=max(0.1, float(args.inference_lock_timeout_s))
                    )
                    if not acquired:
                        raise TimeoutError(
                            f"inference lock timeout after {args.inference_lock_timeout_s:.1f}s"
                        )
                    try:
                        health_error = _session_health_error(session)
                        if health_error:
                            raise RuntimeError(health_error)

                        if session_max_inflight and session.inflight_chunks() >= session_max_inflight:
                            ws_debug["frames_dropped_backpressure"] = (
                                int(ws_debug.get("frames_dropped_backpressure", 0)) + 1
                            )
                            return session._drain_async_results()

                        _rec_i = rec_input
                        if _rec_i is not None:
                            _rec_i.submit(frame, frame_meta.get("t_capture_ms"))
                            ws_debug["rec_in_written"] = _rec_i.frames_written
                            ws_debug["rec_in_dropped"] = _rec_i.frames_dropped_recording
                        return session.push_frame(frame, frame_meta=frame_meta)
                    finally:
                        app.state.inference_lock.release()

                started = time.time()
                try:
                    chunk_results = await asyncio.wait_for(
                        asyncio.to_thread(_run_frame),
                        timeout=max(0.1, float(args.push_frame_timeout_s)),
                    )
                except asyncio.TimeoutError:
                    msg = f"push_frame timeout after {args.push_frame_timeout_s:.1f}s"
                    print(f"#####[WS-GUARD] {msg}", flush=True)
                    await _send_json({"type": "error", "message": msg})
                    break
                except Exception as exc:
                    await _send_json({"type": "error", "message": repr(exc)})
                    break
                if isinstance(chunk_results, tuple) and chunk_results and isinstance(chunk_results[0], str):
                    sentinel = chunk_results[0]
                    if sentinel == "__no_person__":
                        _hold_reason = chunk_results[1] if len(chunk_results) > 1 else "no_person"
                        await _send_json({"type": "no_person", "reason": _hold_reason, "frames_in": frames_in})
                        continue
                    if sentinel == "__person_returned__":
                        face_gate_pending = True
                        gate_state["count"] = 0
                        gate_state["cx"] = None
                        gate_state["cy"] = None
                        gate_state["pe_anchor"] = None
                        gate_state["settle_ax"] = None
                        gate_state["settle_ay"] = None
                        gate_state["subject_count"] = None
                        gate_state["cand"] = None
                        gate_state["cand_n"] = 0

                        if pe_report and pe_report.get("enhanced_prompt"):
                            session_prompt = pe_report["enhanced_prompt"]
                        await _reset_session("person_returned")
                        continue
                if isinstance(chunk_results, str) and chunk_results == "__gate_pe__":
                    if pe_task is None:
                        anchor = gate_state.get("pe_anchor")
                        gate_state["pe_anchor"] = None
                        await _send_json({"type": "pe_running", "frames_in": frames_in})

                        async def _run_pe(_sess=session, _anchor=anchor, _raw=raw_session_prompt, _ref=ref_image):
                            nonlocal pe_defer, pe_report, pe_task
                            try:
                                report = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _enhance_prompt_sync,
                                        raw_prompt=_raw, ref_image=_ref,
                                        pe_frame=_anchor, pe_model=args.pe_model,
                                    ),
                                    timeout=max(1.0, float(args.pe_timeout_s)),
                                )
                                txt = str(report.get("enhanced_prompt") or _raw)

                                def _swap():
                                    with app.state.inference_lock:
                                        _sess.prompt = txt
                                        _sess._initialize(_anchor)
                                await asyncio.to_thread(_swap)
                                pe_report = report
                                ws_debug["pe_report"] = report
                                await _send_json({"type": "prompt_enhanced", **report})
                            except Exception:
                                def _swap_raw():
                                    with app.state.inference_lock:
                                        _sess._initialize(_anchor)
                                await asyncio.to_thread(_swap_raw)
                                await _send_json({"type": "prompt_enhanced", "enabled": True,
                                                  "raw_prompt": _raw, "enhanced_prompt": _raw,
                                                  "fallback": True, "error": True, "elapsed_s": 0.0})
                            finally:
                                pe_defer = False
                                pe_task = None
                                _write_prompt_sidecar()

                        pe_task = asyncio.create_task(_run_pe())
                    last_activity = time.monotonic()
                    continue
                if isinstance(chunk_results, str):
                    await _send_json({"type": "waiting_face", "reason": chunk_results, "frames_in": frames_in})
                    continue
                if face_gate_pending:
                    face_gate_pending = False
                    print("#####[FACE-GATE] passed -> editing starts", flush=True)
                frames_since_session_reset += 1
                ws_debug["frames_since_session_reset"] = frames_since_session_reset
                if gate_state.get("recount"):
                    gate_state["recount"] = False
                    print("#####[PERSON-GATE] person count changed -> re-edit (reset)", flush=True)

                    if pe_report and pe_report.get("enhanced_prompt"):
                        session_prompt = pe_report["enhanced_prompt"]
                    await _reset_session("person_count_changed")
                    continue
                elapsed = time.time() - started
                if not chunk_results:
                    await _send_json(
                        {
                            "type": "accepted",
                            "frames_in": frames_in,
                            "frames_out": frames_out,
                            "next_chunk_needs": session.frames_per_next_chunk - len(session.pending_frames),
                        }
                    )
                    continue

                last_count = 0
                for result in chunk_results:
                    last_count = await _send_chunk_result(
                        result,
                        fallback_meta=frame_meta,
                        fallback_elapsed=elapsed,
                    )
                if last_count == 0:
                    await _send_json(
                        {
                            "type": "accepted",
                            "frames_in": frames_in,
                            "frames_out": frames_out,
                            "next_chunk_needs": session.frames_per_next_chunk - len(session.pending_frames),
                        }
                    )
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:
            if "disconnect" not in str(exc).lower():
                raise
        finally:
            try:
                await _cancel_pe()
                await _stop_output_task()
                await asyncio.to_thread(_stop_recorders)
                if session is not None:
                    await _close_session_safely(session, "finally")
                if getattr(app.state, "active_session", None) is session:
                    app.state.active_session = None
                ws_debug["closed_at"] = time.time()
                ws_debug["send_state"] = "closed"
            finally:
                if ticket is not None:
                    app.state.session_gate.release(ticket)

    return app

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve JoyOmni online v2v streaming inference.")
    parser.add_argument("--dit-ckpt", type=str, default=DEFAULT_DIT_CKPT)
    parser.add_argument("--vae-ckpt", type=str, default=None, help="Override VAE checkpoint dir (else uses the config default).")
    parser.add_argument("--text-encoder-ckpt", type=str, default=None, help="Override text-encoder checkpoint dir (else uses the config default).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vae-device", type=str, default=None)
    parser.add_argument("--vae-encode-device", type=str, default=None)
    parser.add_argument("--vae-decode-device", type=str, default=None)
    parser.add_argument("--vae-pseudo-device", type=str, default=None)
    parser.add_argument("--postprocess-device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=840)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24, help="Client send/playback target fps; seeds the #fps UI control and paces the live camera. Recording follows each frame's t_capture_ms (media time for mp4 uploads, capture wall-clock for the live camera), so this does not set the recorded file's duration.")
    parser.add_argument("--use-pe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe-model", type=str, default=None)
    parser.add_argument("--pe-timeout-s", type=float, default=20.0, help="Hard wall-clock cap for deferred prompt-enhancement. On timeout the session degrades to the RAW prompt and starts editing, so a slow/hung PE endpoint (bad network / provider stall) can never wedge the client in the prompt-enhancement state.")

    parser.add_argument("--face-detector-onnx", type=str, default=DEFAULT_FACE_DETECTOR_ONNX, help="YuNet ONNX weight for the face-presence gate. Missing -> gate disabled (edits run unconditionally).")
    parser.add_argument("--face-gate-score", type=float, default=0.35, help="Min YuNet confidence to count as a face. Lower = detects motion-blurred faces (fewer transient drops), but more false positives.")
    parser.add_argument("--face-present-min-ratio", type=float, default=0.15, help="Mid-session presence: a detected face counts as 'present' only if its short side is >= this fraction of the frame short side. A too-small/partial face (subject sat down so only the top of the head shows) counts as no-face -> black-hold, instead of letting the model t2v-hallucinate a person. 0 = any face counts. Higher = stricter (black out sooner when the face gets small/far). Normal editing faces measure ~0.35, so 0.15 has a wide margin.")
    parser.add_argument("--face-present-edge-margin", type=float, default=0.0, help="Mid-session presence: a face whose box comes within this fraction of ANY frame border counts as a HALF/partial face (turned/leaned out) -> no-face -> black-hold, so the model never edits a half-face frame (which it fills in as a t2v hallucination). 0 = no edge check (default: disabled -- the face-box edge check false-blacked too eagerly when a face merely neared a border). Set e.g. 0.02 to re-enable a lenient check.")
    parser.add_argument("--face-gate-min-below-ratio", type=float, default=0.20, help="Min fraction of frame HEIGHT that must be below the face (torso room, for garment try-on). Bigger -> stricter: the chin must sit higher in frame (back up, sit taller, or re-aim the camera).")
    parser.add_argument("--face-gate-center-margin", type=float, default=0.35, help="Max |face-center-x - 0.5| (fraction of width) for a SINGLE subject to count as centered. Bigger -> more lenient. SKIPPED entirely when 2+ comparable faces are present (side-by-side people can't be centered). Note: motion/stability is enforced separately by --face-gate-move-eps + --face-gate-settle-drift, so this does not affect the swing-into-frame ghost fix.")
    parser.add_argument("--face-gate-move-eps", type=float, default=0.02, help="Max per-frame face-center movement (fraction of frame) to count as 'still'. Bigger -> tolerates more motion. Pairs with --face-gate-settle-drift (net drift from anchor) so a slow glide can't creep through frame-by-frame. Also requires per-frame face-size change <= 0.5*eps.")
    parser.add_argument("--face-gate-settle-drift", type=float, default=0.05, help="Max net drift of the face center AND size from the settle-streak anchor (fraction of frame). Closes the 'slow continuous glide' hole where every per-frame step is < move-eps but they sum to a big slide (swing-into-frame motion baked into chunk0 -> ghost/duplicate person). Smaller = must hold more still. Complements --face-gate-move-eps (per-frame) + --face-gate-stable-frames (streak length).")
    parser.add_argument("--face-gate-stable-frames", type=int, default=24, help="Consecutive centered+still frames required before editing starts (~24fps send rate, so 24 ≈ 1s).")

    parser.add_argument("--online-gate", action=argparse.BooleanOptionalAction, default=True, help="Server-wide master for MID-SESSION monitoring (no-person black-hold + person-count re-edit). A session runs it only when its gate_enabled is also true (browser checkbox / start field, default true); the ENTRY gate follows gate_enabled alone. Also seeds the UI checkbox default. --no-online-gate disables mid-session monitoring for all sessions.")
    parser.add_argument("--presence-absent-frames", type=int, default=12, help="Consecutive not-present frames (body missing, OR face too small / half-out per --face-present-*) before the output goes black. Small = stop FAST (less T2V leak on a quick sit-down / turn-away); larger = tolerate a brief occlusion / head-turn without black-holding. ~24fps, 12 ≈ 0.5s.")
    parser.add_argument("--presence-return-frames", type=int, default=24, help="Consecutive present (body+face) frames required to LEAVE the black-hold and re-run the startup gate. Separate from --presence-absent-frames so entry stays fast (black out quickly) while exit is well de-bounced: a face flickering through finger gaps while hands cover the face won't bounce no_face<->settling. ~24fps, 24 ≈ 1s.")
    parser.add_argument("--person-count-change-frames", type=int, default=24, help="Consecutive frames a NEW face count must hold before it is accepted. An INCREASE then re-edits (reset chunk0) so people who enter later get edited; a decrease only lowers the baseline. Debounce vs transient miscounts (sway / motion blur / a background face flickering in). Keep this larger than --presence-absent-frames so a brief face loss black-holds instead of faking a 0->1 'new person' re-edit. ~24fps, 24 ≈ 1s.")
    parser.add_argument("--count-face-min-ratio", type=float, default=0.45, help="For person-count-change: a face counts as an additional subject only if its short side is >= this fraction of the MAIN (largest/foreground) face's short side; an absolute floor of 5% of the frame short side also applies. Excludes far-smaller BACKGROUND people (e.g. a coworker behind the subject) that otherwise flip the count and trigger spurious re-edits. Higher = stricter (ignore more background).")
    parser.add_argument("--person-detector-onnx", type=str, default=DEFAULT_PERSON_DETECTOR_ONNX, help="YOLOv8n ONNX (fixed 320 input) for mid-session body presence via cv2.dnn. Missing/unloadable -> the body check passes through (always 'present'); face-present rules still apply, so no_face black-holds can still trigger.")
    parser.add_argument("--person-gate-conf", type=float, default=0.4, help="Min YOLO person-class score to count the person as present.")
    parser.add_argument("--person-check-stride", type=int, default=2, help="Run the person detector every Nth frame during editing (YOLO ~27ms; stride amortizes the cost). Smaller = notices the subject LEAVING sooner, more CPU. During a black-hold the check runs every frame regardless, so return detection is unaffected by the stride.")
    parser.add_argument("--person-body-flip-frames", type=int, default=6, help="Consecutive body-misses before the client reason flips to no_person. Below this, a lone YOLO dip (a hand/object over the face also clips the torso) keeps the current reason -- normally show_full_face -- so the hint doesn't strobe no_face<->no_person. Reason-only de-bounce; the black-hold timing (--presence-absent-frames) is unaffected. ~24fps, 6 ≈ 0.25s. Higher = more reluctant to ever show no_person; 1 = report no_person on the first miss (old behavior).")
    parser.add_argument("--output-quality", default="auto",
                        help="Downlink preview quality: 'auto' (RTT-adaptive) or a fixed 1-100.")
    parser.add_argument("--uplink-codec", choices=["auto", "mjpeg"],
                        default=os.environ.get("JOYOMNI_UPLINK_CODEC", "auto"),
                        help="Uplink transport. 'mjpeg' forces per-frame JPEG (preserves face consistency; h264 P-frames smear the model input). 'auto' honors browser h264. Default auto.")
    parser.add_argument("--downlink-codec", choices=["auto", "mjpeg"],
                        default=os.environ.get("JOYOMNI_DOWNLINK_CODEC", "auto"),
                        help="Downlink transport. 'auto' honors browser h264 (saves bandwidth; display-only, does not affect generated identity). 'mjpeg' forces JPEG. Default auto.")
    parser.add_argument("--prompt", type=str, default="Keep the person and scene temporally consistent while applying the requested edit.")
    parser.add_argument("--profile-timings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kv-reset-frames", type=int, default=1080)
    parser.add_argument("--max-temporal-ids", type=int, default=None)
    parser.add_argument("--freeze-kv-on-static", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-diff-thresh", type=float, default=0.5)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-frame-timeout-s", type=float, default=15.0, help="Max seconds a single frame submission may block before releasing the WS session gate.")
    parser.add_argument("--max-inflight-chunks", type=int, default=2, help="Drop incoming frames while this many chunks are already in the pipeline (latency governor; pins display latency to ~N chunk periods). 2 keeps glass-to-glass latency low (~1 chunk) and prevents the session-start init stall from building a frame backlog (the 'ramp-up'); higher values buffer more (smoother under hiccups) at the cost of latency and a startup ramp. 0 disables. Overridable per session via the start payload's max_inflight_chunks.")
    parser.add_argument("--session-close-timeout-s", type=float, default=5.0, help="Best-effort session cleanup timeout during WS teardown.")
    parser.add_argument("--inference-lock-timeout-s", type=float, default=5.0, help="Max seconds to wait for the process-wide inference lock.")

    parser.add_argument("--record-dir", type=str, default=None, help="Directory to record input/output mp4s into (per-session subfolder). Off if unset.")
    parser.add_argument("--record-codec", type=str, default="libx264", help="Recording video codec (PyAV/ffmpeg name).")
    parser.add_argument("--record-bitrate", type=int, default=8_000_000, help="Recording target bitrate in bits/sec.")
    parser.add_argument("--record-segment-seconds", type=int, default=300, help="Roll to a new mp4 segment every N seconds of recorded frames.")
    return parser

def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()

def main() -> None:
    import logging
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
    from xvideo.inductor_autotune_fix import install as _install_autotune_fix
    _install_autotune_fix()
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", ws_max_size=32 * 1024 * 1024, ws_per_message_deflate=False, loop="uvloop")


if __name__ == "__main__":
    main()
