from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import base64
from flask import Blueprint, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from video_processing import DEFAULT_CONFIDENCE, get_model, process_frames

bp = Blueprint("video_upload", __name__)

TEMP_DIR = Path(__file__).parent / "temp_videos"
TEMP_DIR.mkdir(exist_ok=True)
TEMP_FILE = "temp.mp4"
EDITED_FILE = "temp_edited.mp4"
EDITED_JSON = "temp_edited.json"
SEGMENT_SECONDS = 5
BUFFER_SECONDS = 10

@bp.get("/upload")
def upload_page():
    tags = session.get("tags", [])
    video_path = TEMP_DIR / TEMP_FILE
    video_url = None
    if video_path.exists():
        video_url = url_for("video_upload.get_video", filename=TEMP_FILE)
    return render_template("video_upload.html", tags=tags, video_url=video_url)


@bp.get("/play")
def play_page():
    original_path = TEMP_DIR / TEMP_FILE
    video_url = None
    if original_path.exists():
        video_url = url_for(
            "video_upload.get_video",
            filename=TEMP_FILE,
            v=original_path.stat().st_mtime_ns,
        )
    return render_template(
        "play_video.html",
        video_url=video_url,
        buffer_seconds=BUFFER_SECONDS,
    )

class _StreamState:
    def __init__(self, session_id: str, fps: float, width: int, height: int):
        self.session_id = session_id
        self.fps = fps
        self.width = width
        self.height = height
        self.ready: List[Tuple[int, List[str]]] = []
        self.done = False
        self.error: Optional[str] = None
        self.cancelled = False
        self.cond = threading.Condition()


_streams: Dict[str, _StreamState] = {}


def _encode_frames(frames: List) -> List[str]:
    encoded = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode("ascii")
        encoded.append(b64)
    return encoded


def _stream_worker(session_id: str, input_path: Path, model_name: str, vocabulary: List[str]):
    state = _streams[session_id]
    try:
        model = get_model(model_name, vocabulary, device="cpu")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        state.fps = fps
        state.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        state.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        batch_size = int(fps * SEGMENT_SECONDS)

        index = 0
        while True:
            if state.cancelled:
                break
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if not frames:
                break

            annotated_frames, _detections = process_frames(model, frames, DEFAULT_CONFIDENCE)
            encoded_frames = _encode_frames(annotated_frames)

            with state.cond:
                state.ready.append((index, encoded_frames))
                state.cond.notify_all()

            index += 1

        cap.release()
        with state.cond:
            state.done = True
            state.cond.notify_all()
    except Exception as exc:
        with state.cond:
            state.error = str(exc)
            state.done = True
            state.cond.notify_all()


@bp.post("/stream/start")
def stream_start():
    input_path = TEMP_DIR / TEMP_FILE
    if not input_path.exists():
        return jsonify({"error": "No uploaded video found"}), 400

    vocabulary = session.get("tags", [])
    if not vocabulary:
        return jsonify({"error": "No tags available for vocabulary"}), 400

    model_name = os.getenv("YOLOWORLD_MODEL", "yolov8n-worldv2.pt")
    session_id = uuid.uuid4().hex

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    cap.release()

    state = _StreamState(session_id, fps=fps, width=width, height=height)
    _streams[session_id] = state

    worker = threading.Thread(
        target=_stream_worker,
        args=(session_id, input_path, model_name, vocabulary),
        daemon=True,
    )
    worker.start()

    return jsonify(
        {
            "session_id": session_id,
            "fps": state.fps,
            "width": state.width,
            "height": state.height,
        }
    )


@bp.get("/stream/next")
def stream_next():
    session_id = request.args.get("session_id", "")
    last_index = request.args.get("last_index", "-1")
    try:
        last_index = int(last_index)
    except ValueError:
        last_index = -1

    state = _streams.get(session_id)
    if state is None:
        return jsonify({"error": "Invalid session"}), 400

    with state.cond:
        end_time = time.time() + 30
        while True:
            next_item = next((item for item in state.ready if item[0] > last_index), None)
            if next_item:
                index, frames = next_item
                return jsonify(
                    {
                        "status": "ready",
                        "index": index,
                        "frames": frames,
                        "fps": state.fps,
                        "width": state.width,
                        "height": state.height,
                    }
                )

            if state.error:
                return jsonify({"status": "error", "error": state.error}), 500

            if state.done:
                return jsonify(
                    {
                        "status": "done",
                        "fps": state.fps,
                        "width": state.width,
                        "height": state.height,
                    }
                )

            remaining = end_time - time.time()
            if remaining <= 0:
                return jsonify(
                    {
                        "status": "waiting",
                        "fps": state.fps,
                        "width": state.width,
                        "height": state.height,
                    }
                )
            state.cond.wait(timeout=remaining)


@bp.post("/stream/stop")
def stream_stop():
    session_id = request.args.get("session_id", "")
    state = _streams.get(session_id)
    if state is None:
        return jsonify({"status": "ok"})
    with state.cond:
        state.cancelled = True
        state.done = True
        state.cond.notify_all()
    return jsonify({"status": "ok"})

def _process_video_job(input_path: Path, output_path: Path, json_path: Path, vocabulary: List[str]):
    try:
        PROCESS_STATE["status"] = "running"
        PROCESS_STATE["error"] = None

        model_name = os.getenv("YOLOWORLD_MODEL", "yolov8n-worldv2.pt")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("No frames read from video.")

        batch_size = int(fps * SEGMENT_SECONDS)
        batches = [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]

        results: List[Optional[tuple]] = [None] * len(batches)

        def run_batch(model_instance, idx: int, batch_frames: List):
            annotated_frames, detections = process_frames(
                model_instance, batch_frames, DEFAULT_CONFIDENCE
            )
            results[idx] = (idx, annotated_frames, detections)

        max_workers = 2
        threads = []
        next_index = 0
        lock = threading.Lock()

        def worker():
            nonlocal next_index
            model_instance = get_model(model_name, vocabulary, device="cpu")
            while True:
                with lock:
                    if next_index >= len(batches):
                        return
                    idx = next_index
                    next_index += 1
                run_batch(model_instance, idx, batches[idx])

        for _ in range(max_workers):
            t = threading.Thread(target=worker, daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_results = []
        frame_idx = 0

        for idx in range(len(results)):
            _, annotated_frames, detections = results[idx]
            for i, annotated in enumerate(annotated_frames):
                writer.write(annotated)
                frame_record = {
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "detections": detections[i],
                }
                all_results.append(frame_record)
                frame_idx += 1

        writer.release()

        # Ensure browser-compatible encoding (H.264 + yuv420p)
        temp_h264 = output_path.with_name(output_path.stem + "_h264.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "3.0",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temp_h264),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            temp_h264.replace(output_path)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip()
            raise RuntimeError(
                "FFmpeg failed to encode H.264. "
                "Ensure ffmpeg has libx264 enabled.\n"
                f"{err[:800]}"
            )

        output_data = {
            "video": str(input_path),
            "total_frames": frame_idx,
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "confidence_threshold": DEFAULT_CONFIDENCE,
            "auto_enhance": False,
            "target_brightness": None,
            "vocabulary": vocabulary,
            "frames": all_results,
        }
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        PROCESS_STATE["status"] = "done"
    except Exception as exc:
        PROCESS_STATE["status"] = "error"
        PROCESS_STATE["error"] = str(exc)




@bp.post("/upload-video")
def upload_video():
    if "video" not in request.files:
        return redirect(url_for("video_upload.upload_page"))

    file = request.files["video"]
    if not file.filename:
        return redirect(url_for("video_upload.upload_page"))

    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    ext = ext.lower() if ext else ".mp4"
    target = TEMP_DIR / f"temp{ext}"

    # Overwrite the temp video
    file.save(target)

    # Normalize to temp.mp4 if needed
    if target.name != TEMP_FILE:
        final_target = TEMP_DIR / TEMP_FILE
        if final_target.exists():
            final_target.unlink()
        target.rename(final_target)

    return redirect(url_for("video_upload.upload_page"))


@bp.get("/videos/<path:filename>")
def get_video(filename: str):
    return send_from_directory(TEMP_DIR, filename)
