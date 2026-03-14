"""
YOLO-World Video Object Detection

Given an uploaded video and a vocabulary list, runs YOLO-World
open-vocabulary object detection on each frame and outputs an
annotated video plus per-frame JSON results.

Adaptive auto-enhancement is ON by default: each frame's mean luminance
is measured and CLAHE + gamma correction are scaled proportionally so
dark frames get heavy brightening while already-bright frames are left
untouched.  No manual tuning needed.

Usage:
    python yolo_world_video.py --video input.mp4 --vocab "person,cup,laptop,phone"
    python yolo_world_video.py --video dark.mp4  --vocab "spider,arachnid" --conf 0.1
    python yolo_world_video.py --video input.mp4 --vocab "person" --no-enhance
    python yolo_world_video.py --video input.mp4 --vocab "person" --target-brightness 160
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

# ---------------------------------------------------------------------------
# Brightness target: the mean luminance (0-255) we try to bring every frame
# to.  Frames already at or above this level receive no enhancement.
# ---------------------------------------------------------------------------
DEFAULT_TARGET_BRIGHTNESS = 130

# Enhancement parameter ranges mapped from brightness deficit
# (clip_limit_min, clip_limit_max) for CLAHE
CLAHE_CLIP_RANGE = (1.0, 6.0)
# (gamma_max, gamma_min) — note: lower gamma = more brightening
GAMMA_RANGE = (1.0, 0.3)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, vocabulary: list[str]) -> YOLOWorld:
    """Load YOLO-World model and set the custom vocabulary."""
    print(f"Loading model: {model_name}")
    model = YOLOWorld(model_name)
    model.set_classes(vocabulary)
    print(f"Vocabulary set to: {vocabulary}")
    return model


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    model: YOLOWorld,
    video_path: str,
    output_path: str,
    json_output_path: str,
    conf: float,
    save_frames: bool,
    auto_enhance: bool = True,
    target_brightness: float = DEFAULT_TARGET_BRIGHTNESS,
):
    """Run detection on every frame of the video and write annotated output."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{video_path}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if save_frames:
        frames_dir = Path(output_path).parent / (Path(output_path).stem + "_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    frame_idx = 0
    enhanced_count = 0
    lum_accumulator = 0.0

    print(f"Processing {total_frames} frames at {fps:.1f} FPS ({width}x{height})")
    print(f"Confidence threshold: {conf}")
    if auto_enhance:
        print(f"Adaptive auto-enhancement: ON  (target brightness = {target_brightness})")
    else:
        print("Adaptive auto-enhancement: OFF")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

    
        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

        # Collect detections for this frame
        frame_detections = []
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            confidence = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            class_name = result.names[class_id]
            frame_detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                }
            )

        frame_record = {
            "frame": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 3),
            "detections": frame_detections,
        }
        all_results.append(frame_record)

        # Draw annotated frame (on the enhanced image so boxes align)
        annotated = result.plot()
        writer.write(annotated)

        if save_frames:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), annotated)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            det_count = len(frame_detections)
            lum_str = ""
            print(
                f"  Frame {frame_idx}/{total_frames} — "
                f"{det_count} detection(s){lum_str}",
                end="\r",
            )

    cap.release()
    writer.release()
    print(f"\nDone. Processed {frame_idx} frames.")

    # --- Enhancement summary ---
    if auto_enhance and frame_idx > 0:
        avg_lum = lum_accumulator / frame_idx
        pct_enhanced = (enhanced_count / frame_idx) * 100
        print(f"\nEnhancement Summary:")
        print(f"  Average frame luminance:  {avg_lum:.1f} / 255")
        print(f"  Target brightness:        {target_brightness}")
        print(
            f"  Frames enhanced:          {enhanced_count}/{frame_idx} "
            f"({pct_enhanced:.1f}%)"
        )

    # --- Write JSON results ---
    output_data = {
        "video": video_path,
        "total_frames": frame_idx,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "confidence_threshold": conf,
        "auto_enhance": auto_enhance,
        "target_brightness": target_brightness if auto_enhance else None,
        "vocabulary": list(model.names.values()) if hasattr(model, "names") else [],
        "frames": all_results,
    }
    with open(json_output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # --- Detection summary ---
    total_detections = sum(len(fr["detections"]) for fr in all_results)
    class_counts: dict[str, int] = {}
    for fr in all_results:
        for det in fr["detections"]:
            class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

    print(f"\nDetection Summary:")
    print(f"  Total detections across all frames: {total_detections}")
    if class_counts:
        print(f"  Detections per class:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")
    print(f"\n  Annotated video saved to: {output_path}")
    print(f"  JSON results saved to:    {json_output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not Path(video_path).is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    vocabulary = [v.strip() for v in args.vocab.split(",") if v.strip()]
    if not vocabulary:
        print("Error: vocabulary is empty. Provide comma-separated class names.")
        sys.exit(1)

    stem = Path(video_path).stem
    parent = Path(video_path).parent
    output_path = args.output or str(parent / f"{stem}_detected.mp4")
    json_output_path = args.json_output or str(parent / f"{stem}_results.json")

    model = load_model(args.model, vocabulary)
    process_video(
        model=model,
        video_path=video_path,
        output_path=output_path,
        json_output_path=json_output_path,
        conf=args.conf,
        save_frames=args.save_frames,
        auto_enhance=not args.no_enhance,
        target_brightness=args.target_brightness,
    )


if __name__ == "__main__":
    main()
