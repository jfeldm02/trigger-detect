from __future__ import annotations

from typing import TYPE_CHECKING

DEFAULT_CONFIDENCE = 0.25
DEFAULT_TARGET_BRIGHTNESS = 130

from ultralytics import YOLOWorld


def get_model(
    model_name: str,
    vocabulary: list[str],
    device: str | None = None,
) -> "YOLOWorld":
    print(f"Loading model: {model_name}")
    model = YOLOWorld(model_name)
    model.set_classes(vocabulary)
    if device is None:
        device = "cpu"
    if hasattr(model, "to"):
        model.to(device)
    print(f"Model device set to: {device}")
    print(f"Vocabulary set to: {vocabulary}")
    return model


def process_frames(
    model: "YOLOWorld", frames: list, conf: float
) -> tuple[list, list]:
    """
    Run detection on a list of frames and return:
    - annotated_frames: list of annotated frames
    - detections: list of per-frame detections
    """
    annotated_frames = []
    detections = []

    for frame in frames:
        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

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

        detections.append(frame_detections)
        annotated_frames.append(result.plot())

    return annotated_frames, detections
