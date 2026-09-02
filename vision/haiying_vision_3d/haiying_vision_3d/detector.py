"""YOLOv5 ONNX preprocessing and postprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize with unchanged aspect ratio and symmetric gray padding."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR array with shape (H, W, 3)")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or input_size <= 0:
        raise ValueError("image dimensions and input_size must be positive")
    scale = min(input_size / width, input_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def decode_yolov5_predictions(
    prediction: np.ndarray,
    original_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
    target_class_id: int = -1,
) -> list[dict]:
    """Decode one YOLOv5 output tensor into NMS-filtered image-space boxes."""
    output = np.asarray(prediction, dtype=np.float32)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError("only batch size 1 is supported")
        output = output[0]
    if output.ndim != 2 or output.shape[1] < 6:
        raise ValueError("YOLO output must have shape (N, 5 + classes)")
    if scale <= 0.0:
        raise ValueError("letterbox scale must be positive")

    objectness = output[:, 4]
    class_ids = np.argmax(output[:, 5:], axis=1)
    class_scores = output[np.arange(output.shape[0]), class_ids + 5]
    scores = objectness * class_scores
    keep = scores >= confidence_threshold
    if target_class_id >= 0:
        keep &= class_ids == target_class_id
    rows = output[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if rows.size == 0:
        return []

    image_height, image_width = original_shape
    boxes = []
    valid_scores = []
    valid_classes = []
    for row, score, class_id in zip(rows, scores, class_ids):
        center_x, center_y, box_width, box_height = map(float, row[:4])
        left = (center_x - box_width / 2.0 - pad_x) / scale
        top = (center_y - box_height / 2.0 - pad_y) / scale
        width = box_width / scale
        height = box_height / scale
        x1 = max(0.0, min(float(image_width - 1), left))
        y1 = max(0.0, min(float(image_height - 1), top))
        x2 = max(0.0, min(float(image_width - 1), left + width))
        y2 = max(0.0, min(float(image_height - 1), top + height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        valid_scores.append(float(score))
        valid_classes.append(int(class_id))
    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes, valid_scores, confidence_threshold, iou_threshold
    )
    if len(indices) == 0:
        return []
    flattened = np.asarray(indices).reshape(-1)
    detections = []
    for index in flattened:
        x, y, width, height = boxes[int(index)]
        detections.append(
            {
                "class_id": valid_classes[int(index)],
                "confidence": valid_scores[int(index)],
                "box": [x, y, width, height],
                "center": [x + width / 2.0, y + height / 2.0],
            }
        )
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)
