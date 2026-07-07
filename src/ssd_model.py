"""
ssd_model.py
------------
Generic wrapper around the SSD-MobileNetV1 ONNX models produced by NVIDIA's
jetson-inference / pytorch-ssd training pipeline.

Both models used in this project (plate detector and character-OCR detector)
share the exact same I/O contract:

    input  : "input_0"  float32 tensor, shape (1, 3, 300, 300)
    output : "scores"   float32 tensor, shape (1, 3000, num_classes)  -- already softmax-ed
    output : "boxes"    float32 tensor, shape (1, 3000, 4)            -- already decoded,
                                                                          normalized (x1,y1,x2,y2)

This was confirmed by inspecting the graphs directly with onnxruntime, so no
SSD prior-box / anchor decoding is required here -- the export already bakes
that step in (this is standard for jetson-inference's onnx_export.py).
"""

import cv2
import numpy as np
import onnxruntime as ort


class SSDDetector:
    def __init__(self, model_path, labels_path, input_size=300, mean=127.0, std=128.0):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.mean = mean
        self.std = std

        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip() != ""]

    def _preprocess(self, bgr_image):
        resized = cv2.resize(bgr_image, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - self.mean) / self.std
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def detect(self, bgr_image, score_threshold=0.5, nms_threshold=0.45, exclude_background=True):
        """
        Runs the network on a single BGR image and returns a list of detections:
            [{"label": str, "class_id": int, "score": float, "box": (x1, y1, x2, y2)}, ...]
        Box coordinates are in ORIGINAL image pixel space.
        """
        h, w = bgr_image.shape[:2]
        blob = self._preprocess(bgr_image)

        scores, boxes = self.session.run(["scores", "boxes"], {self.input_name: blob})
        scores, boxes = scores[0], boxes[0]  # drop batch dim -> (3000, C), (3000, 4)

        num_classes = scores.shape[1]
        start_class = 1 if exclude_background else 0

        # For every prior, keep the best non-background class if it clears the threshold.
        class_ids = np.argmax(scores[:, start_class:], axis=1) + start_class
        class_scores = scores[np.arange(scores.shape[0]), class_ids]
        keep_mask = class_scores >= score_threshold

        if not np.any(keep_mask):
            return []

        kept_boxes = boxes[keep_mask]
        kept_scores = class_scores[keep_mask]
        kept_classes = class_ids[keep_mask]

        # scale normalized corner-form boxes to pixel space, clip to image bounds
        px_boxes = kept_boxes.copy()
        px_boxes[:, [0, 2]] = np.clip(px_boxes[:, [0, 2]] * w, 0, w - 1)
        px_boxes[:, [1, 3]] = np.clip(px_boxes[:, [1, 3]] * h, 0, h - 1)

        # cv2.dnn.NMSBoxes expects (x, y, width, height)
        nms_input = [
            [b[0], b[1], max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])] for b in px_boxes
        ]
        indices = cv2.dnn.NMSBoxes(nms_input, kept_scores.tolist(), score_threshold, nms_threshold)
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        detections = []
        for i in indices:
            x1, y1, x2, y2 = px_boxes[i]
            detections.append({
                "label": self.labels[kept_classes[i]],
                "class_id": int(kept_classes[i]),
                "score": float(kept_scores[i]),
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })
        return detections
