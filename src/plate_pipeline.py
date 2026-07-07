"""
plate_pipeline.py
------------------
Two-stage license plate recognition pipeline:

    1. Plate detector  (az_plate)  -> finds license plate bounding boxes in the full frame
    2. Character OCR   (az_ocr)    -> finds individual character boxes inside the cropped plate

Detected characters are sorted into reading order (handles both single-row and
the common two-row Vietnamese plate layout) and joined into a plate string.
"""

import os
import numpy as np

from .ssd_model import SSDDetector

NETWORKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "networks")


class LicensePlateRecognizer:
    def __init__(self,
                 plate_model=os.path.join(NETWORKS_DIR, "az_plate", "az_plate_ssdmobilenetv1.onnx"),
                 plate_labels=os.path.join(NETWORKS_DIR, "az_plate", "labels.txt"),
                 ocr_model=os.path.join(NETWORKS_DIR, "az_ocr", "az_ocr_ssdmobilenetv1_2.onnx"),
                 ocr_labels=os.path.join(NETWORKS_DIR, "az_ocr", "labels.txt"),
                 plate_threshold=0.5,
                 char_threshold=0.5,
                 plate_padding=0.08):
        self.plate_detector = SSDDetector(plate_model, plate_labels)
        self.ocr_detector = SSDDetector(ocr_model, ocr_labels)
        self.plate_threshold = plate_threshold
        self.char_threshold = char_threshold
        self.plate_padding = plate_padding

    def _crop_with_padding(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.plate_padding)
        pad_y = int(bh * self.plate_padding)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    @staticmethod
    def _assemble_text(chars):
        """chars: list of {"label", "score", "box"} within the plate crop.
        Groups into rows (handles 2-line plates), sorts each row left-to-right."""
        if not chars:
            return ""

        centers_y = np.array([(c["box"][1] + c["box"][3]) / 2.0 for c in chars])
        heights = np.array([c["box"][3] - c["box"][1] for c in chars])
        avg_h = float(np.mean(heights)) if len(heights) else 1.0

        order = np.argsort(centers_y)
        rows, current_row, last_y = [], [], None
        for idx in order:
            cy = centers_y[idx]
            if last_y is None or abs(cy - last_y) < avg_h * 0.6:
                current_row.append(chars[idx])
            else:
                rows.append(current_row)
                current_row = [chars[idx]]
            last_y = cy
        if current_row:
            rows.append(current_row)

        rows.sort(key=lambda row: np.mean([(c["box"][1] + c["box"][3]) / 2.0 for c in row]))

        line_strings = []
        for row in rows:
            row_sorted = sorted(row, key=lambda c: c["box"][0])
            line_strings.append("".join(c["label"] for c in row_sorted))
        return "-".join(line_strings)

    def recognize(self, frame):
        """
        Runs the full pipeline on one BGR frame.
        Returns list of results:
            [{"plate_box": (x1,y1,x2,y2), "plate_score": float,
              "text": str, "chars": [...]}, ...]
        """
        results = []
        plate_detections = self.plate_detector.detect(frame, score_threshold=self.plate_threshold)

        for det in plate_detections:
            if det["label"] != "plate":
                continue
            crop, offset = self._crop_with_padding(frame, det["box"])
            if crop.size == 0:
                continue

            char_detections = self.ocr_detector.detect(crop, score_threshold=self.char_threshold)
            text = self._assemble_text(char_detections)

            # shift char boxes back to full-frame coordinates for drawing/debugging
            for c in char_detections:
                x1, y1, x2, y2 = c["box"]
                c["box"] = (x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1])

            results.append({
                "plate_box": det["box"],
                "plate_score": det["score"],
                "text": text,
                "chars": char_detections,
            })

        return results
