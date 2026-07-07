#!/usr/bin/env python3
"""
run.py
------
Real-time License Plate Recognition (PC port of the Jetson Nano project).

This reuses the exact same trained ONNX models that shipped with the original
Jetson Nano repo, but replaces jetson.inference/jetson.utils (which require a
Jetson board + TensorRT + CSI/USB camera driver stack) with plain OpenCV +
ONNX Runtime, so it runs on any Windows/Mac/Linux machine with a webcam.

Usage:
    # Live webcam (default camera 0)
    python3 run.py --source 0

    # Video file
    python3 run.py --source path/to/video.mp4 --output out.mp4

    # Single image
    python3 run.py --source path/to/car.jpg --output out.jpg

    # Tune detection thresholds
    python3 run.py --source 0 --plate-threshold 0.6 --char-threshold 0.4
"""

import argparse
import time
import sys

import cv2

from src.plate_pipeline import LicensePlateRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time License Plate Recognition")
    parser.add_argument("--source", type=str, default="0",
                         help="Camera index (e.g. 0), video file path, or image file path")
    parser.add_argument("--output", type=str, default=None,
                         help="Optional path to save annotated video/image output")
    parser.add_argument("--plate-threshold", type=float, default=0.5, help="Plate detection confidence threshold")
    parser.add_argument("--char-threshold", type=float, default=0.5, help="Character detection confidence threshold")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window (useful over SSH)")
    return parser.parse_args()


def is_image_file(path):
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def draw_results(frame, results):
    for res in results:
        x1, y1, x2, y2 = res["plate_box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = res["text"] if res["text"] else "?"
        text_str = f"{label} ({res['plate_score']:.2f})"
        (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, text_str, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        for c in res["chars"]:
            cx1, cy1, cx2, cy2 = c["box"]
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 180, 0), 1)
    return frame


def run_on_image(recognizer, args):
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"[ERROR] Could not read image: {args.source}")
        sys.exit(1)

    results = recognizer.recognize(frame)
    for res in results:
        print(f"Detected plate: {res['text']!r}  (score={res['plate_score']:.2f})")

    frame = draw_results(frame, results)
    out_path = args.output or "output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Saved annotated image to {out_path}")

    if not args.headless:
        cv2.imshow("License Plate Recognition", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_stream(recognizer, args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    seen_plates = {}  # simple de-duplicated log of recognized plate strings
    prev_time = time.time()

    print("[INFO] Press 'q' to quit." if not args.headless else "[INFO] Running headless. Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = recognizer.recognize(frame)
            for res in results:
                if res["text"] and res["text"] not in seen_plates:
                    seen_plates[res["text"]] = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[PLATE] {res['text']}  detected at {seen_plates[res['text']]}")

            frame = draw_results(frame, results)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if writer:
                writer.write(frame)

            if not args.headless:
                cv2.imshow("License Plate Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print("\n[SUMMARY] Unique plates recognized this session:")
    for plate, ts in seen_plates.items():
        print(f"  {plate}  (first seen {ts})")


def main():
    args = parse_args()
    recognizer = LicensePlateRecognizer(
        plate_threshold=args.plate_threshold,
        char_threshold=args.char_threshold,
    )

    if is_image_file(args.source):
        run_on_image(recognizer, args)
    else:
        run_on_stream(recognizer, args)


if __name__ == "__main__":
    main()
