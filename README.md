# Real-Time License Plate Recognition (PC Edition)


## Why it was adapted

The original repo is written specifically for an **NVIDIA Jetson Nano**: it
imports `jetson.inference` / `jetson.utils`, and needs a MIPI CSI camera or a
TensorRT engine compiled on-device. Those libraries only exist on a physical
Jetson board, so the code as-is can't run on a normal laptop.

This build keeps **the exact same trained models** from the zip
(`az_plate_ssdmobilenetv1.onnx` and `az_ocr_ssdmobilenetv1_2.onnx`, both
SSD-MobileNetV1 detectors) but replaces the Jetson-only runtime with
**OpenCV + ONNX Runtime**, so it runs on any Windows/Mac/Linux machine with a
webcam, video file, or image — CPU only, no GPU/TensorRT required.

## How it works (two-stage pipeline)

1. **Plate detector** (`networks/az_plate`) scans the full frame and finds
   license-plate bounding boxes.
2. Each plate is cropped out of the frame.
3. **Character detector** (`networks/az_ocr`) runs on the crop and finds
   individual character boxes (`0-9`, `A-Z`).
4. Characters are grouped into rows (handles both single-row and two-row
   plate layouts) and sorted left-to-right to assemble the final plate string.

Both models' ONNX outputs were inspected directly with onnxruntime before
writing the decoder — they already return post-softmax scores and decoded
corner-form boxes, so no manual SSD anchor-box math is needed, just
thresholding + non-max suppression.

## Project structure

```
lpr/
├── networks/
│   ├── az_plate/            # plate detector model + labels (from the zip)
│   └── az_ocr/              # character OCR model + labels (from the zip)
├── src/
│   ├── ssd_model.py         # generic ONNX SSD wrapper (preprocess/infer/NMS)
│   └── plate_pipeline.py    # two-stage pipeline + text assembly
├── run.py                   # CLI app: webcam / video / image
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

**Live webcam:**
```bash
python3 run.py --source 0
```

**Video file, saving the annotated output:**
```bash
python3 run.py --source my_traffic_video.mp4 --output annotated.mp4
```

**Single image:**
```bash
python3 run.py --source car.jpg --output result.jpg
```

**Tune sensitivity:**
```bash
python3 run.py --source 0 --plate-threshold 0.6 --char-threshold 0.4
```

Press `q` in the display window to stop a live/video run. Use `--headless`
if running over SSH with no display — output still gets saved if `--output`
is given, and recognized plates print to the console with timestamps.

## Notes on accuracy

- The plate detector is quite reliable even on small/blurry frames (it was
  trained specifically for this).
- The character OCR stage needs a reasonably sharp, well-lit, front-on plate
  crop to read cleanly — exactly like the real Jetson demo, it works best
  with the camera a few meters from the vehicle and the plate roughly
  horizontal.
- Both `--plate-threshold` and `--char-threshold` are worth tuning per your
  camera/lighting; start around 0.5 and lower it if plates aren't being
  detected, raise it if you get false positives.

## Credits

Original models, dataset, and Jetson Nano implementation from the
`Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano` project. This
build only replaces the Jetson-specific runtime layer so the same models can
be used on ordinary hardware.

