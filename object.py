import math
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MIN_CONTENT_LENGTH"] = 5 * 1024 * 1024

DEFAULT_CONFIDENCE = 0.5
INFERENCE_SIZE = 960

model = YOLO(str(BASE_DIR / "yolov8s.pt"))
open_vocabulary_model = None
open_vocabulary_model_lock = Lock()


def parse_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Confidence must be a number between 0.1 and 0.9.") from error

    if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
        raise ValueError("Confidence must be between 0.1 and 0.9.")
    return confidence


def parse_requested_classes(value):
    if len(value) > 400:
        raise ValueError("Keep the object list under 400 characters.")

    classes = []
    seen = set()
    for item in value.split(","):
        label = item.strip()
        if not label:
            continue
        if len(label) > 40:
            raise ValueError("Each object name must be 40 characters or fewer.")
        normalized_label = label.casefold()
        if normalized_label not in seen:
            classes.append(label)
            seen.add(normalized_label)

    if not classes:
        raise ValueError("Enter at least one object name.")
    if len(classes) > 20:
        raise ValueError("Enter no more than 20 object names.")
    return classes


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:filename>")
def local_files(filename):
    file_path = BASE_DIR / filename
    if file_path.is_file():
        return send_from_directory(BASE_DIR, filename)
    return jsonify(error="Page or asset not found."), 404


@app.post("/detect")
def detect():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify(error="An image is required."), 400

    uploaded_image = image_file.read()
    if not uploaded_image:
        return jsonify(error="The uploaded image is empty."), 400

    image_bytes = np.frombuffer(uploaded_image, dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="The uploaded file is not a valid image."), 400

    try:
        confidence = parse_confidence(request.form.get("confidence", str(DEFAULT_CONFIDENCE)))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    requested_classes = request.form.get("classes", "").strip()
    if requested_classes:
        try:
            classes = parse_requested_classes(requested_classes)
        except ValueError as error:
            return jsonify(error=str(error)), 400

        global open_vocabulary_model
        try:
            with open_vocabulary_model_lock:
                if open_vocabulary_model is None:
                    open_vocabulary_model = YOLO(str(BASE_DIR / "yolov8m-worldv2.pt"))
                open_vocabulary_model.set_classes(classes)
                results = open_vocabulary_model.predict(
                    frame,
                    conf=confidence,
                    imgsz=INFERENCE_SIZE,
                    verbose=False,
                )
        except Exception:
            app.logger.exception("Open-vocabulary object detection failed.")
            return jsonify(error="Custom object detection failed. Check the server log for details."), 503
    else:
        try:
            results = model.predict(
                frame,
                conf=confidence,
                imgsz=INFERENCE_SIZE,
                verbose=False,
            )
        except Exception:
            app.logger.exception("Object detection failed.")
            return jsonify(error="Object detection failed. Check the server log for details."), 503

    height, width = frame.shape[:2]
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": result.names[class_id],
                "confidence": float(box.conf[0]),
            })

    return jsonify(width=width, height=height, detections=detections)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=False)
 