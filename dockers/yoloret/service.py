"""Flask wrapper around the YOLO-ReT model code in modelData/yoloret (bind-mounted at
/app/modelData/yoloret), replacing the subprocess-per-call approach infer_once.py is used for on
bare-host runs - the model is loaded once at startup instead of once per request."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelData", "yoloret"))

from flask import Flask, jsonify, request

from yolo3.enums import BACKBONE
from yolo import YOLO

YOLORET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelData", "yoloret")
WEIGHTS = os.path.join(YOLORET_DIR, "checkpoints", "efficientnetb3_416_coco.h5")
CLASSES_PATH = os.path.join(YOLORET_DIR, "model_data", "coco_classes.txt")
ANCHORS_PATH = os.path.join(YOLORET_DIR, "model_data", "yolo_anchors.txt")
INPUT_SIZE = (416, 416)

app = Flask(__name__)
_yolo = YOLO(
    dict(
        backbone=BACKBONE.EFFICIENTNETB3,
        classes_path=CLASSES_PATH,
        anchors_path=ANCHORS_PATH,
        input_size=INPUT_SIZE,
        model=WEIGHTS,
        score=0.2,
        nms=0.5,
        num_scales=3,
    )
)


@app.route("/health")
def health():
    return "", 200


@app.route("/detect", methods=["POST"])
def detect():
    # score/nms thresholds are baked into the model's YoloEval layer at construction time (see
    # yolo.py's YoloModel.__init__) - they can't be changed per-request without rebuilding the
    # model, so this service (like every caller in this project) just uses the startup defaults.
    data = request.files["image"].read()
    t0 = time.perf_counter()
    boxes, scores, classes = _yolo.detect_image(data, draw=False)
    ms = round((time.perf_counter() - t0) * 1000, 1)  # compute time only, not upload or JSON
    return jsonify(
        ms=ms,
        boxes=[[float(v) for v in box] for box in boxes],
        scores=[float(s) for s in scores],
        classes=[_yolo.class_names[int(c)] for c in classes],
        class_ids=[int(c) for c in classes],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)
