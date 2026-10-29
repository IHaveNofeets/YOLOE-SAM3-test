import base64
import json
import os
import shutil
import tempfile
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLOE

from yoloe_bake import bake_visual_prompts

BASE_WEIGHTS = "modelData/yoloe/yoloe-26l-seg.pt"
CURRENT_DIR = "modelData/currentModels/"
PROMPT_EMBEDDINGS = CURRENT_DIR + "lego_prompt.npz"
EXPORTED_MODEL = CURRENT_DIR + "yoloe-26l-seg.onnx"

app = Flask(__name__)
_lock = threading.Lock()
_model = None  # cached loaded YOLOE(EXPORTED_MODEL), (re)loaded after each successful /bake


def _decode_masks(encoded):
    if not encoded:
        return None
    masks = []
    for b64 in encoded:
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        masks.append((gray > 127).astype(np.float32))
    return np.stack(masks)


@app.route("/health")
def health():
    with _lock:
        ok = _model is not None
    return ("", 200) if ok else ("model not baked yet", 503)


@app.route("/bake", methods=["POST"])
def bake():
    global _model
    payload = json.loads(request.form["payload"])
    bboxes = np.array(payload["bboxes"], dtype=np.float32)
    cls = np.array(payload["cls"], dtype=np.int64)
    names = payload["names"]
    masks = _decode_masks(payload.get("masks"))

    os.makedirs(CURRENT_DIR, exist_ok=True)
    with _lock:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                ref_path = os.path.join(tmp_dir, "ref.jpg")
                request.files["image"].save(ref_path)

                prompts = {"masks": masks, "cls": cls} if masks is not None else {"bboxes": bboxes, "cls": cls}

                model = YOLOE(BASE_WEIGHTS)
                bake_visual_prompts(model, ref_path, prompts, names=names)
                model.save_prompt_embeddings(PROMPT_EMBEDDINGS)

                model = YOLOE(BASE_WEIGHTS)
                model.load_prompt_embeddings(PROMPT_EMBEDDINGS)
                exported_path = model.export(format="onnx")
                if os.path.abspath(exported_path) != os.path.abspath(EXPORTED_MODEL):
                    shutil.move(exported_path, EXPORTED_MODEL)

            _model = YOLOE(EXPORTED_MODEL)
        except Exception as e:
            return jsonify(status="error", message=str(e)), 500

    return jsonify(status="ok", num_classes=len(set(cls.tolist())))


@app.route("/detect", methods=["POST"])
def detect():
    with _lock:
        model = _model
    if model is None:
        return jsonify(status="error", message="model not baked yet"), 503

    conf = float(request.form.get("conf", 0.25))
    buf = np.frombuffer(request.files["image"].read(), np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    result = model.predict(image, conf=conf, verbose=False)[0]
    detections = [
        {
            "bbox": [float(v) for v in box.xyxy[0].tolist()],
            "conf": float(box.conf[0]),
            "cls": int(box.cls[0]),
            "name": result.names[int(box.cls[0])],
        }
        for box in result.boxes
    ]
    return jsonify(detections=detections, names=list(result.names.values()))


if os.path.exists(EXPORTED_MODEL):
    # A previous /bake already produced this (currentModels is a persistent bind mount) - load
    # it at startup so /detect works without requiring a fresh bake after every container restart.
    _model = YOLOE(EXPORTED_MODEL)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8600)
