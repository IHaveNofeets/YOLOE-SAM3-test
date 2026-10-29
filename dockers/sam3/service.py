import json
import os
import tempfile
import time

import cv2
from flask import Flask, jsonify, request

from sam3_predict import detect_target_like_yoloe, segment_target_like_yoloe, segment_with_sam3

SAM3_WEIGHTS = "modelData/sam3/sam3.pt"

app = Flask(__name__)


@app.route("/health")
def health():
    return ("", 200) if os.path.exists(SAM3_WEIGHTS) else ("weights not found", 503)


def _encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode result as PNG")
    return buf.tobytes()


@app.route("/segment", methods=["POST"])
def segment():
    bboxes = json.loads(request.form["bboxes"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "image.jpg")
        request.files["image"].save(image_path)
        result = segment_with_sam3(image_path, bboxes, weights=SAM3_WEIGHTS)
    return _encode_png(result.plot()), 200, {"Content-Type": "image/png"}


@app.route("/segment_like_yoloe", methods=["POST"])
def segment_like_yoloe():
    ref_bboxes = json.loads(request.form["ref_bboxes"])
    height = int(request.form.get("height", 720))
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_path = os.path.join(tmp_dir, "ref.jpg")
        target_path = os.path.join(tmp_dir, "target.jpg")
        request.files["ref_image"].save(ref_path)
        request.files["target_image"].save(target_path)
        vis = segment_target_like_yoloe(ref_path, ref_bboxes, target_path, weights=SAM3_WEIGHTS, height=height)
    return _encode_png(vis), 200, {"Content-Type": "image/png"}


@app.route("/detect_like_yoloe", methods=["POST"])
def detect_like_yoloe():
    """Boxes as JSON instead of a drawn picture: SAM 3 prompted with ref_bboxes on ref_image,
    searching target_image. Returns {"boxes": [[x1, y1, x2, y2], ...], "scores": [...], "width", "height"}
    in target-image pixels."""
    ref_bboxes = json.loads(request.form["ref_bboxes"])
    height = int(request.form.get("height", 720))
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_path = os.path.join(tmp_dir, "ref.jpg")
        target_path = os.path.join(tmp_dir, "target.jpg")
        request.files["ref_image"].save(ref_path)
        request.files["target_image"].save(target_path)
        t0 = time.perf_counter()  # compute only: building the composite + SAM 3, not the upload or JSON
        found, w, h = detect_target_like_yoloe(ref_path, ref_bboxes, target_path, weights=SAM3_WEIGHTS, height=height)
        ms = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(boxes=[list(b[:4]) for b in found], scores=[b[4] for b in found], width=w, height=h, ms=ms)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8700)
