import base64
import json
import os
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, request

from box_refine import refine_to_masks

app = Flask(__name__)


def _encode_png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return base64.b64encode(buf).decode("ascii")


@app.route("/health")
def health():
    return "", 200


@app.route("/refine", methods=["POST"])
def refine():
    """Refine rough boxes into tighter GrabCut masks - the "getting reference data ready" step
    other services (yoloe's /bake) consume, kept independent of both webpage and yoloe."""
    bboxes = np.array(json.loads(request.form["bboxes"]), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "ref.jpg")
        request.files["image"].save(image_path)
        masks, preview_bgr = refine_to_masks(image_path, bboxes)

    return jsonify(
        masks=[_encode_png_b64((m * 255).astype(np.uint8)) for m in masks],
        preview=_encode_png_b64(preview_bgr),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8900)
