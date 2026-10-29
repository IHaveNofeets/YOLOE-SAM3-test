import os
import tempfile

from flask import Flask, jsonify, request

from classical_detect import detect_objects

app = Flask(__name__)


@app.route("/health")
def health():
    return "", 200


@app.route("/detect", methods=["POST"])
def detect():
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "image.jpg")
        request.files["image"].save(image_path)
        boxes = detect_objects(image_path)
    return jsonify(boxes=boxes)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800)
