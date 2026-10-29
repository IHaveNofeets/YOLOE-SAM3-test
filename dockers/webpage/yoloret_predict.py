import json
import os
import subprocess
import tempfile

import cv2
import requests

# YOLO-ReT (https://github.com/prakharg24/yoloret) is TensorFlow-based and only supports
# Python <=3.9 - it can't be imported into this project's Python/torch process. Two ways to
# reach it are supported: an HTTP call to the `yoloret` Docker service (when YOLORET_SERVICE_URL
# is set - see dockers/yoloret), or, on bare-host/desktop use (main.py), a subprocess into its own
# local venv under modelData/yoloret/.venv39 (see modelData/yoloret/README.md for how that venv
# was set up). The HTTP path is tried first only when YOLORET_SERVICE_URL is set, so bare-host
# behavior is completely unchanged when it isn't.
YOLORET_SERVICE_URL = os.environ.get("YOLORET_SERVICE_URL")

YOLORET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dockers", "yoloret", "modelData", "yoloret")
YOLORET_VENV_PYTHON = os.path.join(YOLORET_DIR, ".venv39", "Scripts", "python.exe")
YOLORET_INFER_SCRIPT = os.path.join(YOLORET_DIR, "infer_once.py")

# COCO EfficientNet-B3 @ 416x416 - the most accurate of the checkpoints bundled with the repo.
YOLORET_WEIGHTS = os.path.join(YOLORET_DIR, "checkpoints", "efficientnetb3_416_coco.h5")
YOLORET_BACKBONE = "EFFICIENTNETB3"
YOLORET_CLASSES = os.path.join(YOLORET_DIR, "model_data", "coco_classes.txt")
YOLORET_ANCHORS = os.path.join(YOLORET_DIR, "model_data", "yolo_anchors.txt")
YOLORET_INPUT_SIZE = (416, 416)


def yoloret_available():
    if YOLORET_SERVICE_URL:
        try:
            return requests.get(f"{YOLORET_SERVICE_URL}/health", timeout=2).ok
        except requests.RequestException:
            return False
    return os.path.exists(YOLORET_VENV_PYTHON) and os.path.exists(YOLORET_WEIGHTS)


def detect_with_yoloret(image_path, score=0.2, nms=0.5):
    """Run YOLO-ReT - a fixed-vocabulary (COCO) detector, unlike YOLOE's visual prompting - on
    image_path and return an annotated BGR image, for a fully independent comparison.
    """
    if YOLORET_SERVICE_URL:
        result = _detect_via_service(image_path, score, nms)
    else:
        result = _detect_via_subprocess(image_path, score, nms)

    vis = cv2.imread(image_path)
    for (top, left, bottom, right), conf, cls_name in zip(result["boxes"], result["scores"], result["classes"]):
        top, left, bottom, right = round(top), round(left), round(bottom), round(right)
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 165, 255), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(vis, label, (left, max(15, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    return vis


def _detect_via_service(image_path, score, nms):
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{YOLORET_SERVICE_URL}/detect",
            files={"image": f},
            data={"score": score, "nms": nms},
            timeout=60,
        )
    if not resp.ok:
        raise RuntimeError(f"YOLO-ReT service failed: {resp.status_code} {resp.text[:2000]}")
    return resp.json()


def _detect_via_subprocess(image_path, score, nms):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = os.path.join(tmp_dir, "result.json")
        cmd = [
            YOLORET_VENV_PYTHON,
            YOLORET_INFER_SCRIPT,
            "--image", os.path.abspath(image_path),
            "--weights", os.path.abspath(YOLORET_WEIGHTS),
            "--backbone", YOLORET_BACKBONE,
            "--classes_path", os.path.abspath(YOLORET_CLASSES),
            "--anchors_path", os.path.abspath(YOLORET_ANCHORS),
            "--input_size", str(YOLORET_INPUT_SIZE[0]), str(YOLORET_INPUT_SIZE[1]),
            "--score", str(score),
            "--nms", str(nms),
            "--out", out_json,
        ]
        proc = subprocess.run(cmd, cwd=YOLORET_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"YOLO-ReT subprocess failed:\n{proc.stderr[-4000:]}")
        with open(out_json) as f:
            return json.load(f)
