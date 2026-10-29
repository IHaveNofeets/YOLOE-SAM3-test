import base64
import json
import os

import cv2
import numpy as np
import requests

YOLOE_SERVICE_URL = os.environ.get("YOLOE_SERVICE_URL", "http://localhost:8600")
SAM3_SERVICE_URL = os.environ.get("SAM3_SERVICE_URL", "http://localhost:8700")
CLASSICAL_SERVICE_URL = os.environ.get("CLASSICAL_SERVICE_URL", "http://localhost:8800")
REFERENCE_SERVICE_URL = os.environ.get("REFERENCE_SERVICE_URL", "http://localhost:8900")

_PALETTE = [
    (0, 200, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255),
    (0, 220, 220), (200, 0, 0), (128, 0, 255), (0, 128, 255),
]


def draw_detections(image_rgb, detections):
    """Draw colored box + 'name conf' label per detection on a copy of image_rgb.

    Stands in for Ultralytics' Results.plot() now that predict() runs in a separate
    container and only structured JSON crosses the network - style matches the manual
    drawing already used elsewhere in this project (classical/YOLO-ReT step images)."""
    vis = image_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = (round(v) for v in d["bbox"])
        color = _PALETTE[int(d["cls"]) % len(_PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def _encode_masks(masks):
    """Encode a (N, H, W) float32 0/1 mask array as one base64 PNG per box, for JSON transport."""
    if masks is None:
        return None
    encoded = []
    for mask in masks:
        ok, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
        if not ok:
            raise RuntimeError("Failed to encode mask as PNG")
        encoded.append(base64.b64encode(buf).decode("ascii"))
    return encoded


def yoloe_available():
    try:
        return requests.get(f"{YOLOE_SERVICE_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def yoloe_bake(ref_image_path, bboxes, cls, names, masks=None):
    """POST the reference image + boxes/classes/names (and optional GrabCut masks) to the yoloe
    service's /bake endpoint. Raises RuntimeError on failure; returns the parsed JSON on success."""
    payload = {
        "bboxes": np.asarray(bboxes).tolist(),
        "cls": np.asarray(cls).tolist(),
        "names": list(names),
        "masks": _encode_masks(masks),
    }
    with open(ref_image_path, "rb") as f:
        resp = requests.post(
            f"{YOLOE_SERVICE_URL}/bake",
            files={"image": f},
            data={"payload": _json_dumps(payload)},
            timeout=300,
        )
    if not resp.ok:
        raise RuntimeError(f"YOLOE bake failed: {resp.status_code} {resp.text[:2000]}")
    result = resp.json()
    if result.get("status") != "ok":
        raise RuntimeError(f"YOLOE bake failed: {result.get('message', 'unknown error')}")
    return result


def yoloe_detect(image, conf):
    """Run YOLOE detection via the yoloe service. `image` may be a path or a BGR/RGB np.ndarray
    (ndarrays are PNG-encoded before sending). Returns a list of
    {"bbox": [x1,y1,x2,y2], "conf": float, "cls": int, "name": str}."""
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode image as PNG")
        image_bytes = buf.tobytes()
    else:
        with open(image, "rb") as f:
            image_bytes = f.read()

    resp = requests.post(
        f"{YOLOE_SERVICE_URL}/detect",
        files={"image": image_bytes},
        data={"conf": conf},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"YOLOE detect failed: {resp.status_code} {resp.text[:2000]}")
    return resp.json()["detections"]


def sam3_available():
    try:
        return requests.get(f"{SAM3_SERVICE_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def sam3_segment(image_path, bboxes):
    """Run SAM 3's box-exemplar mode via the sam3 service. Returns an annotated BGR np.ndarray."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SAM3_SERVICE_URL}/segment",
            files={"image": f},
            data={"bboxes": _json_dumps(np.asarray(bboxes).tolist())},
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"SAM 3 segment failed: {resp.status_code} {resp.text[:2000]}")
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)


def sam3_segment_like_yoloe(ref_image_path, ref_bboxes, target_image_path, height=720):
    """Run SAM 3 "like YOLOE" (reference boxes searched against the target image) via the sam3
    service. Returns an annotated BGR np.ndarray cropped to the target-image region."""
    with open(ref_image_path, "rb") as ref_f, open(target_image_path, "rb") as target_f:
        resp = requests.post(
            f"{SAM3_SERVICE_URL}/segment_like_yoloe",
            files={"ref_image": ref_f, "target_image": target_f},
            data={"ref_bboxes": _json_dumps(np.asarray(ref_bboxes).tolist()), "height": height},
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"SAM 3 segment_like_yoloe failed: {resp.status_code} {resp.text[:2000]}")
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)


def classical_available():
    try:
        return requests.get(f"{CLASSICAL_SERVICE_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def classical_detect(image_path):
    """Find candidate object boxes via the classical (HSV color-segmentation, no model) service.
    Returns a list of [x1, y1, x2, y2] boxes."""
    with open(image_path, "rb") as f:
        resp = requests.post(f"{CLASSICAL_SERVICE_URL}/detect", files={"image": f}, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Classical detect failed: {resp.status_code} {resp.text[:2000]}")
    return resp.json()["boxes"]


def _decode_mask_b64(b64):
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return (gray > 127).astype(np.float32)


def reference_available():
    try:
        return requests.get(f"{REFERENCE_SERVICE_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def reference_refine(image_path, bboxes):
    """Refine rough boxes into tighter GrabCut masks via the reference service. Returns
    (masks, preview_bgr) - masks is a (N, H, W) float32 0/1 np.ndarray, matching what
    box_refine.refine_to_masks() returns locally."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{REFERENCE_SERVICE_URL}/refine",
            files={"image": f},
            data={"bboxes": _json_dumps(np.asarray(bboxes).tolist())},
            timeout=60,
        )
    if not resp.ok:
        raise RuntimeError(f"Reference refine failed: {resp.status_code} {resp.text[:2000]}")
    result = resp.json()
    masks = np.stack([_decode_mask_b64(m) for m in result["masks"]])
    preview = cv2.imdecode(np.frombuffer(base64.b64decode(result["preview"]), np.uint8), cv2.IMREAD_COLOR)
    return masks, preview


def _json_dumps(obj):
    return json.dumps(obj)
