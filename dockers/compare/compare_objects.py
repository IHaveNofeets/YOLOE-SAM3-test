import base64
import os

import cv2
from flask import Flask, request

import service_clients as svc
from detect_utils import dedupe_detections

ASSETS_DIR = "./assets/"
TARGET_IMAGE = ASSETS_DIR + "input/3.jpg"
CONF_THRESHOLD = 0.28
DEDUPE_IOU = 0.9
CROP_PAD = 20  # extra pixels of context kept around each YOLOE-detected box


def _encode_png(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


def _crop(img, x1, y1, x2, y2, pad=CROP_PAD):
    h, w = img.shape[:2]
    x1, y1 = max(0, round(x1) - pad), max(0, round(y1) - pad)
    x2, y2 = min(w, round(x2) + pad), min(h, round(y2) + pad)
    return img[y1:y2, x1:x2]


def build_cards_yoloe():
    """Run YOLOE once, then SAM 3 once per YOLOE-detected box, and build one card per object."""
    detections = svc.yoloe_detect(TARGET_IMAGE, conf=CONF_THRESHOLD)
    detections = dedupe_detections(detections, iou_thresh=DEDUPE_IOU)
    target_bgr = cv2.imread(TARGET_IMAGE)
    yoloe_vis = svc.draw_detections(target_bgr, detections)

    sam_ok = bool(detections) and svc.sam3_available()

    cards = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        card = {
            "name": d["name"],
            "conf": d["conf"],
            "yoloe_b64": _encode_png(_crop(yoloe_vis, x1, y1, x2, y2)),
            "sam_b64": None,
        }
        if sam_ok:
            # SAM 3's native same-image exemplar mode: prompted with this one box, confirms/
            # refines just that region, independent of - and directly comparable to - YOLOE's call.
            sam_result = svc.sam3_segment(TARGET_IMAGE, [[x1, y1, x2, y2]])
            card["sam_b64"] = _encode_png(_crop(sam_result, x1, y1, x2, y2))
        cards.append(card)
    return cards


def build_cards_classical():
    """Find candidate objects with plain OpenCV (no model), then ask YOLOE and SAM 3 what each one
    is individually - unlike the 'yoloe' mode, detection (where objects are) and identification
    (what they are) are fully decoupled, so this also surfaces objects YOLOE's own detector missed."""
    target = cv2.imread(TARGET_IMAGE)
    boxes = svc.classical_detect(TARGET_IMAGE)
    sam_ok = bool(boxes) and svc.sam3_available()

    cards = []
    for x1, y1, x2, y2 in boxes:
        crop = _crop(target, x1, y1, x2, y2, pad=0)  # detect_objects already padded the box itself
        detections = svc.yoloe_detect(crop, conf=CONF_THRESHOLD)
        if detections:
            best = max(detections, key=lambda d: d["conf"])
            name, conf = best["name"], best["conf"]
            yoloe_crop_vis = svc.draw_detections(crop, detections)
        else:
            name, conf = "no YOLOE match", 0.0
            yoloe_crop_vis = crop

        card = {"name": name, "conf": conf, "yoloe_b64": _encode_png(yoloe_crop_vis), "sam_b64": None}
        if sam_ok:
            sam_result = svc.sam3_segment(TARGET_IMAGE, [[x1, y1, x2, y2]])
            card["sam_b64"] = _encode_png(_crop(sam_result, x1, y1, x2, y2))
        cards.append(card)
    return cards


app = Flask(__name__)
_cards_by_mode = {}


@app.route("/")
def index():
    mode = request.args.get("mode", "yoloe")
    if mode not in ("yoloe", "classical"):
        mode = "yoloe"

    if mode not in _cards_by_mode:
        print(f"Running detection ({mode})...")
        _cards_by_mode[mode] = build_cards_yoloe() if mode == "yoloe" else build_cards_classical()
        print(f"Found {len(_cards_by_mode[mode])} object(s).")

    cards = _cards_by_mode[mode]

    card_html = "\n".join(
        f"""
        <div class="card">
          <h2>{c["name"]} <span class="conf">{c["conf"]:.2f}</span></h2>
          <div class="row">
            <div class="col">
              <div class="label">YOLOE</div>
              <img src="data:image/png;base64,{c["yoloe_b64"]}">
            </div>
            {
            f'''<div class="col">
              <div class="label">SAM 3</div>
              <img src="data:image/png;base64,{c["sam_b64"]}">
            </div>'''
            if c["sam_b64"]
            else '<div class="col"><div class="label">SAM 3</div><p class="missing">not available</p></div>'
        }
          </div>
        </div>"""
        for c in cards
    )

    def nav_link(target_mode, label):
        cls = "active" if mode == target_mode else ""
        return f'<a class="{cls}" href="/?mode={target_mode}">{label}</a>'

    return f"""<!doctype html>
<html>
<head>
<title>YOLOE vs SAM 3 - per-object comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1c1c1e; color: #eee; margin: 0; padding: 24px; }}
  h1 {{ font-weight: 600; }}
  nav {{ margin-bottom: 20px; }}
  nav a {{ color: #999; text-decoration: none; margin-right: 16px; padding: 6px 12px; border-radius: 6px; }}
  nav a.active {{ background: #2a2a2d; color: #8ab4f8; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
  .card {{ background: #2a2a2d; border-radius: 10px; padding: 16px; width: 420px; }}
  .card h2 {{ margin: 0 0 12px; font-size: 18px; }}
  .conf {{ color: #8ab4f8; font-weight: 400; font-size: 14px; }}
  .row {{ display: flex; gap: 12px; }}
  .col {{ flex: 1; }}
  .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #999; margin-bottom: 6px; }}
  img {{ width: 100%; border-radius: 6px; display: block; }}
  .missing {{ color: #777; font-style: italic; }}
</style>
</head>
<body>
  <h1>YOLOE vs SAM 3 - {len(cards)} object(s) in {os.path.basename(TARGET_IMAGE)}</h1>
  <nav>
    {nav_link("yoloe", "YOLOE detections")}
    {nav_link("classical", "Classical detections")}
  </nav>
  <div class="grid">
    {card_html if cards else "<p>No detections.</p>"}
  </div>
</body>
</html>"""


if __name__ == "__main__":
    print("Serving at http://127.0.0.1:5000 (detection runs lazily per mode on first visit)")
    app.run(host="0.0.0.0", port=5000, debug=False)
