import threading

import cv2
import numpy as np

from ultralytics import SAM

_sam = None  # loaded once and reused by detect_target_like_yoloe()
_sam_weights = None
_sam_lock = threading.Lock()  # one predictor object is not thread-safe


def segment_with_sam3(image, bboxes, weights="sam3.pt"):
    """Run SAM 3's box-exemplar mode: given box(es), find similar instances across that image.

    `image` may be a path or an array. SAM 3 has no cross-image exemplar support (unlike YOLOE's
    refer_image) - a box only finds matches within the same image it was drawn on.
    """
    sam = SAM(weights)
    results = sam.predict(image, bboxes=bboxes, verbose=False)
    return results[0]


def target_boxes(xyxy, conf, offset, target_scale, width, height):
    """Map SAM 3 boxes found on the reference|target composite back to target-image pixels.

    Boxes whose centre lies in the reference half (x < offset) are matches inside the reference
    image and are dropped; the rest are shifted by the reference width, undone from the resize and
    clipped to the target image. Returns [(x1, y1, x2, y2, conf), ...]."""
    out = []
    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        if (x1 + x2) / 2 < offset:
            continue
        out.append((
            float(np.clip((x1 - offset) / target_scale, 0, width)),
            float(np.clip(y1 / target_scale, 0, height)),
            float(np.clip((x2 - offset) / target_scale, 0, width)),
            float(np.clip(y2 / target_scale, 0, height)),
            float(c),
        ))
    return out


def detect_target_like_yoloe(ref_image_path, ref_bboxes, target_image_path, weights="sam3.pt", height=720):
    """Same composite trick as segment_target_like_yoloe(), but returns the matches as data:
    (boxes in target-image pixels [(x1, y1, x2, y2, conf), ...], target width, target height)."""
    global _sam, _sam_weights
    ref_img = cv2.imread(ref_image_path)
    target_img = cv2.imread(target_image_path)
    th, tw = target_img.shape[:2]

    ref_scale = height / ref_img.shape[0]
    target_scale = height / th
    ref_fit = cv2.resize(ref_img, (round(ref_img.shape[1] * ref_scale), height))
    target_fit = cv2.resize(target_img, (round(tw * target_scale), height))
    composite = cv2.hconcat([ref_fit, target_fit])
    offset = ref_fit.shape[1]
    scaled_bboxes = np.asarray(ref_bboxes, dtype=np.float32) * ref_scale

    with _sam_lock:
        if _sam is None or _sam_weights != weights:
            _sam, _sam_weights = SAM(weights), weights
        result = _sam.predict(composite, bboxes=scaled_bboxes, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return [], tw, th
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return target_boxes(xyxy, conf, offset, target_scale, tw, th), tw, th


def segment_target_like_yoloe(ref_image_path, ref_bboxes, target_image_path, weights="sam3.pt", height=720):
    """Run SAM 3 "like YOLOE": prompted with the reference boxes, searching the target image.

    Works around SAM 3 only supporting same-image exemplars by concatenating the reference and
    target images side by side and prompting on that composite - the target region is then part
    of "the same image" SAM 3 searches. Returns the annotated result cropped back down to just the
    target-image region, sized to `height`.
    """
    ref_img = cv2.imread(ref_image_path)
    target_img = cv2.imread(target_image_path)

    ref_scale = height / ref_img.shape[0]
    target_scale = height / target_img.shape[0]
    ref_fit = cv2.resize(ref_img, (round(ref_img.shape[1] * ref_scale), height))
    target_fit = cv2.resize(target_img, (round(target_img.shape[1] * target_scale), height))
    composite = cv2.hconcat([ref_fit, target_fit])
    offset = ref_fit.shape[1]

    scaled_bboxes = np.asarray(ref_bboxes, dtype=np.float32) * ref_scale
    result = segment_with_sam3(composite, scaled_bboxes, weights=weights)

    vis = result.plot()
    return vis[:, offset:]
