def iou(a, b):
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_regions(result, iou_thresh=0.9):
    """Keep only the highest-confidence label per physical region, operating on an Ultralytics
    Results object (used by main.py, which runs YOLOE in-process and never crosses the network).

    YOLOE's visual-prompt head scores each detected region against every baked class
    independently, so one object can clear the confidence bar for more than one class at
    once (e.g. a brick scoring high for both object0 and object1). Ultralytics' own NMS
    treats that as intentional multi-label output and won't collapse it, so this does a
    simple greedy pass instead: sort by confidence, keep a box only if it doesn't heavily
    overlap one already kept.
    """
    order = result.boxes.conf.argsort(descending=True).tolist()
    xyxy = result.boxes.xyxy.tolist()
    keep = []
    for i in order:
        if all(iou(xyxy[i], xyxy[j]) < iou_thresh for j in keep):
            keep.append(i)
    return result[sorted(keep)]


def dedupe_detections(detections, iou_thresh=0.9):
    """Same greedy dedup as dedupe_regions(), but for the plain list[dict] shape returned by the
    yoloe service's /detect endpoint ({"bbox": [x1,y1,x2,y2], "conf": float, "cls": int, "name":
    str}) - used by main_web.py/compare_objects.py, which reach YOLOE over the network and never
    hold an Ultralytics Results object."""
    order = sorted(range(len(detections)), key=lambda i: detections[i]["conf"], reverse=True)
    keep = []
    for i in order:
        if all(iou(detections[i]["bbox"], detections[j]["bbox"]) < iou_thresh for j in keep):
            keep.append(i)
    return [detections[i] for i in sorted(keep)]


def filter_and_dedupe_detections(detections, conf_thresh, iou_thresh=0.9):
    """Derive the final, thresholded detections from one low-confidence raw prediction: keep only
    detections at or above conf_thresh, then dedupe_detections() the rest. Reuses a single raw
    predict() call for both a "see everything" diagnostic view and the real filtered result,
    instead of predicting twice at two different confidence levels."""
    kept = [d for d in detections if d["conf"] >= conf_thresh]
    return dedupe_detections(kept, iou_thresh=iou_thresh)
