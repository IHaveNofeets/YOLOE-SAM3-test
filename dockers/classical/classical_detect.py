import cv2
import numpy as np


def detect_objects(image_path, hsv_lo=(80, 25, 60), hsv_hi=(115, 255, 255), min_area=400, pad=4):
    """Classical CV object detection via color segmentation - no model, no learning.

    Targets the objects' own hue directly (tuned for the pale cyan/blue LEGO bricks against a grey
    desk and a black keyboard/chair) rather than generic local contrast - much more robust for this
    scene, since color separates the objects from both distractors in one step instead of needing to
    fight their edges. Returns one bounding box per resulting blob above min_area (in pixels).
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # bridges gaps between studs

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, num_labels):  # label 0 is the background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        boxes.append([int(max(0, x - pad)), int(max(0, y - pad)), int(min(w, x + bw + pad)), int(min(h, y + bh + pad))])
    return boxes
