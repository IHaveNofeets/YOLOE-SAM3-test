import cv2
import numpy as np


def _largest_component(mask_u8):
    """Keep only the largest connected blob, dropping any stray disconnected specks."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _seed_mask(v_channel, x1, y1, x2, y2, h, w, shadow_ratio=0.95):
    """Build GrabCut's initial mask, pre-marking dark pixels inside the box as probable
    background. A plain rect-init treats the whole box interior as probable foreground,
    which biases GrabCut's color model to absorb a shadow instead of cutting it away."""
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    pad = max(15, round(0.25 * min(x2 - x1, y2 - y1)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = v_channel[ry1:ry2, rx1:rx2].astype(np.float32)
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False  # exclude the box itself
    if not ring_mask.any():
        return mask
    bg_v = np.median(ring[ring_mask])

    box_v = v_channel[y1:y2, x1:x2].astype(np.float32)
    shadow = box_v < bg_v * shadow_ratio

    # keep a guaranteed foreground core so GrabCut always has a solid seed to build a color model from
    cy0, cy1 = round((y2 - y1) * 0.25), round((y2 - y1) * 0.75)
    cx0, cx1 = round((x2 - x1) * 0.25), round((x2 - x1) * 0.75)
    shadow[cy0:cy1, cx0:cx1] = False

    region = mask[y1:y2, x1:x2]
    region[shadow] = cv2.GC_PR_BGD
    mask[y1:y2, x1:x2] = region
    return mask


def refine_to_masks(image_path, bboxes, iterations=5):
    """Refine rough boxes into tighter foreground masks using GrabCut, seeded with a
    brightness cue so a drop shadow next to the object starts out as probable background
    rather than getting absorbed into the foreground.

    Classic OpenCV segmentation - no model weights or downloads needed. Returns the
    per-box masks plus a preview image with the refined outlines drawn on it.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    v_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    masks = np.zeros((len(bboxes), h, w), dtype=np.float32)
    overlay = img.copy()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        gc_mask = _seed_mask(v_channel, x1, y1, x2, y2, h, w)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(img, gc_mask, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
        fg_u8 = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_u8 = _largest_component(fg_u8)
        fg = (fg_u8 > 0).astype(np.float32)
        masks[i] = fg

        tint = np.zeros_like(img)
        tint[:, :, 1] = (fg * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.4, 0)
        contours, _ = cv2.findContours((fg * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)

    return masks, overlay
