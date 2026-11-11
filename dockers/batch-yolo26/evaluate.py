"""Scoring detections against ground truth, threshold-free.

Matching is greedy in descending confidence (per image, per class, per IoU threshold): a detection
takes the unmatched ground-truth box it overlaps most, if that IoU reaches the threshold. Because a
detection's match only depends on the detections ranked above it, the matches of "all detections with
conf >= t" are the same for every t. So the matched list is stored once and precision / recall / F1 at
any confidence are computed from it afterwards (here and in the page) without running the models again.
"""
import numpy as np

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # COCO 0.50:0.95
CURVE_CONFS = [round(0.01 * i, 2) for i in range(1, 100)]       # 0.01 ... 0.99
CSV_CONFS = [0.05] + [round(0.1 * i, 1) for i in range(1, 10)]   # 0.05, 0.1 ... 0.9


def iou_matrix(a, b):
    """IoU between every box of a (N, 4 xyxy) and every box of b (M, 4)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(dets, gts, n_classes):
    """dets: [(box, conf, cls)], gts: [(box, cls, flags)] for one image.
    Returns per detection (in the given order): (tp bitmask over IOU_THRESHOLDS, flags of the ground
    truth matched at IoU 0.5 or -1)."""
    out = [(0, -1)] * len(dets)
    for c in range(n_classes):
        di = [i for i, d in enumerate(dets) if d[2] == c]
        gi = [j for j, g in enumerate(gts) if g[1] == c]
        if not di:
            continue
        di.sort(key=lambda i: -dets[i][1])
        ious = iou_matrix([dets[i][0] for i in di], [gts[j][0] for j in gi])
        result = {i: [0, -1] for i in di}
        for t_idx, thr in enumerate(IOU_THRESHOLDS):
            taken = np.zeros(len(gi), dtype=bool)
            for row, i in enumerate(di):
                if not len(gi):
                    break
                cand = np.where(~taken, ious[row], -1.0)
                j = int(np.argmax(cand))
                if cand[j] >= thr:
                    taken[j] = True
                    result[i][0] |= 1 << t_idx
                    if t_idx == 0:
                        result[i][1] = gts[gi[j]][2]
        for i, (bits, fl) in result.items():
            out[i] = (bits, fl)
    return out


def average_precision(tp, n_gt):
    """COCO 101-point interpolated AP from tp flags sorted by descending confidence."""
    if n_gt == 0:
        return None
    if not len(tp):
        return 0.0
    tp = np.asarray(tp, dtype=np.float64)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # precision envelope
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(recall, r, side="left")
        ap += precision[k] if k < len(precision) else 0.0
    return ap / 101


def prf_at(confs, tp50, n_gt, t):
    """Precision, recall, F1 and counts using only detections with conf >= t (confs sorted desc)."""
    k = int(np.searchsorted(-np.asarray(confs), -t, side="right"))  # number of detections with conf >= t
    tp = int(np.sum(tp50[:k])) if k else 0
    fp = k - tp
    fn = n_gt - tp
    p = tp / k if k else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_class(entries, n_gt):
    """entries: [(conf, tp_bits, matched_flags)] of one class over all images."""
    entries = sorted(entries, key=lambda e: -e[0])
    confs = np.array([e[0] for e in entries])
    bits = np.array([e[1] for e in entries], dtype=np.int64)
    aps = [average_precision((bits >> k) & 1, n_gt) for k in range(len(IOU_THRESHOLDS))]
    tp50 = (bits & 1).astype(np.int64)
    curve = [prf_at(confs, tp50, n_gt, t) for t in CURVE_CONFS]
    best_i = int(np.argmax([c["f1"] for c in curve])) if curve else 0
    return {
        "ap50": aps[0],
        "ap50_95": None if aps[0] is None else float(np.mean(aps)),
        "best": {"conf": CURVE_CONFS[best_i], **curve[best_i]},
        "curve": curve,
        "sorted": entries,
    }


def score_model(per_image, class_names, class_agnostic=False):
    """per_image: [{"dets": [(box, conf, cls)], "gts": [(box, cls, flags)]}]. Returns (metrics, matches,
    per-image match results), where matches is the compact list the page recomputes P/R/F1 from."""
    n_classes = 1 if class_agnostic else len(class_names)
    names = ["any object"] if class_agnostic else list(class_names)
    per_class = [[] for _ in range(n_classes)]
    n_gt = [0] * n_classes
    n_flag = [[0, 0] for _ in range(n_classes)]  # uncompleted, covered ground-truth counts
    image_matches = []
    for img in per_image:
        dets = [(b, s, 0 if class_agnostic else c) for b, s, c in img["dets"]]
        gts = [(b, 0 if class_agnostic else c, fl) for b, c, fl in img["gts"]]
        for _, c, fl in gts:
            n_gt[c] += 1
            n_flag[c][0] += bool(fl & 1)
            n_flag[c][1] += bool(fl & 2)
        m = match_image(dets, gts, n_classes)
        image_matches.append(m)
        for (b, s, c), (bits, fl) in zip(dets, m):
            per_class[c].append((float(s), int(bits), int(fl)))

    classes, matches = [], []
    for c in range(n_classes):
        sc = score_class(per_class[c], n_gt[c])
        classes.append({"name": names[c], "n_gt": n_gt[c], "n_gt_uncompleted": n_flag[c][0],
                        "n_gt_covered": n_flag[c][1], "n_det": len(per_class[c]),
                        "ap50": sc["ap50"], "ap50_95": sc["ap50_95"], "best": sc["best"],
                        "curve": compact_curve(sc["curve"]), **flagged_recall(sc["sorted"], n_flag[c], sc["best"]["conf"])})
        matches.append({"name": names[c], "n_gt": n_gt[c], "n_gt_uncompleted": n_flag[c][0], "n_gt_covered": n_flag[c][1],
                        "d": [[round(s, 4), bits & 1, fl] for s, bits, fl in sc["sorted"]]})

    # whole model: AP averaged over the classes that have ground truth, P/R/F1 pooled over all classes
    with_gt = [c for c in classes if c["n_gt"]]
    pooled = sorted([e for pc in per_class for e in pc], key=lambda e: -e[0])
    pooled_sc = score_class(pooled, sum(n_gt)) if with_gt else None
    model = {
        "ap50": float(np.mean([c["ap50"] for c in with_gt])) if with_gt else None,
        "ap50_95": float(np.mean([c["ap50_95"] for c in with_gt])) if with_gt else None,
        "n_gt": sum(n_gt),
        "n_det": len(pooled),
        "best": pooled_sc["best"] if pooled_sc else None,
        "curve": compact_curve(pooled_sc["curve"]) if pooled_sc else [],
        **(flagged_recall(pooled, [sum(f[0] for f in n_flag), sum(f[1] for f in n_flag)], pooled_sc["best"]["conf"])
           if pooled_sc else {}),
        "class_agnostic": class_agnostic,
        "classes": classes,
    }
    return model, matches, image_matches


def flagged_recall(sorted_entries, n_flag, conf):
    """Recall on uncompleted and on covered ground truth at a confidence."""
    hit_u = sum(1 for s, bits, fl in sorted_entries if s >= conf and bits & 1 and fl >= 0 and fl & 1)
    hit_c = sum(1 for s, bits, fl in sorted_entries if s >= conf and bits & 1 and fl >= 0 and fl & 2)
    return {"recall_uncompleted": hit_u / n_flag[0] if n_flag[0] else None,
            "recall_covered": hit_c / n_flag[1] if n_flag[1] else None}


def compact_curve(curve):
    return [[t, round(c["p"], 4), round(c["r"], 4), round(c["f1"], 4)] for t, c in zip(CURVE_CONFS, curve)]
