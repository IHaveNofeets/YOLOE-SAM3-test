import os
import time
from datetime import datetime

import cv2
import gradio as gr
import numpy as np

import service_clients as svc
from detect_utils import filter_and_dedupe_detections
from yoloret_predict import detect_with_yoloret, yoloret_available

ASSETS_DIR = "./assets/"
MODEL_DIR = "./modelData/"
CURRENT_DIR = MODEL_DIR + "currentModels/"  # the baked/exported model + its reference records
OUTPUT_DIR = ASSETS_DIR + "output/"
EXPORTED_MODEL = CURRENT_DIR + "yoloe-26l-seg.onnx"  # produced by create_reference(), via the yoloe service
BBOX_RECORD = CURRENT_DIR + "lego_bboxes.npy"
BOX_NAMES_RECORD = CURRENT_DIR + "lego_box_names.npy"
BOX_CLS_RECORD = CURRENT_DIR + "lego_box_cls.npy"  # remapped int class per box, kept for reference display
REFINE_PREVIEW = CURRENT_DIR + "refine_preview.png"
REF_IMAGE_TMP = MODEL_DIR + "web_ref.jpg"
TARGET_IMAGE_TMP = MODEL_DIR + "web_target.jpg"
CONF_THRESHOLD = 0.28
DEDUPE_IOU = 0.9  # above this overlap, treat two detections as the same physical region
RAW_CONF = 0.01  # near-zero floor for the undeduped "every class match" diagnostic step


def clean_rows(rows):
    """Drop incomplete/malformed rows - e.g. a freshly-added blank row, or a cell mid-edit -
    that the editable Dataframe can pass through with empty-string or missing cells."""
    clean = []
    for row in rows or []:
        try:
            x1, y1, x2, y2, cls = (float(v) for v in row)
        except (TypeError, ValueError):
            continue
        clean.append([x1, y1, x2, y2, cls])
    return clean


def render_boxes(img, rows, pending=None):
    """Draw the current box(es) - and, mid-box, the first clicked corner - on a copy of img (RGB)."""
    vis = img.copy()
    for x1, y1, x2, y2, cls in clean_rows(rows):
        x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f"cls {cls}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    if pending is not None:
        cv2.circle(vis, (int(pending[0]), int(pending[1])), 5, (255, 0, 0), -1)
    return vis


def on_ref_upload(image):
    """New reference image uploaded/captured: reset boxes and pending click."""
    return image, [], None, image


def on_ref_click(ref_image, rows, pending, evt: gr.SelectData):
    """First click on the reference image sets one corner; the second completes the box."""
    if ref_image is None:
        return rows, pending, ref_image
    rows = clean_rows(rows)
    x, y = evt.index
    if pending is None:
        return rows, (x, y), render_boxes(ref_image, rows, pending=(x, y))
    x1, y1 = pending
    x2, y2 = x, y
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), len(rows)]  # new box defaults to its own class
    rows = rows + [box]
    return rows, None, render_boxes(ref_image, rows, pending=None)


def on_rows_change(ref_image, rows):
    """Dataframe edited by hand (e.g. 'cls' changed to group boxes) - redraw to match."""
    if ref_image is None:
        return None
    return render_boxes(ref_image, rows or [], pending=None)


def undo_box(ref_image, rows):
    rows = clean_rows(rows)[:-1]
    return rows, render_boxes(ref_image, rows, pending=None) if ref_image is not None else None


def clear_boxes(ref_image):
    return [], None, ref_image


def annotate_classical(target_bgr, boxes):
    """For each classically-found box: YOLOE(raw, conf=RAW_CONF) on just that crop -> final YOLOE
    (best class at or above CONF_THRESHOLD, else 'no match') - then draw the final call onto a
    copy of the full target image."""
    vis = target_bgr.copy()
    log_lines = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        crop = target_bgr[y1:y2, x1:x2]
        detections = svc.yoloe_detect(crop, conf=RAW_CONF)
        above_thresh = [d for d in detections if d["conf"] >= CONF_THRESHOLD]
        if above_thresh:
            best = max(above_thresh, key=lambda d: d["conf"])
            name, conf, color = best["name"], best["conf"], (0, 200, 0)
        else:
            name, conf, color = "no match", 0.0, (0, 0, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(vis, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        log_lines.append(f"box{i}: {label}  bbox=({x1}, {y1}, {x2}, {y2})")
    return vis, log_lines


def _load_reference_arrays():
    """Load persisted reference boxes/classes/rows from disk (as saved by create_reference()).
    Returns (bboxes, cls, class_names, rows), all None if nothing has been baked yet."""
    if not os.path.exists(BBOX_RECORD):
        return None, None, None, None
    bboxes = np.load(BBOX_RECORD)
    cls = np.load(BOX_CLS_RECORD) if os.path.exists(BOX_CLS_RECORD) else np.zeros(len(bboxes), dtype=int)
    num_cls = int(cls.max()) + 1 if len(cls) else 0
    class_names = [f"object{c}" for c in range(num_cls)]
    rows = [[*bboxes[i].tolist(), int(cls[i])] for i in range(len(bboxes))]
    return bboxes, cls, class_names, rows


def _build_reference_step_image():
    """Rebuild the annotated Reference step image straight from disk (the GrabCut-refined preview
    when one exists, else the plain reference image, plus the saved boxes) - or None if nothing has
    been baked yet."""
    if not os.path.exists(BBOX_RECORD):
        return None
    if os.path.exists(REFINE_PREVIEW):
        base = cv2.cvtColor(cv2.imread(REFINE_PREVIEW), cv2.COLOR_BGR2RGB)
    elif os.path.exists(REF_IMAGE_TMP):
        base = cv2.cvtColor(cv2.imread(REF_IMAGE_TMP), cv2.COLOR_BGR2RGB)
    else:
        return None
    _, _, _, rows = _load_reference_arrays()
    return render_boxes(base, rows or [], pending=None)


def _panel(img, title, height=400):
    """Resize to a common height and stamp a title bar on top, for a saved side-by-side strip.
    Operates on RGB arrays (this file's convention throughout)."""
    scale = height / img.shape[0]
    img = cv2.resize(img, (round(img.shape[1] * scale), height))
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def create_reference(ref_image, rows, use_refine, progress=gr.Progress()):
    """Bake the reference image + boxes into a fixed-class YOLOE model and export it. Always
    (re)creates and overwrites the exported model when clicked - there is no reuse toggle; the
    Target tab's Run always just uses whatever this last produced."""
    rows = clean_rows(rows)
    if ref_image is None or not rows:
        raise gr.Error("Upload a reference image and draw at least one box on it first.")

    t_start = time.perf_counter()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CURRENT_DIR, exist_ok=True)
    cv2.imwrite(REF_IMAGE_TMP, cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))

    # cls holds one class index per box, assigned by drag-order or edited in the table
    # (boxes sharing a value are the same class); remap to a clean, contiguous 0..N-1 range.
    bboxes = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32)
    cls = np.unique(np.array([int(r[4]) for r in rows]), return_inverse=True)[1]
    num_cls = int(cls.max()) + 1
    class_names = [f"object{c}" for c in range(num_cls)]

    img_h, img_w = ref_image.shape[:2]
    if (bboxes[:, 0] < 0).any() or (bboxes[:, 1] < 0).any() or (bboxes[:, 2] > img_w).any() or (bboxes[:, 3] > img_h).any():
        raise gr.Error(
            "Box coordinates fall outside the reference image - this can happen if boxes were saved "
            "against a different-sized image. Clear the boxes and redraw them on the current image."
        )

    if use_refine and not svc.reference_available():
        return "Failed to create reference: the 'reference' service isn't reachable - make sure that container is running.", gr.update()
    if not svc.yoloe_available():
        return "Failed to create reference: the 'yoloe' service isn't reachable - make sure that container is running.", gr.update()

    np.save(BBOX_RECORD, bboxes)
    np.save(BOX_NAMES_RECORD, np.array([class_names[c] for c in cls]))
    np.save(BOX_CLS_RECORD, cls)

    try:
        progress(0.1, desc="Refining..." if use_refine else "Baking...")
        masks = None
        if use_refine:
            masks, refine_vis = svc.reference_refine(REF_IMAGE_TMP, bboxes)
            cv2.imwrite(REFINE_PREVIEW, refine_vis)
        elif os.path.exists(REFINE_PREVIEW):
            os.remove(REFINE_PREVIEW)  # stale preview from a previous run that did refine

        progress(0.3, desc="Baking + exporting (yoloe service)...")
        svc.yoloe_bake(REF_IMAGE_TMP, bboxes, cls, class_names, masks=masks)
    except Exception as e:
        return f"Failed to create reference: {e}", gr.update()

    progress(1.0, desc="Done")
    elapsed = time.perf_counter() - t_start
    status = f"Baked {len(rows)} box(es), {num_cls} class(es) in {elapsed:.1f}s. Exported model ready."
    ref_preview = _build_reference_step_image() if use_refine else gr.update()
    return status, ref_preview


def run_pipeline(target_image, use_sam3, use_yoloret, detection_source, progress=gr.Progress()):
    if target_image is None:
        raise gr.Error("Upload or capture a target/input image first.")
    if not os.path.exists(EXPORTED_MODEL):
        raise gr.Error(
            "No baked model found yet - go to the Reference tab, draw box(es), and click "
            "'Create reference' first."
        )
    if not svc.yoloe_available():
        raise gr.Error("The 'yoloe' service isn't reachable - make sure that container is running.")

    bboxes, _, _, _ = _load_reference_arrays()

    timings = {}
    t_total = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(TARGET_IMAGE_TMP, cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))

    # Classical detections step - always computed (cheap, no model), so it's visible regardless
    # of which detection source actually drives the Output panel below.
    progress(0.15, desc="Finding objects with classical detection...")
    t0 = time.perf_counter()
    if svc.classical_available():
        candidate_boxes = svc.classical_detect(TARGET_IMAGE_TMP)
    else:
        print("Classical service not available - skipping classical detection step.")
        candidate_boxes = []
    classical_bgr = cv2.imread(TARGET_IMAGE_TMP)
    for i, (x1, y1, x2, y2) in enumerate(candidate_boxes):
        cv2.rectangle(classical_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(classical_bgr, f"box{i}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    classical_vis = cv2.cvtColor(classical_bgr, cv2.COLOR_BGR2RGB)
    timings["Classical"] = time.perf_counter() - t0

    # YOLOE(raw, conf=RAW_CONF) on the whole image - always computed too, and reused below as the
    # "YOLOE detections" mode's own raw step (one predict call feeds both the diagnostic view and
    # the final, thresholded result - no redundant second predict at a different confidence).
    progress(0.3, desc="Running YOLOE (raw)...")
    t0 = time.perf_counter()
    raw_detections = svc.yoloe_detect(TARGET_IMAGE_TMP, conf=RAW_CONF)
    yoloe_raw_vis = svc.draw_detections(target_image, raw_detections)
    timings["YOLOE raw"] = time.perf_counter() - t0

    # YOLO-ReT, run independently on the target image - unlike YOLOE/SAM 3, it's a fixed-vocabulary
    # (COCO) detector with no visual prompting, so it needs no reference boxes at all. Guarded so a
    # missing/broken venv degrades to "skipped" instead of crashing the whole run.
    yoloret_vis = None
    timings["YOLO-ReT"] = None
    if use_yoloret:
        if not yoloret_available():
            print("YOLO-ReT requested but not available (venv/weights missing) - skipping.")
        else:
            progress(0.45, desc="Running YOLO-ReT (separate TF process, this can take a while)...")
            t0 = time.perf_counter()
            try:
                yoloret_vis = cv2.cvtColor(detect_with_yoloret(TARGET_IMAGE_TMP), cv2.COLOR_BGR2RGB)
                timings["YOLO-ReT"] = time.perf_counter() - t0
            except (RuntimeError, OSError) as e:
                print(f"YOLO-ReT failed, skipping: {e}")

    sam3_vis = None
    timings["SAM 3"] = None
    if detection_source == "Classical detections":
        # Classical: input -> Classical detections -> (per bbox) YOLOE(raw, conf=RAW_CONF) -> final YOLOE.
        # Detection (where objects are) and identification (what they are) are fully decoupled
        # here: plain OpenCV finds candidate regions with no model at all, then YOLOE is asked
        # individually about each crop - this also surfaces objects YOLOE's own detector missed.
        progress(0.6, desc="Asking YOLOE about each classical candidate...")
        t0 = time.perf_counter()
        target_bgr = cv2.imread(TARGET_IMAGE_TMP)
        yoloe_vis_bgr, log_lines = annotate_classical(target_bgr, candidate_boxes)
        yoloe_vis = cv2.cvtColor(yoloe_vis_bgr, cv2.COLOR_BGR2RGB)
        timings["Final YOLOE"] = time.perf_counter() - t0

        if use_sam3 and svc.sam3_available() and candidate_boxes:
            progress(0.85, desc="Running SAM 3 (this can take a while)...")
            t0 = time.perf_counter()
            sam3_vis = cv2.cvtColor(svc.sam3_segment(TARGET_IMAGE_TMP, candidate_boxes), cv2.COLOR_BGR2RGB)
            timings["SAM 3"] = time.perf_counter() - t0
    else:
        # YOLOE: input -> YOLOE(raw, conf=RAW_CONF) -> final YOLOE.
        t0 = time.perf_counter()
        detections = filter_and_dedupe_detections(raw_detections, CONF_THRESHOLD, iou_thresh=DEDUPE_IOU)
        log_lines = [
            f"{d['name']}: conf={d['conf']:.2f}  "
            f"bbox=({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})"
            for d in detections
        ]
        yoloe_vis = svc.draw_detections(target_image, detections)
        timings["Final YOLOE"] = time.perf_counter() - t0

        if use_sam3 and svc.sam3_available() and bboxes is not None and os.path.exists(REF_IMAGE_TMP):
            progress(0.85, desc="Running SAM 3 (this can take a while)...")
            t0 = time.perf_counter()
            sam3_vis = cv2.cvtColor(
                svc.sam3_segment_like_yoloe(REF_IMAGE_TMP, bboxes, TARGET_IMAGE_TMP),
                cv2.COLOR_BGR2RGB,
            )
            timings["SAM 3"] = time.perf_counter() - t0

    ref_vis = _build_reference_step_image()

    # Save every step image - individually, and as one combined strip - under a shared run id so
    # repeated runs don't overwrite each other's output (the old fixed filename did).
    progress(0.95, desc="Saving steps...")
    steps = [
        ("reference", "Reference", ref_vis),
        ("input", "Input", target_image),
        ("classical", "Classical", classical_vis),
        ("yoloe_raw", "YOLOE raw", yoloe_raw_vis),
        ("output_yoloe", "Output YOLOE", yoloe_vis),
        ("output_sam3", "Output SAM 3", sam3_vis),
        ("output_yoloret", "Output YOLO-ReT", yoloret_vis),
    ]
    for name, _title, img in steps:
        if img is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{run_id}_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    panels = [_panel(img, title) for _name, title, img in steps if img is not None]
    if panels:
        combined = cv2.hconcat(panels)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{run_id}_all_steps.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    timings["Total"] = time.perf_counter() - t_total
    timing_lines = [f"{name}: {'skipped' if secs is None else f'{secs:.2f}s'}" for name, secs in timings.items()]

    log_text = "\n".join(log_lines) or "No detections."
    return (
        ref_vis,
        target_image,
        yoloe_vis,
        yoloe_vis,
        sam3_vis,
        yoloret_vis,
        classical_vis,
        yoloe_raw_vis,
        log_text,
        "\n".join(timing_lines),
    )


with gr.Blocks(title="YOLOE visual-prompt detector") as demo:
    gr.Markdown("# YOLOE visual-prompt detector")

    state_ref_image = gr.State(None)
    state_pending = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Reference"):
            gr.Markdown(
                "Upload a reference image, **click two opposite corners** to draw a box around each object "
                "(each new box defaults to its own class - edit the **cls** column below to group boxes into "
                "the same class), then click **Create reference** to bake and export the model."
            )
            with gr.Row():
                with gr.Column():
                    ref_img = gr.Image(
                        label="Reference image (click to draw boxes)", type="numpy", sources=["upload", "webcam"]
                    )

                with gr.Column():
                    gr.Markdown("### Settings")
                    refine_cb = gr.Checkbox(
                        label="Refine box(es) into tighter masks using GrabCut before baking", value=False
                    )
                    create_ref_btn = gr.Button("Create reference", variant="primary")
                    create_ref_status = gr.Textbox(label="Status", lines=2, interactive=False)

                    gr.Markdown("### Boxes")
                    boxes_table = gr.Dataframe(
                        headers=["x1", "y1", "x2", "y2", "cls"],
                        datatype=["number"] * 5,
                        type="array",
                        row_count=(0, "dynamic"),
                        interactive=True,
                        label="Boxes - edit 'cls' to group boxes into the same class",
                    )
                    with gr.Row():
                        undo_btn = gr.Button("Undo last box")
                        clear_btn = gr.Button("Clear boxes")

        with gr.Tab("Target"):
            with gr.Row():
                source_radio = gr.Radio(
                    ["YOLOE detections", "Classical detections"],
                    value="YOLOE detections",
                    label="Detection source",
                    info=(
                        "YOLOE detections: YOLOE finds and classifies objects in one pass. "
                        "Classical detections: plain OpenCV (no model) finds candidate objects first, "
                        "then YOLOE is asked individually about each one - this also reveals objects "
                        "YOLOE's own detector would otherwise miss."
                    ),
                )
                sam3_cb = gr.Checkbox(
                    label="Also run SAM 3 for an independent comparison (slow, needs modelData/sam3.pt)",
                    value=False,
                )
                yoloret_cb = gr.Checkbox(
                    label=(
                        "Also run YOLO-ReT for an independent comparison "
                        "(slow, runs in a separate TF venv - see modelData/yoloret)"
                    ),
                    value=False,
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Row():
                with gr.Column():
                    target_img = gr.Image(label="Target / input image", type="numpy", sources=["upload", "webcam"])
                with gr.Column():
                    yoloe_out = gr.Image(label="Output YOLOE")

        with gr.Tab("Result"):
            gr.Markdown("### Steps")
            with gr.Row():
                ref_out = gr.Image(label="Reference", height=220)
                input_out = gr.Image(label="Input", height=220)
                classical_out = gr.Image(label="Classical (raw candidate boxes)", height=220)
                yoloe_raw_out = gr.Image(label=f"YOLOE (raw, every class, conf ≥ {RAW_CONF})", height=220)
                yoloe_out2 = gr.Image(label="Output YOLOE", height=220)
                sam3_out = gr.Image(label="Output SAM 3", height=220)
                yoloret_out = gr.Image(label="Output YOLO-ReT", height=220)
            with gr.Row():
                log_out = gr.Textbox(label="Detections", lines=8)
                timing_out = gr.Textbox(label="Timing", lines=8)

    ref_img.upload(on_ref_upload, inputs=[ref_img], outputs=[state_ref_image, boxes_table, state_pending, ref_img])
    ref_img.select(
        on_ref_click, inputs=[state_ref_image, boxes_table, state_pending], outputs=[boxes_table, state_pending, ref_img]
    )
    boxes_table.change(on_rows_change, inputs=[state_ref_image, boxes_table], outputs=[ref_img])
    undo_btn.click(undo_box, inputs=[state_ref_image, boxes_table], outputs=[boxes_table, ref_img])
    clear_btn.click(clear_boxes, inputs=[state_ref_image], outputs=[boxes_table, state_pending, ref_img])

    create_ref_btn.click(
        create_reference,
        inputs=[state_ref_image, boxes_table, refine_cb],
        outputs=[create_ref_status, ref_img],
    )

    run_btn.click(
        run_pipeline,
        inputs=[target_img, sam3_cb, yoloret_cb, source_radio],
        outputs=[
            ref_out,
            input_out,
            yoloe_out,
            yoloe_out2,
            sam3_out,
            yoloret_out,
            classical_out,
            yoloe_raw_out,
            log_out,
            timing_out,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
