import os

# compose.amd.yaml always passes HSA_OVERRIDE_GFX_VERSION, empty when the GPU doesn't need it; ROCm reads
# it when PyTorch first touches the GPU, so an empty value is removed before that (the training process
# inherits this environment)
if os.environ.get("HSA_OVERRIDE_GFX_VERSION", None) == "":
    del os.environ["HSA_OVERRIDE_GFX_VERSION"]

import collections
import csv
import ctypes
import gc
import io
import json
import re
import shutil
import statistics
import tempfile
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

import evaluate
import prompt_tune
import sam3_models
from dataset import Dataset, bbox_xyxy, flags_of
from detect_utils import filter_and_dedupe_detections

# SAM 3's weights are in this container's own bind mount (dockers/batch-sam3/modelData/sam3). The download
# is gated on Hugging Face, so they are not fetched automatically: copy sam3.pt there.
WEIGHTS = "modelData/sam3/sam3.pt"
MODELS_DIR = "models"
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")  # the dataset root, bind-mounted (compose.yaml)
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", "SAM3")  # the only place written to under the dataset root
WORK_DIR = os.environ.get("WORK_DIR", "/app/work")  # prompt tuning's feature cache, inside the container
RAW_CONF = 0.01
DEFAULT_CONF = 0.02
DEDUPE_IOU = 0.9
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# SAM 3 runs in PyTorch (there is no ONNX export), so with a GPU build (compose.gpu.yaml / compose.amd.yaml)
# both inference and prompt tuning run on the GPU. torch.cuda is also the API for an AMD GPU under ROCm.
TORCH_GPU = torch.cuda.is_available()
IS_ROCM = bool(getattr(torch.version, "hip", None))
GPU_NAME = torch.cuda.get_device_name(0) if TORCH_GPU else None
TORCH_DEVICE = "0" if TORCH_GPU else "cpu"
DEVICE = f"{'ROCm' if IS_ROCM else 'cuda'} ({GPU_NAME})" if TORCH_GPU else "cpu"
print(f"device: {DEVICE}")

app = Flask(__name__, static_folder="static", static_url_path="/static")
_models_lock = threading.Lock()
_models = {}  # name -> {"meta": dict}
_engine_lock = threading.Lock()  # one SAM 3 at a time: it is not thread-safe, and times are taken one model at a time
_engine = {"engine": None, "models": {}}  # the loaded SAM 3 and the built models made runnable on it
DS = Dataset(DATA_ROOT)

KIND_LABELS = {"text": "Text prompt", "exemplar": "Exemplar prompt", "visual": "Visual prompt",
               "prompt_tune": "Prompt-tuned"}


def _model_dir(name):
    return os.path.join(MODELS_DIR, name)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_meta(name):
    with open(os.path.join(_model_dir(name), "meta.json")) as f:
        return json.load(f)


def _release_memory():
    """Hand freed memory back to the system (glibc keeps it for reuse otherwise, so the container's
    memory use would never go down after a job)."""
    gc.collect()
    if TORCH_GPU:
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _get_engine():
    """SAM 3, loaded on first use (about a minute on CPU). The caller holds _engine_lock."""
    if _engine["engine"] is None:
        if not os.path.exists(WEIGHTS):
            raise RuntimeError(f"SAM 3 weights not found: copy sam3.pt to dockers/batch-sam3/{WEIGHTS}")
        t0 = time.perf_counter()
        e = sam3_models.Sam3Engine(WEIGHTS, TORCH_DEVICE)
        if TORCH_GPU:
            # the GPU compiles its kernels on the first run (MIOpen / cuDNN autotuning): do that now, so the
            # first measured image isn't slower than the rest
            f = e.features(np.zeros((640, 640, 3), np.uint8))
            e.detect(f, e.text_tokens(["object"]), (640, 640))
        _engine["engine"] = e
        print(f"SAM 3 loaded in {time.perf_counter() - t0:.0f} s")
    return _engine["engine"]


def _unload_engine():
    """Free SAM 3 (it is loaded again when next used), so a prompt-tuning process has the memory."""
    with _engine_lock:
        was = _engine["engine"] is not None
        _engine["engine"] = None
        _engine["models"] = {}
    _release_memory()
    return was


def _runnable(name, meta):
    """The built model `name` made runnable on the loaded SAM 3 (its prompt tokens prepared once).
    The caller holds _engine_lock."""
    m = _engine["models"].get(name)
    if m is None or m.meta.get("created") != meta.get("created"):
        m = sam3_models.Sam3Model(_get_engine(), meta, _model_dir(name))
        _engine["models"][name] = m
    return m


def _public_meta(meta):
    return {**meta, "num_classes": len(meta.get("class_names", []))}


def _load_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in sorted(os.listdir(MODELS_DIR)):
        if os.path.exists(os.path.join(_model_dir(name), "meta.json")):
            try:
                _models[name] = {"meta": _read_meta(name)}
            except Exception as e:
                print(f"failed to read model '{name}': {e}")


def _predict(meta, image, conf):
    """(detections, ms, None). The time covers the whole model on this image - backbone (on the composite
    for an exemplar model), prompts, decoding and our filtering - but not loading SAM 3 or preparing the
    model's prompt tokens, which happen once."""
    with _engine_lock:
        model = _runnable(meta["name"], meta)
        t0 = time.perf_counter()
        dets = model.predict(image, RAW_CONF)
        names = meta["class_names"]
        raw = [{"bbox": [x1, y1, x2, y2], "conf": s, "cls": c, "name": names[c], "prompt": p}
               for x1, y1, x2, y2, s, c, p in dets]
        # several prompts of a class (text synonyms) find the same object: keep the highest confidence
        out = filter_and_dedupe_detections(raw, conf, iou_thresh=DEDUPE_IOU)
        ms = round((time.perf_counter() - t0) * 1000, 1)
    return out, ms, None


# =====================================================================================================
# jobs: builds and tests wait in one queue and run one after the other in a background worker
# =====================================================================================================
class Job:
    _seq = 0

    def __init__(self, kind, title, fn, target=None):
        Job._seq += 1
        self.id = f"{int(time.time() * 1000)}_{Job._seq}"
        self.kind, self.title, self.fn = kind, title, fn
        self.target = target  # build: the model name it creates
        self.status = "queued"  # queued | running | done | error | cancelled
        self.progress, self.message = 0.0, "waiting in the queue"
        self.log = collections.deque(maxlen=400)
        self.queued, self.started, self.finished = time.time(), None, None
        self.result = None
        self.cancel = threading.Event()

    def say(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        print(f"[job {self.kind}] {msg}")

    def set(self, progress=None, message=None):
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            self.message = message

    def as_dict(self, with_log=True):
        elapsed = ((self.finished or time.time()) - self.started) if self.started else 0
        eta = elapsed / self.progress - elapsed if self.status == "running" and self.progress > 0.01 else None
        return {"id": self.id, "kind": self.kind, "title": self.title, "target": self.target, "status": self.status,
                "progress": round(self.progress, 4), "message": self.message, "log": list(self.log) if with_log else [],
                "queued": self.queued, "started": self.started, "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1) if eta else None, "result": self.result}


_jobs = []  # the jobs of this session in queue order; finished ones are kept for the page (capped)
_jobs_cv = threading.Condition()
MAX_FINISHED_JOBS = 30


def _enqueue(kind, title, fn, target=None):
    job = Job(kind, title, fn, target)
    with _jobs_cv:
        _jobs.append(job)
        finished = [j for j in _jobs if j.status not in ("queued", "running")]
        for old in finished[:-MAX_FINISHED_JOBS]:
            _jobs.remove(old)
        _jobs_cv.notify()
    return job


def _queued_builds():
    with _jobs_cv:
        return {j.target for j in _jobs if j.kind == "build" and j.status in ("queued", "running")}


def _run_job(job):
    job.started = time.time()
    job.set(0.0, "starting")
    try:
        job.result = job.fn(job)
        job.status = "cancelled" if job.cancel.is_set() else "done"
        job.set(1.0, job.status)
    except prompt_tune.Cancelled:
        job.status = "cancelled"
        job.say("cancelled")
        job.set(message="cancelled")
    except Exception as e:
        job.status = "error"
        job.say(f"ERROR: {e}")
        job.say(traceback.format_exc().strip().splitlines()[-1])
        job.set(message=str(e))
        traceback.print_exc()
    finally:
        job.finished = time.time()
        # SAM 3 is freed after every job, so an idle container (or a following prompt tuning) has the memory
        _unload_engine()


def _worker():
    while True:
        with _jobs_cv:
            while not any(j.status == "queued" for j in _jobs):
                _jobs_cv.wait()
            job = next(j for j in _jobs if j.status == "queued")
            job.status = "running"  # claimed under the lock, so a remove from the queue can't race it
        _run_job(job)


threading.Thread(target=_worker, daemon=True, name="job-worker").start()


@app.route("/api/jobs")
def jobs_list():
    """The queue: the running job (with its log), the queued ones in order, and recent finished ones."""
    with _jobs_cv:
        jobs = list(_jobs)
    running = next((j for j in jobs if j.status == "running"), None)
    done = [j for j in reversed(jobs) if j.status in ("done", "error", "cancelled")]
    return jsonify(running=running.as_dict() if running else None,
                   queued=[j.as_dict(False) for j in jobs if j.status == "queued"],
                   finished=[j.as_dict(k == 0) for k, j in enumerate(done)])


@app.route("/api/jobs/current")
def job_current():
    """The running job, else the most recent one (kept for scripts)."""
    with _jobs_cv:
        job = next((j for j in _jobs if j.status == "running"), _jobs[-1] if _jobs else None)
    return jsonify(job=job.as_dict() if job else None)


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def job_cancel_one(job_id):
    """Cancel the running job, or take a queued one out of the queue."""
    with _jobs_cv:
        job = next((j for j in _jobs if j.id == job_id), None)
        if job is None:
            abort(404)
        if job.status == "queued":
            job.status, job.finished = "cancelled", time.time()
            job.set(message="removed from the queue")
        elif job.status == "running":
            job.cancel.set()
            job.say("cancel requested")
    return jsonify(status="ok")


@app.route("/api/jobs/cancel", methods=["POST"])
def job_cancel():
    with _jobs_cv:
        job = next((j for j in _jobs if j.status == "running"), None)
    if job:
        job.cancel.set()
        job.say("cancel requested")
    return jsonify(status="ok")


@app.route("/api/jobs/clear", methods=["POST"])
def jobs_clear():
    """Forget the finished jobs (the queue and the running job stay)."""
    with _jobs_cv:
        _jobs[:] = [j for j in _jobs if j.status in ("queued", "running")]
    return jsonify(status="ok")


# =====================================================================================================
# dataset
# =====================================================================================================
def _dataset_available():
    return os.path.isdir(os.path.join(DATA_ROOT, "trainImages")) or os.path.isdir(os.path.join(DATA_ROOT, "testImages"))


@app.route("/api/dataset")
def dataset_info():
    if not _dataset_available():
        return jsonify(status="error", message=f"no dataset at {DATA_ROOT} (mount it in compose.yaml)"), 404
    return jsonify(DS.scan(refresh=request.args.get("refresh") == "1"))


@app.route("/api/image")
def dataset_image():
    """An image file from the dataset root, by root-relative path (only below the root, only images)."""
    rel = request.args.get("path", "")
    full = os.path.realpath(os.path.join(DATA_ROOT, rel))
    if not full.startswith(os.path.realpath(DATA_ROOT) + os.sep) or not full.lower().endswith((".png", ".jpg", ".jpeg")):
        abort(404)
    if not os.path.isfile(full):
        abort(404)
    return send_file(full)


# =====================================================================================================
# models
# =====================================================================================================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return ("", 200) if os.path.exists(WEIGHTS) else (f"SAM 3 weights not found: {WEIGHTS}", 503)


@app.route("/api/models")
def list_models():
    with _models_lock:
        metas = [_public_meta(e["meta"]) for e in _models.values()]
    return jsonify(models=sorted(metas, key=lambda m: m["name"]), device=DEVICE, cpu_threads=os.cpu_count(),
                   weights=os.path.exists(WEIGHTS))


# BGR versions of the page's box colours, one per class
BOX_COLORS = [(75, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245),
              (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191)]
NEG_COLOR = (128, 128, 128)


def _draw_boxes(img, boxes, cls, names):
    """Boxes with "#i class" labels; class -1 (a negative exemplar) in grey, labelled "not"."""
    h, w = img.shape[:2]
    lw = max(2, round(max(h, w) / 400))
    scale = max(0.5, max(h, w) / 1200)
    thick = max(1, lw // 2)
    for i, (box, c) in enumerate(zip(boxes, cls)):
        x1, y1, x2, y2 = (round(v) for v in box)
        color = NEG_COLOR if c < 0 else BOX_COLORS[c % len(BOX_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        label = f"#{i} " + ("not" if c < 0 else names[c] if c < len(names) else f"object{c}")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 if y1 - th - base - 4 < 0 else y1 - th - base - 4  # above the box, or inside at the top
        cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + base + 4), color, -1)
        cv2.putText(img, label, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def _ref_images(name, meta):
    """[(file name, annotated BGR image)]: the exemplar panel (exemplar) and the reference images with
    the boxes the model was made from (exemplar, visual)."""
    names = meta.get("class_names") or []
    out = []
    if meta["type"] == "exemplar":
        panel = cv2.imread(os.path.join(_model_dir(name), "panel.png"))
        if panel is not None:
            p = meta["panel"]
            out.append((f"{name}_panel.png", _draw_boxes(panel, p["boxes"], p["cls"], names)))
    for i, r in enumerate(meta.get("refs") or []):
        img = cv2.imread(os.path.join(DATA_ROOT, r["path"]))
        if img is not None:
            stem = r["path"].split("/")[-2]
            out.append((f"{name}_ref{i + 1}_{stem}.png", _draw_boxes(img, r["bboxes"], r["cls"], names)))
    return out


def _model_files(name):
    """What a model was made from, as [(file name, bytes)]: the panel and reference images with their
    boxes (exemplar, visual), the prompts per class (text) or the training summary (prompt-tuned)."""
    meta = _read_meta(name)
    kind = meta["type"]
    if kind in ("exemplar", "visual"):
        return [(f, cv2.imencode(".png", img)[1].tobytes()) for f, img in _ref_images(name, meta)]
    if kind == "text":
        text = "\n".join(f"{c}: " + " | ".join(p) for c, p in zip(meta["class_names"], meta["prompts"]))
        return [(f"{name}_prompts.txt", (text + "\n").encode())]
    return [(f"{name}_training.json", json.dumps(meta.get("training", {}), indent=2).encode())]


@app.route("/api/models/<name>/preview")
def model_preview(name):
    """A small JPEG of the exemplar panel, or of the first reference image with its boxes."""
    if not NAME_RE.match(name) or not os.path.exists(os.path.join(_model_dir(name), "meta.json")):
        abort(404)
    refs = _ref_images(name, _read_meta(name))
    if not refs:
        abort(404)
    img = refs[0][1]
    s = 420 / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return send_file(io.BytesIO(cv2.imencode(".jpg", img)[1].tobytes()), mimetype="image/jpeg")


@app.route("/api/models/<name>/download")
def download_model(name):
    if not NAME_RE.match(name) or not os.path.exists(os.path.join(_model_dir(name), "meta.json")):
        abort(404)
    files = _model_files(name)
    if len(files) == 1:
        fname, data = files[0]
        mimetype = "image/png" if fname.endswith(".png") else "text/plain"
        return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=fname)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, data in files:
            z.writestr(fname, data)
        z.write(os.path.join(_model_dir(name), "meta.json"), "meta.json")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{name}.zip")


@app.route("/api/models/download.zip")
def download_all():
    """Every model's files from _model_files(), plus its meta.json, one folder per model."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(MODELS_DIR)):
            d = _model_dir(name)
            if not NAME_RE.match(name) or not os.path.exists(os.path.join(d, "meta.json")):
                continue
            for fname, data in _model_files(name):
                z.writestr(f"{name}/{fname}", data)
            z.write(os.path.join(d, "meta.json"), f"{name}/meta.json")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="sam3_models.zip")


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    if not NAME_RE.match(name):
        abort(400)
    with _models_lock:
        _models.pop(name, None)
    shutil.rmtree(_model_dir(name), ignore_errors=True)
    return jsonify(status="ok")


# ----------------------------------------------------------------------------- building
def _save_model(name, meta, extra_files=()):
    """Replace models/<name>/ with meta.json and the extra (src, dst name) files."""
    d = _model_dir(name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for src, dst in extra_files:
        shutil.move(src, os.path.join(d, dst))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _ref_objects(spec, class_ids):
    """The scope's reference objects: [(image dir, truth, objects to use, all objects of the image)], and
    how many in-scope objects were left out because of their flags."""
    out, skipped = [], 0
    for d in DS.scope_images(spec["scope"], "ref"):
        truth = DS.truth(d)
        keep, excluded = DS.split_objects(truth, class_ids, spec["include_uncompleted"], spec["include_covered"])
        skipped += len(excluded)
        if keep:
            out.append((d, truth, keep))
    return out, skipped


def _words(spec, names):
    """The text SAM 3 gets next to the boxes: "visual" (SAM 3's default for box prompts) or the class name."""
    return list(names) if spec["options"].get("with_text") else [sam3_models.VISUAL_WORD] * len(names)


def _build_text(job, name, spec, class_ids, names, tmp_dir):
    """Text prompt; every class can have several prompts (synonyms), each its own entry."""
    groups = spec["options"].get("prompts") or [[n] for n in names]
    if len(groups) != len(names):
        raise ValueError(f"expected prompts for {len(names)} class(es), got {len(groups)}")
    groups = [list(dict.fromkeys(p.strip() for p in g if p and p.strip())) for g in groups]
    for n, g in zip(names, groups):
        if not g:
            raise ValueError(f"class '{n}' has no prompt")
    flat = [p for g in groups for p in g]
    dup = [p for p in set(flat) if flat.count(p) > 1]
    if dup:
        raise ValueError(f"the same prompt is used more than once: {', '.join(dup)}")
    job.say(f"{len(flat)} prompt(s) for {len(names)} class(es)")
    return [], {"prompts": groups, "prompt_names": flat, "prompt_class": [i for i, g in enumerate(groups) for _ in g]}


def _build_exemplar(job, name, spec, class_ids, names, tmp_dir):
    """Up to N reference objects per class, cropped with some context and packed into a panel that is
    pasted beside every test image. With "negatives", objects of other classes in the reference images
    are added as negative exemplars (for an all-classes model, the other classes' crops are the negatives)."""
    o = spec["options"]
    per_class = max(1, int(o.get("max_exemplars") or 4))
    model_idx = {cid: i for i, cid in enumerate(class_ids)}
    refs, skipped = _ref_objects(spec, class_ids)
    chosen = {i: [] for i in range(len(names))}  # class -> [(image dir, object)]
    negatives = []
    for d, truth, keep in refs:
        for obj in keep:
            c = model_idx[int(obj["class"])]
            if len(chosen[c]) < per_class:
                chosen[c].append((d, obj))
        if o.get("negatives") and spec["scope"]["type"] == "single":
            for obj in truth.get("objects", []):
                if int(obj["class"]) not in model_idx and len(negatives) < per_class \
                        and not obj.get("uncompleted") and not obj.get("covered"):
                    negatives.append((d, obj))
    missing = [names[c] for c, v in chosen.items() if not v]
    if missing:
        raise ValueError(f"no usable reference object for {', '.join(missing)} in refImages (check the flag checkboxes)")
    crops, cls, refs_meta, images = [], [], {}, {}
    for c, items in list(chosen.items()) + [(-1, negatives)]:
        for d, obj in items:
            if d not in images:
                images[d] = cv2.imread(os.path.join(d, "org.png"))
            crops.append(sam3_models.crop_object(images[d], bbox_xyxy(obj)))
            cls.append(c)
            r = refs_meta.setdefault(d, {"path": DS.rel(os.path.join(d, "org.png")), "bboxes": [], "cls": []})
            r["bboxes"].append([round(v, 1) for v in bbox_xyxy(obj)])
            r["cls"].append(c)
    panel, boxes = sam3_models.build_panel(crops)
    path = os.path.join(tmp_dir, "panel.png")
    cv2.imwrite(path, panel)
    job.say(f"panel: {sum(c >= 0 for c in cls)} exemplar(s) for {len(names)} class(es)"
            f"{f', {len(negatives)} negative(s)' if negatives else ''}; {skipped} object(s) left out because of their flags")
    return [(path, "panel.png")], {
        "panel": {"boxes": [[round(v, 1) for v in b] for b in boxes], "cls": cls, "width": panel.shape[1], "height": panel.shape[0]},
        "refs": list(refs_meta.values()),
        "exemplars_per_class": {names[c]: len(v) for c, v in chosen.items()}}


def _build_visual(job, name, spec, class_ids, names, tmp_dir):
    """Exemplar tokens encoded once on the reference images (every allowed object of the class, as SAM 3's
    geometry encoder sees them there), concatenated per class over the images and stored."""
    model_idx = {cid: i for i, cid in enumerate(class_ids)}
    refs, skipped = _ref_objects(spec, class_ids)
    if not refs:
        raise ValueError("no usable reference object in refImages for this scope (check the flag checkboxes)")
    tokens = [[] for _ in names]
    refs_meta = []
    with _engine_lock:
        job.set(0.05, "loading SAM 3")
        e = _get_engine()
        for k, (d, truth, keep) in enumerate(refs):
            if job.cancel.is_set():
                raise prompt_tune.Cancelled()
            job.set(0.1 + 0.85 * k / len(refs), f"encoding reference image {k + 1}/{len(refs)}")
            img = cv2.imread(os.path.join(d, "org.png"))
            feats = e.features(img)
            cls = [model_idx[int(o["class"])] for o in keep]
            for c in sorted(set(cls)):
                tok, _ = e.box_tokens(feats, img.shape, [bbox_xyxy(o) for o, k2 in zip(keep, cls) if k2 == c])
                tokens[c].append(tok.cpu())
            refs_meta.append({"path": DS.rel(os.path.join(d, "org.png")),
                              "bboxes": [[round(v, 1) for v in bbox_xyxy(o)] for o in keep], "cls": cls})
    missing = [names[c] for c, t in enumerate(tokens) if not t]
    if missing:
        raise ValueError(f"no usable reference object for {', '.join(missing)} in refImages (check the flag checkboxes)")
    path = os.path.join(tmp_dir, "prompt.pt")
    torch.save({"text": _words(spec, names), "tokens": [torch.cat(t) for t in tokens]}, path)
    job.say(f"{len(refs)} reference image(s), {sum(len(r['cls']) for r in refs_meta)} object(s) encoded, "
            f"{skipped} left out because of their flags")
    return [(path, "prompt.pt")], {"refs": refs_meta,
                                   "ref_images_per_class": {n: len(t) for n, t in zip(names, tokens)}}


def _build_prompt_tune(job, name, spec, class_ids, names, tmp_dir):
    o = spec["options"]
    start = [str(s).strip() for s in (o.get("start_text") or [])]
    if len(start) != len(names) or not all(start):
        start = list(names)
    args = {"data_root": DATA_ROOT, "scope": spec["scope"], "class_ids": class_ids, "names": names, "start_text": start,
            "include_uncompleted": spec["include_uncompleted"], "include_covered": spec["include_covered"],
            "max_images": max(0, int(o.get("max_images") or 0)), "seed": int(o.get("seed") or 0),
            "epochs": max(1, int(o.get("epochs") or 10)), "tokens": max(1, int(o.get("tokens") or 4)),
            "lr": float(o.get("lr") or 5e-3), "work": os.path.join(WORK_DIR, name),
            "weights": os.path.abspath(WEIGHTS), "device": TORCH_DEVICE}
    job.say(f"prompt tuning: {args['epochs']} epochs, {args['tokens']} token(s) per class, lr {args['lr']}, device "
            f"{('GPU ' + GPU_NAME) if TORCH_GPU else 'cpu (slow: about a minute per image per epoch)'}")
    # SAM 3 is not loaded in the server during a job (see _run_job): the training process has the memory
    path, summary = prompt_tune.tune_in_process(args, log=job.say, progress=lambda p, m: job.set(0.02 + 0.96 * p, m),
                                                cancelled=job.cancel.is_set)
    dst = os.path.join(tmp_dir, "prompt.pt")
    shutil.move(path, dst)
    shutil.rmtree(args["work"], ignore_errors=True)
    return [(dst, "prompt.pt")], {"training": {"epochs": args["epochs"], "tokens": args["tokens"], "lr": args["lr"],
                                               "max_images": args["max_images"], "seed": args["seed"], **summary}}


BUILDERS = {"text": _build_text, "exemplar": _build_exemplar, "visual": _build_visual, "prompt_tune": _build_prompt_tune}


@app.route("/api/models", methods=["POST"])
def build_model():
    """Build a model from the dataset, as a background job.
    Body: {name, kind, scope, include_uncompleted, include_covered, options}."""
    spec = request.get_json(force=True)
    name = spec.get("name", "")
    if not NAME_RE.match(name):
        return jsonify(status="error", message="name must be 1-64 chars of A-Z a-z 0-9 _ -"), 400
    kind = spec.get("kind")
    if kind not in BUILDERS:
        return jsonify(status="error", message=f"unknown kind '{kind}'"), 400
    if not _dataset_available():
        return jsonify(status="error", message=f"no dataset at {DATA_ROOT}"), 400
    try:
        class_ids = DS.validate_scope(spec.get("scope"))
    except ValueError as e:
        return jsonify(status="error", message=str(e)), 400
    if name in _queued_builds():
        return jsonify(status="error", message=f"a build of '{name}' is already in the queue"), 409
    spec["include_uncompleted"] = bool(spec.get("include_uncompleted"))
    spec["include_covered"] = bool(spec.get("include_covered"))
    spec["options"] = spec.get("options") or {}
    all_names = DS.class_names()
    names = [all_names[c] for c in class_ids]

    def run(job):
        t0 = time.perf_counter()
        job.say(f"building {KIND_LABELS[kind]} model '{name}' for {DS.describe_scope(spec['scope'], all_names)}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            extra, info = BUILDERS[kind](job, name, spec, class_ids, names, tmp_dir)
            if job.cancel.is_set():
                raise prompt_tune.Cancelled()
            meta = {
                "name": name, "type": kind, "engine": "sam3", "scope": spec["scope"],
                "scope_label": DS.describe_scope(spec["scope"], all_names),
                "class_ids": class_ids, "class_names": names,
                "include_uncompleted": spec["include_uncompleted"], "include_covered": spec["include_covered"],
                "options": spec["options"], "created": _now(), **info,
            }
            meta["build_seconds"] = round(time.perf_counter() - t0, 1)
            _save_model(name, meta, extra)
        with _models_lock:
            _models[name] = {"meta": meta}
        job.say(f"done in {meta['build_seconds']} s")
        return {"model": _public_meta(meta)}

    job = _enqueue("build", f"Build {name} ({KIND_LABELS[kind]}, {DS.describe_scope(spec['scope'], all_names)})", run, target=name)
    return jsonify(status="ok", job=job.as_dict())


# =====================================================================================================
# testing
# =====================================================================================================
def _model_test_plan(name):
    """(scope, global class ids that are ground truth, meta) for testing one model."""
    with _models_lock:
        entry = _models.get(name)
    if entry is None:
        raise ValueError(f"unknown model {name!r}")
    meta = entry["meta"]
    return meta["scope"], meta["class_ids"], meta


@app.route("/api/tests", methods=["POST"])
def start_test():
    """Test models on testImages, as a background job. Body: {name?, models: [...]}.
    No confidence threshold: everything down to the raw 0.01 is kept and scored threshold-free."""
    body = request.get_json(force=True)
    model_names = [m for m in body.get("models") or [] if m]
    if not model_names:
        return jsonify(status="error", message="select at least one model"), 400
    if not _dataset_available():
        return jsonify(status="error", message=f"no dataset at {DATA_ROOT}"), 400
    pending = _queued_builds()
    try:
        for n in model_names:
            if n not in pending:
                _model_test_plan(n)
    except ValueError as e:
        return jsonify(status="error", message=str(e)), 400
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", body.get("name") or "_".join(model_names))[:60].strip("_") or "test"
    max_images = max(0, int(body.get("max_images") or 0))

    def run(job):
        plans = {n: _model_test_plan(n) for n in model_names}  # fails if a queued build failed
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"
        return _run_test(job, run_id, model_names, plans, max_images)

    job = _enqueue("test", f"Test {', '.join(model_names)}", run)
    return jsonify(status="ok", job=job.as_dict())


def _run_test(job, run_id, model_names, plans, max_images=0):
    all_names = DS.class_names()
    images = {}  # img dir -> [model names]
    for n in model_names:
        scope = plans[n][0]
        dirs = DS.scope_images(scope, "test")
        if max_images:  # a quick look: the first N test images of each model's scope
            dirs = dirs[:max_images]
        for d in dirs:
            images.setdefault(d, []).append(n)
    order = sorted(images)
    total = sum(len(v) for v in images.values())
    job.say(f"{len(order)} test image(s), {total} model run(s)" + (f" (first {max_images} per scope)" if max_images else ""))
    with _engine_lock:
        job.set(0.0, "loading SAM 3")
        _get_engine()

    per_model = {n: [] for n in model_names}  # for scoring: [{"dets", "gts"}]
    out_images = []
    done = 0
    for d in order:
        if job.cancel.is_set():
            raise prompt_tune.Cancelled()
        truth = DS.truth(d)
        path = os.path.join(d, "org.png")
        image = cv2.imread(path)
        h, w = image.shape[:2]
        rec = {"path": DS.rel(path), "width": w, "height": h, "results": {}, "model_ms": {}, "model_split_ms": {},
               "truth": [{"bbox": [round(v, 1) for v in bbox_xyxy(o)], "class_id": int(o["class"]),
                          "name": o.get("class_name", all_names.get(int(o["class"]), "")),
                          "uncompleted": bool(o.get("uncompleted")), "covered": bool(o.get("covered"))}
                         for o in truth.get("objects", [])]}
        for n in images[d]:
            scope, class_ids, meta = plans[n]
            # one model at a time (the engine lock), so each time is that model alone
            dets, ms, _ = _predict(meta, image, RAW_CONF)
            idx = {cid: i for i, cid in enumerate(class_ids)}
            gts = [(bbox_xyxy(o), idx[int(o["class"])], flags_of(o)) for o in DS.test_objects(truth, class_ids)]
            per_model[n].append({"dets": [(x["bbox"], x["conf"], x["cls"]) for x in dets], "gts": gts, "rec": rec, "raw": dets})
            rec["results"][n] = dets
            rec["model_ms"][n] = ms
            done += 1
            job.set(0.95 * done / total, f"{done}/{total}: {n} on {rec['path'].split('/')[-2]}")
        out_images.append(rec)

    job.set(0.96, "scoring")
    metrics_models, matches = [], {}
    for n in model_names:
        scope, class_ids, meta = plans[n]
        names = [all_names[c] for c in class_ids]
        m, match, image_matches = evaluate.score_model(per_model[n], names, class_agnostic=False)
        for img, im in zip(per_model[n], image_matches):
            for det, (bits, _) in zip(img["raw"], im):
                det["tp"] = bool(bits & 1)
        ms = [img["rec"]["model_ms"][n] for img in per_model[n]]
        m.update({"name": n, "type": meta.get("type"), "scope": scope, "scope_label": DS.describe_scope(scope, all_names),
                  "legacy": False, "n_images": len(per_model[n]),
                  "ms_median": round(statistics.median(ms), 1) if ms else None,
                  "ms_mean": round(statistics.fmean(ms), 1) if ms else None})
        metrics_models.append(m)
        matches[n] = {"class_agnostic": False, "classes": match}
        best = m.get("best") or {}
        job.say(f"{n}: AP50 {_fmt(m['ap50'])}, AP50-95 {_fmt(m['ap50_95'])}, best F1 {_fmt(best.get('f1'))} "
                f"@ {best.get('conf')}, median {m['ms_median']} ms")

    out = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(out, exist_ok=True)
    models_info = [{"name": n, "type": plans[n][2].get("type"), "scope": plans[n][0], "class_ids": plans[n][1],
                    "class_names": [all_names[c] for c in plans[n][1]], "class_agnostic": False, "meta": plans[n][2]}
                   for n in model_names]
    timing = {"mode": "accurate", "device": DEVICE, "cpu_threads": os.cpu_count()}
    results = {
        "created": _now(), "folder": run_id, "conf_threshold": RAW_CONF, "timing": timing,
        "models": [{"name": mi["name"], "type": mi["type"], "class_names": mi["class_names"], "class_agnostic": False}
                   for mi in models_info],
        "images": out_images,
    }
    _write_json(os.path.join(out, "results.json"), results)
    _write_json(os.path.join(out, "metrics.json"), {"created": results["created"], "id": run_id, "timing": timing, "models": metrics_models})
    _write_json(os.path.join(out, "matches.json"), matches)
    _write_json(os.path.join(out, "test.json"), {"id": run_id, "created": results["created"], "models": models_info,
                                                 "raw_conf": RAW_CONF, "dedupe_iou": DEDUPE_IOU,
                                                 "iou_thresholds": evaluate.IOU_THRESHOLDS, "max_images": max_images,
                                                 "timing": timing, "data_root": DATA_ROOT})
    _write_csv(os.path.join(out, "metrics.csv"), metrics_models)
    job.say(f"written to output/SAM3/{run_id}/")
    return {"id": run_id}


def _fmt(v):
    return "–" if v is None else f"{v:.3f}"


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def _curve_at(curve, conf):
    """P, R, F1 from a compact curve ([[t, p, r, f1], ...]) at the closest confidence step."""
    row = min(curve, key=lambda c: abs(c[0] - conf)) if curve else None
    return (row[1], row[2], row[3]) if row else (None, None, None)


def _write_csv(path, models):
    head = ["model", "type", "scope", "class", "n_gt", "n_det", "AP50", "AP50-95", "best_conf", "P@best", "R@best", "F1@best",
            "recall_uncompleted@best", "recall_covered@best", "ms_median", "ms_mean"]
    for c in evaluate.CSV_CONFS:
        head += [f"P@{c}", f"R@{c}", f"F1@{c}"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(head)
        for m in models:
            for row_name, c in [("ALL", m)] + [(cl["name"], cl) for cl in m["classes"]]:
                best = c.get("best") or {}
                row = [m["name"], m["type"], m["scope_label"], row_name, c["n_gt"], c["n_det"], c["ap50"], c["ap50_95"],
                       best.get("conf"), best.get("p"), best.get("r"), best.get("f1"),
                       c.get("recall_uncompleted"), c.get("recall_covered"),
                       m["ms_median"] if row_name == "ALL" else "", m["ms_mean"] if row_name == "ALL" else ""]
                for t in evaluate.CSV_CONFS:
                    row += list(_curve_at(c.get("curve") or [], t))
                wr.writerow(["" if v is None else (round(v, 4) if isinstance(v, float) else v) for v in row])


def _test_dir(test_id):
    if not re.match(r"^[A-Za-z0-9_-]{1,100}$", test_id):
        abort(404)
    d = os.path.join(OUTPUT_DIR, test_id)
    if not os.path.isdir(d):
        abort(404)
    return d


@app.route("/api/tests")
def list_tests():
    out = []
    if os.path.isdir(OUTPUT_DIR):
        for n in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            mp = os.path.join(OUTPUT_DIR, n, "metrics.json")
            if os.path.isfile(mp):
                try:
                    with open(mp, encoding="utf-8") as f:
                        m = json.load(f)
                    out.append({"id": n, "created": m.get("created"),
                                "models": [x["name"] for x in m.get("models", [])]})
                except (OSError, ValueError):
                    pass
    return jsonify(tests=out)


@app.route("/api/tests/<test_id>/<part>")
def test_part(test_id, part):
    if part not in ("metrics", "results", "matches", "test"):
        abort(404)
    return send_file(os.path.join(_test_dir(test_id), f"{part}.json"), mimetype="application/json")


@app.route("/api/tests/<test_id>/metrics.csv")
def test_csv(test_id):
    return send_file(os.path.join(_test_dir(test_id), "metrics.csv"), mimetype="text/csv", as_attachment=True,
                     download_name=f"{test_id}_metrics.csv")


@app.route("/api/tests/<test_id>", methods=["DELETE"])
def delete_test(test_id):
    shutil.rmtree(_test_dir(test_id), ignore_errors=True)
    return jsonify(status="ok")


# =====================================================================================================
# single-image detection (also for other programs)
# =====================================================================================================
@app.route("/api/detect", methods=["POST"])
def detect():
    """Run models on one uploaded image, one after the other. Form: image, models (comma list, default
    all), conf. SAM 3 is loaded for this if it isn't (about a minute on CPU)."""
    conf = float(request.form.get("conf", DEFAULT_CONF))
    wanted = [m for m in request.form.get("models", "").split(",") if m]
    with _models_lock:
        metas = {n: e["meta"] for n, e in _models.items() if not wanted or n in wanted}
    if not metas:
        return jsonify(status="error", message="no models selected"), 400
    image = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(status="error", message="could not decode image"), 400
    results, errors, model_ms = {}, {}, {}
    for n, meta in metas.items():
        try:
            results[n], model_ms[n], _ = _predict(meta, image, conf)
        except Exception as e:
            errors[n] = str(e)
    h, w = image.shape[:2]
    return jsonify(width=w, height=h, results=results, errors=errors, model_ms=model_ms, timing="accurate")


os.makedirs(WORK_DIR, exist_ok=True)
if _dataset_available():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
_load_all()
if not os.path.exists(WEIGHTS):
    print(f"WARNING: SAM 3 weights not found at {WEIGHTS} - copy sam3.pt into dockers/batch-sam3/modelData/sam3/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7871, threaded=True)
