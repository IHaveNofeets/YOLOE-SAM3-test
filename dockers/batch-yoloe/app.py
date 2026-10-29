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
import os
import re
import shutil
import statistics
import tempfile
import threading
import time
import traceback
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
import onnxruntime
import requests
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from ultralytics import YOLOE
from ultralytics.utils import SETTINGS
from ultralytics.utils.downloads import attempt_download_asset

import evaluate
import train
from dataset import Dataset, bbox_xyxy, flags_of
from detect_utils import filter_and_dedupe_detections
from yoloe_bake import bake_visual_prompts_multi

# All weights live in this container's own bind mount (dockers/batch-yoloe/modelData/yoloe)
WEIGHTS_DIR = "modelData/yoloe"
BASE_WEIGHTS = f"{WEIGHTS_DIR}/yoloe-26l-seg.pt"
# prompt-free, built-in vocabulary; fetched on first use. yoloe-11, not 26: the yoloe-26 -pf weights
# score random vocabulary classes at conf 1.0 under ultralytics 8.4.137 (e.g. "daffodil" on bus.jpg)
PF_WEIGHTS = f"{WEIGHTS_DIR}/yoloe-11l-seg-pf.pt"
TEXT_ENCODER = f"{WEIGHTS_DIR}/mobileclip2_b.ts"  # yoloe-26's text model (text_model: mobileclip2:b); first use
MAX_LISTED_CLASSES = 200  # longer class lists (the prompt-free vocabulary) are left out of API listings
MODELS_DIR = "models"
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")  # the dataset root, bind-mounted (compose.yaml)
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", "YOLOE")  # the only place written to under the dataset root
WORK_DIR = os.environ.get("WORK_DIR", "/app/work")  # training datasets (symlinks + labels), inside the container
RAW_CONF = 0.01
DEFAULT_CONF = 0.02
DEDUPE_IOU = 0.9
MAX_PARALLEL = max(1, int(os.environ.get("MAX_PARALLEL", "8")))  # models running at once (/api/detect parallel)
# Detectors made by others, run by their own containers (compose.yaml). Reached over the network and
# listed as read-only "external" models when their /health answers. Empty URL = feature off.
YOLORET_URL = os.environ.get("YOLORET_SERVICE_URL", "").rstrip("/")
SAM3_URL = os.environ.get("SAM3_SERVICE_URL", "").rstrip("/")
SAM3_PREFIX = "sam3_"  # sam3_<name>: SAM 3 prompted with the boxes of (legacy, single-image) visual model <name>
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# GPU when the image was built with a CUDA TORCH_VARIANT (compose.gpu.yaml) and a GPU is attached
# Two devices: inference (ONNX Runtime) and training (PyTorch).
#  - NVIDIA (compose.gpu.yaml): both on the GPU.
#  - AMD (compose.amd.yaml, ROCm PyTorch): training on the GPU, inference on the CPU - ONNX Runtime's
#    AMD provider (MIGraphX) needs the full ROCm stack in the image, and inference is the cheap part.
#  - neither: both on the CPU.
TORCH_GPU = torch.cuda.is_available()  # also True for an AMD GPU under ROCm PyTorch (it uses the cuda API)
IS_ROCM = bool(getattr(torch.version, "hip", None))
GPU_NAME = torch.cuda.get_device_name(0) if TORCH_GPU else None
USE_GPU = TORCH_GPU and "CUDAExecutionProvider" in onnxruntime.get_available_providers()  # GPU inference
if USE_GPU and hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls()  # the CUDA/cuDNN libs come from torch's nvidia-* pip packages
TRAIN_DEVICE = "0" if TORCH_GPU else "cpu"
PREDICT_DEVICE = None if USE_GPU else "cpu"  # None: Ultralytics picks the GPU
if USE_GPU:
    DEVICE = f"cuda ({GPU_NAME})"
elif TORCH_GPU:
    DEVICE = f"cpu inference, {'ROCm' if IS_ROCM else 'GPU'} training ({GPU_NAME})"
else:
    DEVICE = "cpu"
print(f"device: {DEVICE}")


class _SharedPoolSession(onnxruntime.InferenceSession):
    """InferenceSession that always runs on ONNX Runtime's one global thread pool."""

    def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
        sess_options = sess_options or onnxruntime.SessionOptions()
        sess_options.use_per_session_threads = False
        super().__init__(path_or_bytes, sess_options, *args, **kwargs)


if not USE_GPU:
    # CPU: every session shares one pool with a thread per core, instead of each session owning a fixed
    # slice of the cores. Threads a model isn't using (fewer models than MAX_PARALLEL, or the last
    # models of an image still running) then help the models that are running. Measured on a 16-thread
    # CPU vs a fixed cpu_count // MAX_PARALLEL threads per session: 1 model 17%, 3 models 34%,
    # 8 models 7% less time per image. Must be set up before the first session is created; the
    # subclass is patched in so Ultralytics' own sessions join the pool too (once the global pool
    # exists, ONNX Runtime refuses sessions with use_per_session_threads left on).
    onnxruntime.capi._pybind_state.set_global_thread_pool_sizes(os.cpu_count() or 1, 1)
    onnxruntime.InferenceSession = _SharedPoolSession

app = Flask(__name__, static_folder="static", static_url_path="/static")
_models_lock = threading.Lock()
_models = {}  # name -> {"model": YOLOE, "lock": Lock, "meta": dict}
_timing_lock = threading.Lock()  # accurate timing: one image at a time, the models one after the other
_pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
DS = Dataset(DATA_ROOT)

KIND_LABELS = {"finetune": "Fine-tuned", "visual": "Visual prompt", "text": "Text prompt", "prompt_free": "Prompt-free"}


def _model_dir(name):
    return os.path.join(MODELS_DIR, name)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_meta(name):
    with open(os.path.join(_model_dir(name), "meta.json")) as f:
        return {"type": "visual", **json.load(f)}  # models baked before types existed are visual


def _load_model(name):
    """A model's registry entry. The ONNX model itself is loaded only when it is used (_ensure_loaded)
    and unloaded again around every job, so idle models take no memory away from training."""
    return {"model": None, "lock": threading.Lock(), "meta": _read_meta(name)}


def _ensure_loaded(entry):
    """Load the entry's ONNX model if it isn't loaded. The caller holds entry["lock"]."""
    if entry["model"] is None:
        model = YOLOE(os.path.join(_model_dir(entry["meta"]["name"]), "model.onnx"))
        # warm up at the real input size (ONNX Runtime allocates its buffers and the thread pool spins up
        # on the first full-size run), so the first measured image isn't slower than the rest
        for _ in range(2):
            model.predict(np.zeros((640, 640, 3), np.uint8), conf=RAW_CONF, verbose=False, device=PREDICT_DEVICE)
        entry["model"] = model
        print(f"loaded model '{entry['meta']['name']}'")
    return entry["model"]


def _release_memory():
    """Hand freed memory back to the system (glibc keeps it for reuse otherwise, so the container's
    memory use would never go down after a job)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _unload_models():
    """Unload every loaded ONNX model; they load again the next time they are used."""
    with _models_lock:
        entries = list(_models.values())
    n = 0
    for e in entries:
        with e["lock"]:
            if e["model"] is not None:
                e["model"] = None
                n += 1
    _release_memory()
    return n


def _ensure_asset(path):
    """Download an Ultralytics asset (weights, text encoder) into modelData/yoloe if it's missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {path}...")
        attempt_download_asset(path)
    if not os.path.exists(path):
        raise RuntimeError(f"could not download {path}")


def _ensure_base_weights():
    """Fetch the base weights on first start. Also point Ultralytics' weights_dir at modelData/yoloe,
    so the text encoder that get_text_pe() loads by bare name is found there instead of being
    downloaded into the working directory."""
    SETTINGS.update(weights_dir=os.path.abspath(WEIGHTS_DIR))
    _ensure_asset(BASE_WEIGHTS)


def _public_meta(meta):
    """Model meta for API listings: the type defaults to visual (models baked before types existed),
    a model without a dataset scope is "legacy", and a huge class list (prompt-free) is replaced by
    its count."""
    out = {"type": "visual", **meta, "num_classes": len(meta.get("class_names", []))}
    out["legacy"] = "scope" not in meta
    if out["num_classes"] > MAX_LISTED_CLASSES:
        out.pop("class_names")
    return out


def _load_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in sorted(os.listdir(MODELS_DIR)):
        d = _model_dir(name)
        if os.path.exists(os.path.join(d, "model.onnx")) and os.path.exists(os.path.join(d, "meta.json")):
            try:
                _models[name] = _load_model(name)
            except Exception as e:
                print(f"failed to load model '{name}': {e}")


def _predict(entry, image, conf):
    with entry["lock"]:  # one YOLOE/predictor object is not thread-safe; different models are
        model = _ensure_loaded(entry)  # loading isn't counted in the time
        t0 = time.perf_counter()  # after the lock: waiting for a free slot or lock isn't counted
        result = model.predict(image, conf=RAW_CONF, verbose=False, device=PREDICT_DEVICE)[0]
    raw = [
        {
            "bbox": [float(v) for v in box.xyxy[0].tolist()],
            "conf": float(box.conf[0]),
            "cls": int(box.cls[0]),
            "name": result.names[int(box.cls[0])],
        }
        for box in result.boxes
    ]
    meta = entry["meta"]
    if meta.get("prompt_class"):
        # text model with several prompts (synonyms) per class: each prompt is its own entry in the model;
        # map it back to its class, keep the prompt that fired. The de-duplication below then merges
        # boxes of the same object found by different synonyms, keeping the highest confidence.
        pc, names = meta["prompt_class"], meta["class_names"]
        for d in raw:
            d["prompt"] = d["name"]
            d["cls"] = pc[d["cls"]]
            d["name"] = names[d["cls"]]
    dets = filter_and_dedupe_detections(raw, conf, iou_thresh=DEDUPE_IOU)
    # (detections, total ms, split): the total is preprocess + inference + postprocess + our filtering
    # for this model; the split is Ultralytics' own preprocess / inference / postprocess timing
    split = {k: round(v, 1) for k, v in result.speed.items()}
    return dets, round((time.perf_counter() - t0) * 1000, 1), split


# =====================================================================================================
# jobs: builds and tests wait in one queue and run one after the other in a background worker (two
# trainings at once would each run at well under half speed on a CPU, and could run out of memory)
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
    # every job starts with no models loaded (a test loads the ones it tests), and they are unloaded
    # again afterwards, so a training never competes with idle models for memory
    n = _unload_models()
    if n:
        job.say(f"unloaded {n} idle model(s)")
    try:
        job.result = job.fn(job)
        job.status = "cancelled" if job.cancel.is_set() else "done"
        job.set(1.0, job.status)
    except train.Cancelled:
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
        _unload_models()


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
def _service_up(url, _cache={}):
    """Whether an external service answers /health (cached 5 s, so listing stays fast when it's down)."""
    if not url:
        return False
    checked, up = _cache.get(url, (0, False))
    if time.time() - checked > 5:
        try:
            up = requests.get(f"{url}/health", timeout=1.5).ok
        except requests.RequestException:
            up = False
        _cache[url] = (time.time(), up)
    return up


def _external_models():
    """Read-only entries for the reachable external detectors: YOLO-ReT (COCO, needs no prompt) and,
    for every legacy visual model (one reference image with boxes), a SAM 3 counterpart."""
    out = {}
    if _service_up(YOLORET_URL):
        out["yoloret"] = {"name": "yoloret", "type": "external", "engine": "yoloret", "num_classes": 80,
                          "created": "", "class_agnostic": True}
    if _service_up(SAM3_URL):
        with _models_lock:
            visual = [e["meta"] for e in _models.values()
                      if e["meta"].get("type", "visual") == "visual" and "bboxes" in e["meta"]
                      and os.path.exists(os.path.join(_model_dir(e["meta"]["name"]), "ref.jpg"))]
        for m in visual:
            name = SAM3_PREFIX + m["name"]
            if NAME_RE.match(name):
                out[name] = {"name": name, "type": "external", "engine": "sam3", "based_on": m["name"],
                             "class_names": m["class_names"], "num_classes": len(m["class_names"]),
                             "created": m["created"], "class_agnostic": True}
    return out


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return "", 200


@app.route("/api/models")
def list_models():
    with _models_lock:
        metas = [_public_meta(e["meta"]) for e in _models.values()]
    metas += _external_models().values()
    return jsonify(models=sorted(metas, key=lambda m: m["name"]), max_parallel=MAX_PARALLEL, device=DEVICE,
                   cpu_threads=os.cpu_count())


# BGR versions of the page's box colours, one per class
BOX_COLORS = [(75, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245),
              (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191)]


def _draw_boxes(img, boxes, cls, names):
    h, w = img.shape[:2]
    lw = max(2, round(max(h, w) / 400))
    scale = max(0.5, max(h, w) / 1200)
    thick = max(1, lw // 2)
    for i, (box, c) in enumerate(zip(boxes, cls)):
        x1, y1, x2, y2 = (round(v) for v in box)
        color = BOX_COLORS[c % len(BOX_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        label = f"#{i} {names[c] if c < len(names) else f'object{c}'}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 if y1 - th - base - 4 < 0 else y1 - th - base - 4  # above the box, or inside at the top
        cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + base + 4), color, -1)
        cv2.putText(img, label, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def _ref_images(name, meta):
    """[(file name, annotated BGR image)] for a visual model: its dataset reference images with the
    prompt boxes, or the single ref.jpg of a legacy model."""
    names = meta.get("class_names") or []
    if meta.get("refs"):
        out = []
        for i, r in enumerate(meta["refs"]):
            img = cv2.imread(os.path.join(DATA_ROOT, r["path"]))
            if img is not None:
                stem = r["path"].split("/")[-2]
                out.append((f"{name}_ref{i + 1}_{stem}.png", _draw_boxes(img, r["bboxes"], r["cls"], names)))
        return out
    img = cv2.imread(os.path.join(_model_dir(name), "ref.jpg"))
    return [(f"{name}_ref_boxes.png", _draw_boxes(img, meta.get("bboxes") or [], meta.get("cls") or [], names))] if img is not None else []


def _model_files(name):
    """What a model was made from, as [(file name, bytes)]: the reference images with their boxes
    (visual), the prompts per class (text), the built-in vocabulary (prompt-free) or the training
    summary (fine-tuned)."""
    meta = _read_meta(name)
    kind = meta["type"]
    if kind == "visual":
        return [(f, cv2.imencode(".png", img)[1].tobytes()) for f, img in _ref_images(name, meta)]
    if kind == "text":
        groups = meta.get("prompts")
        if groups and isinstance(groups[0], list):
            text = "\n".join(f"{c}: " + " | ".join(p) for c, p in zip(meta["class_names"], groups))
        else:
            text = "\n".join(groups or meta["class_names"])
        return [(f"{name}_prompts.txt", (text + "\n").encode())]
    if kind == "finetune":
        return [(f"{name}_training.json", json.dumps(meta.get("training", {}), indent=2).encode())]
    return [(f"{name}_vocabulary.txt", ("\n".join(meta["class_names"]) + "\n").encode())]


@app.route("/api/models/<name>/preview")
def model_preview(name):
    """A small JPEG of the first reference image with its boxes (visual models)."""
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
    """Every model's files from _model_files(), plus its meta.json (and a legacy ref.jpg), one folder
    per model."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(MODELS_DIR)):
            d = _model_dir(name)
            if not NAME_RE.match(name) or not os.path.exists(os.path.join(d, "meta.json")):
                continue
            for fname, data in _model_files(name):
                z.writestr(f"{name}/{fname}", data)
            for extra in ("ref.jpg", "meta.json"):
                if os.path.exists(os.path.join(d, extra)):
                    z.write(os.path.join(d, extra), f"{name}/{extra}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="yoloe_models.zip")


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    if not NAME_RE.match(name):
        abort(400)
    with _models_lock:
        _models.pop(name, None)
    shutil.rmtree(_model_dir(name), ignore_errors=True)
    return jsonify(status="ok")


# ----------------------------------------------------------------------------- building
def _save_model(name, exported_path, meta, extra_files=()):
    """Replace models/<name>/ with the exported ONNX, meta.json and any extra (src, dst name) files."""
    d = _model_dir(name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.move(exported_path, os.path.join(d, "model.onnx"))
    for src, dst in extra_files:
        shutil.move(src, os.path.join(d, dst))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _export_with_embeddings(tmp_dir, set_prompts):
    """Copy the base weights into tmp_dir (export() writes next to the weights file), let
    set_prompts(model) bake class embeddings in, then save/reload them and export to ONNX - the
    same sequence as dockers/yoloe/service.py. Returns (onnx path, prompt.npz path, set_prompts' result)."""
    weights = os.path.join(tmp_dir, os.path.basename(BASE_WEIGHTS))
    shutil.copy(BASE_WEIGHTS, weights)
    prompt_path = os.path.join(tmp_dir, "prompt.npz")
    model = YOLOE(weights)
    info = set_prompts(model)
    model.save_prompt_embeddings(prompt_path)

    model = YOLOE(weights)
    model.load_prompt_embeddings(prompt_path)
    return model.export(format="onnx"), prompt_path, info


def _polygon_mask(obj, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in train._polygon_points(obj):
        cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m.astype(np.float32)


def _build_visual(job, name, spec, class_ids, names, tmp_dir):
    """Visual prompt from the scope's refImages: each allowed in-scope object is a prompt (its box, or
    its polygon as a mask); the embeddings are averaged per class over the reference images."""
    use_masks = spec["options"].get("prompt") == "polygons"
    model_idx = {cid: i for i, cid in enumerate(class_ids)}
    images, refs, skipped = [], [], 0
    for d in DS.scope_images(spec["scope"], "ref"):
        truth = DS.truth(d)
        keep, excluded = DS.split_objects(truth, class_ids, spec["include_uncompleted"], spec["include_covered"])
        skipped += len(excluded)
        if not keep:
            continue
        cls = np.array([model_idx[int(o["class"])] for o in keep], dtype=np.int64)
        boxes = np.array([bbox_xyxy(o) for o in keep], dtype=np.float32)
        path = os.path.join(d, "org.png")
        if use_masks:
            h, w = truth["height"], truth["width"]
            prompts = {"masks": np.stack([_polygon_mask(o, h, w) for o in keep]), "cls": cls}
        else:
            prompts = {"bboxes": boxes, "cls": cls}
        images.append((path, prompts))
        refs.append({"path": DS.rel(path), "bboxes": boxes.round(1).tolist(), "cls": cls.tolist()})
    if not images:
        raise ValueError("no usable reference object in refImages for this scope (check the flag checkboxes)")
    job.say(f"{len(images)} reference image(s), {sum(len(r['cls']) for r in refs)} prompt object(s), "
            f"{skipped} left out because of their flags")
    job.set(0.2, "computing visual prompt embeddings")
    exported, prompt_path, counts = _export_with_embeddings(
        tmp_dir, lambda model: bake_visual_prompts_multi(model, images, names))
    return exported, [(prompt_path, "prompt.npz")], {
        "refs": refs, "prompt": "polygons" if use_masks else "boxes",
        "ref_images_per_class": dict(zip(names, counts)),
    }


def _build_text(job, name, spec, class_ids, names, tmp_dir):
    """Text prompt; every class can have several prompts (synonyms), each baked as its own entry."""
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
    prompt_class = [i for i, g in enumerate(groups) for _ in g]
    _ensure_asset(TEXT_ENCODER)
    job.say(f"{len(flat)} prompt(s) for {len(names)} class(es)")
    job.set(0.2, "embedding the prompts")
    exported, prompt_path, _ = _export_with_embeddings(
        tmp_dir, lambda model: model.set_classes(flat, model.get_text_pe(flat)))
    return exported, [(prompt_path, "prompt.npz")], {"prompts": groups, "prompt_names": flat, "prompt_class": prompt_class}


def _build_prompt_free(job, name, spec, class_ids, names, tmp_dir):
    # the -pf weights carry their own built-in vocabulary; nothing to bake, just export. The scope only
    # decides the test set and the ground truth; it is scored class-agnostically.
    _ensure_asset(PF_WEIGHTS)
    weights = os.path.join(tmp_dir, os.path.basename(PF_WEIGHTS))
    shutil.copy(PF_WEIGHTS, weights)
    job.set(0.2, "exporting")
    model = YOLOE(weights)
    vocab = list(model.names.values())
    return model.export(format="onnx"), [], {"vocabulary_size": len(vocab), "class_names_override": vocab,
                                             "class_agnostic": True, "scope_class_names": names}


def _build_finetune(job, name, spec, class_ids, names, tmp_dir):
    o = spec["options"]
    method = o.get("method", "linear")
    if method not in ("linear", "full"):
        raise ValueError("method must be 'linear' or 'full'")
    epochs = int(o.get("epochs") or (30 if method == "linear" else 50))
    imgsz = int(o.get("imgsz") or 640)
    batch = int(o.get("batch") or 4)
    patience = int(o.get("patience") or 20)
    workers = int(o.get("workers") if o.get("workers") is not None else 2)
    _ensure_asset(TEXT_ENCODER)  # the trainer initialises the class embeddings from the class names
    work = os.path.join(WORK_DIR, name)
    job.say(f"preparing the training data in {work}")
    data_yaml, stats = train.build_yolo_dataset(
        DS, spec["scope"], class_ids, names, spec["include_uncompleted"], spec["include_covered"], work,
        log=job.say, progress=lambda p, m: job.set(0.1 * p, m), cancelled=job.cancel.is_set)
    job.say(f"training: {method}, {epochs} epochs, imgsz {imgsz}, batch {batch}, device "
            f"{('GPU ' + GPU_NAME) if TORCH_GPU else 'cpu'}")
    # training and the ONNX export run in a separate process: all their memory is returned when it exits,
    # and if it runs out of memory only that process is killed, not the server and its queue
    best, exported, summary = train.finetune_in_process(
        {"base_weights": os.path.abspath(BASE_WEIGHTS), "weights_dir": os.path.abspath(WEIGHTS_DIR),
         "data_yaml": data_yaml, "work": work, "method": method, "epochs": epochs, "imgsz": imgsz, "batch": batch,
         "patience": patience, "workers": workers, "device": TRAIN_DEVICE},
        log=job.say, progress=lambda p, m: job.set(0.1 + 0.86 * p, m), cancelled=job.cancel.is_set)
    dst = os.path.join(tmp_dir, "model.onnx")
    shutil.move(exported, dst)
    shutil.copy(best, os.path.join(tmp_dir, "best.pt"))
    # the images (symlinks and greyed copies) are only needed for training
    shutil.rmtree(os.path.join(work, "images"), ignore_errors=True)
    return dst, [(os.path.join(tmp_dir, "best.pt"), "best.pt")], {
        "training": {"method": method, "epochs": epochs, "imgsz": imgsz, "batch": batch, "patience": patience,
                     "data": stats, **summary}}


BUILDERS = {"visual": _build_visual, "text": _build_text, "prompt_free": _build_prompt_free, "finetune": _build_finetune}


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
            exported, extra, info = BUILDERS[kind](job, name, spec, class_ids, names, tmp_dir)
            if job.cancel.is_set():
                raise train.Cancelled()
            class_names = info.pop("class_names_override", names)
            meta = {
                "name": name, "type": kind, "scope": spec["scope"], "scope_label": DS.describe_scope(spec["scope"], all_names),
                "class_ids": class_ids, "class_names": class_names,
                "include_uncompleted": spec["include_uncompleted"], "include_covered": spec["include_covered"],
                "options": spec["options"], "created": _now(), **info,
            }
            meta["build_seconds"] = round(time.perf_counter() - t0, 1)
            _save_model(name, exported, meta, extra)
        entry = _load_model(name)  # registered; its ONNX model loads when it is first used
        with _models_lock:
            _models[name] = entry
        job.say(f"done in {meta['build_seconds']} s")
        return {"model": _public_meta(meta)}

    job = _enqueue("build", f"Build {name} ({KIND_LABELS[kind]}, {DS.describe_scope(spec['scope'], all_names)})", run, target=name)
    return jsonify(status="ok", job=job.as_dict())


# =====================================================================================================
# testing
# =====================================================================================================
def _model_test_plan(name, legacy_scope):
    """(scope, global class ids that are ground truth, class-agnostic?) for testing one model."""
    with _models_lock:
        entry = _models.get(name)
    if entry is None:
        ext = _external_models().get(name)
        if ext is None:
            raise ValueError(f"unknown model {name!r}")
        scope = legacy_scope
        return scope, DS.scope_class_ids(scope), True, None, ext
    meta = entry["meta"]
    if "scope" in meta:
        scope = meta["scope"]
        class_ids = meta["class_ids"]
        agnostic = bool(meta.get("class_agnostic")) or meta["type"] == "prompt_free"
    else:  # legacy: no scope, classes are object0.. -> scored class-agnostically on the scope chosen for the test
        scope = legacy_scope
        class_ids = DS.scope_class_ids(scope)
        agnostic = True
    return scope, class_ids, agnostic, entry, None


def _ext_runner(spec):
    def run(image_bytes):
        if spec["engine"] == "yoloret":
            return _predict_yoloret(image_bytes, RAW_CONF)
        return _predict_sam3(spec, image_bytes, RAW_CONF)
    return run


@app.route("/api/tests", methods=["POST"])
def start_test():
    """Test models on testImages, as a background job. Body: {name?, models: [...], legacy_scope?}.
    No confidence threshold: everything down to the raw 0.01 is kept and scored threshold-free."""
    body = request.get_json(force=True)
    model_names = [m for m in body.get("models") or [] if m]
    if not model_names:
        return jsonify(status="error", message="select at least one model"), 400
    if not _dataset_available():
        return jsonify(status="error", message=f"no dataset at {DATA_ROOT}"), 400
    legacy_scope = body.get("legacy_scope") or {"type": "all"}
    # models can still be waiting in the queue to be built: check now that each exists or will, and
    # work out what it is tested on when the test starts
    pending = _queued_builds()
    try:
        DS.validate_scope(legacy_scope)
        for n in model_names:
            if n not in pending:
                _model_test_plan(n, legacy_scope)
    except ValueError as e:
        return jsonify(status="error", message=str(e)), 400
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", body.get("name") or "_".join(model_names))[:60].strip("_") or "test"

    def run(job):
        plans = {n: _model_test_plan(n, legacy_scope) for n in model_names}  # fails if a queued build failed
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"
        return _run_test(job, run_id, model_names, plans, legacy_scope)

    job = _enqueue("test", f"Test {', '.join(model_names)}", run)
    return jsonify(status="ok", job=job.as_dict())


def _run_test(job, run_id, model_names, plans, legacy_scope):
    all_names = DS.class_names()
    # every test image needed by any model, and which models run on it
    images = {}  # img dir -> [model names]
    for n in model_names:
        scope = plans[n][0]
        for d in DS.scope_images(scope, "test"):
            images.setdefault(d, []).append(n)
    order = sorted(images)
    total = sum(len(v) for v in images.values())
    job.say(f"{len(order)} test image(s), {total} model run(s)")

    per_model = {n: [] for n in model_names}  # for scoring: [{"dets", "gts"}]
    out_images = []
    done = 0
    for d in order:
        if job.cancel.is_set():
            raise train.Cancelled()
        truth = DS.truth(d)
        path = os.path.join(d, "org.png")
        image_bytes = open(path, "rb").read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        rec = {"path": DS.rel(path), "width": w, "height": h, "results": {}, "model_ms": {}, "model_split_ms": {},
               "truth": [{"bbox": [round(v, 1) for v in bbox_xyxy(o)], "class_id": int(o["class"]),
                          "name": o.get("class_name", all_names.get(int(o["class"]), "")),
                          "uncompleted": bool(o.get("uncompleted")), "covered": bool(o.get("covered"))}
                         for o in truth.get("objects", [])]}
        for n in images[d]:
            scope, class_ids, agnostic, entry, ext = plans[n]
            # accurate timing: one model at a time, nothing else running (the lock also holds off /api/detect)
            with _timing_lock:
                if entry is not None:
                    dets, ms, split = _predict(entry, image, RAW_CONF)
                else:
                    dets, ms, split = _ext_runner(ext)(image_bytes)
            idx = {cid: i for i, cid in enumerate(class_ids)}
            gts = [(bbox_xyxy(o), idx[int(o["class"])], flags_of(o)) for o in DS.test_objects(truth, class_ids)]
            per_model[n].append({"dets": [(x["bbox"], x["conf"], x["cls"]) for x in dets], "gts": gts, "rec": rec, "raw": dets})
            rec["results"][n] = dets
            rec["model_ms"][n] = ms
            if split:
                rec["model_split_ms"][n] = split
            done += 1
            job.set(0.95 * done / total, f"{done}/{total}: {n} on {rec['path'].split('/')[-2]}")
        out_images.append(rec)

    job.set(0.96, "scoring")
    metrics_models, matches = [], {}
    for n in model_names:
        scope, class_ids, agnostic, entry, ext = plans[n]
        meta = entry["meta"] if entry else ext
        names = [all_names[c] for c in class_ids]
        m, match, image_matches = evaluate.score_model(per_model[n], names, class_agnostic=agnostic)
        # mark each stored detection as TP/FP at IoU 0.5 for the viewer
        for img, im in zip(per_model[n], image_matches):
            for det, (bits, _) in zip(img["raw"], im):
                det["tp"] = bool(bits & 1)
        ms = [img["rec"]["model_ms"][n] for img in per_model[n]]
        m.update({"name": n, "type": meta.get("type"), "scope": scope, "scope_label": DS.describe_scope(scope, all_names),
                  "legacy": entry is not None and "scope" not in meta, "n_images": len(per_model[n]),
                  "ms_median": round(statistics.median(ms), 1) if ms else None,
                  "ms_mean": round(statistics.fmean(ms), 1) if ms else None})
        metrics_models.append(m)
        matches[n] = {"class_agnostic": agnostic, "classes": match}
        best = m.get("best") or {}
        job.say(f"{n}: AP50 {_fmt(m['ap50'])}, AP50-95 {_fmt(m['ap50_95'])}, best F1 {_fmt(best.get('f1'))} "
                f"@ {best.get('conf')}, median {m['ms_median']} ms")

    out = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(out, exist_ok=True)
    models_info = []
    for n in model_names:
        scope, class_ids, agnostic, entry, ext = plans[n]
        meta = dict(entry["meta"]) if entry else dict(ext)
        if len(meta.get("class_names", [])) > MAX_LISTED_CLASSES:
            meta["class_names"] = None
        models_info.append({"name": n, "type": meta.get("type"), "scope": scope, "class_ids": class_ids,
                            "class_names": None if agnostic else [all_names[c] for c in class_ids],
                            "class_agnostic": agnostic, "meta": meta})
    timing = {"mode": "accurate", "device": DEVICE, "cpu_threads": os.cpu_count()}
    results = {
        "created": _now(), "folder": run_id, "conf_threshold": RAW_CONF, "timing": timing,
        "models": [{"name": mi["name"], "type": mi["type"], "class_names": mi["class_names"] or ["any object"],
                    "class_agnostic": mi["class_agnostic"]} for mi in models_info],
        "images": out_images,
    }
    _write_json(os.path.join(out, "results.json"), results)
    _write_json(os.path.join(out, "metrics.json"), {"created": results["created"], "id": run_id, "timing": timing, "models": metrics_models})
    _write_json(os.path.join(out, "matches.json"), matches)
    _write_json(os.path.join(out, "test.json"), {"id": run_id, "created": results["created"], "models": models_info,
                                                 "legacy_scope": legacy_scope, "raw_conf": RAW_CONF,
                                                 "dedupe_iou": DEDUPE_IOU, "iou_thresholds": evaluate.IOU_THRESHOLDS,
                                                 "timing": timing, "data_root": DATA_ROOT})
    _write_csv(os.path.join(out, "metrics.csv"), metrics_models)
    job.say(f"written to output/YOLOE/{run_id}/")
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
    if part == "csv":
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
# single-image detection (also for other programs) and the external detectors
# =====================================================================================================
def _predict_yoloret(image_bytes, conf):
    t0 = time.perf_counter()
    r = requests.post(f"{YOLORET_URL}/detect", files={"image": ("image.png", image_bytes)}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"YOLO-ReT: {r.status_code} {r.text[:300]}")
    j = r.json()
    ms = j.get("ms", round((time.perf_counter() - t0) * 1000, 1))  # the service's own compute time (else the round trip)
    ids = j.get("class_ids") or [zlib.crc32(n.encode()) % 10000 for n in j["classes"]]  # old service: stable fake id
    raw = [{"bbox": b, "conf": s, "cls": i, "name": n} for b, s, i, n in zip(j["boxes"], j["scores"], ids, j["classes"])]
    # the service's own score threshold (0.2) is baked into the model, so nothing below it ever arrives
    return filter_and_dedupe_detections(raw, conf, iou_thresh=DEDUPE_IOU), ms, None


def _predict_sam3(spec, image_bytes, conf):
    """SAM 3 with the reference boxes of the legacy visual model spec['based_on'], once per class (its
    exemplar mode finds instances similar to the given boxes, it has no notion of classes)."""
    t0 = time.perf_counter()
    d = _model_dir(spec["based_on"])
    meta = _read_meta(spec["based_on"])
    ref_bytes = open(os.path.join(d, "ref.jpg"), "rb").read()
    raw, ms = [], 0.0  # ms: compute time the service measured, summed over the classes
    for c, cname in enumerate(meta["class_names"]):
        boxes = [b for b, k in zip(meta["bboxes"], meta["cls"]) if k == c]
        if not boxes:
            continue
        r = requests.post(
            f"{SAM3_URL}/detect_like_yoloe",
            files={"ref_image": ("ref.jpg", ref_bytes), "target_image": ("target.png", image_bytes)},
            data={"ref_bboxes": json.dumps(boxes)},
            timeout=600,
        )
        if not r.ok:
            raise RuntimeError(f"SAM 3: {r.status_code} {r.text[:300]}")
        j = r.json()
        raw += [{"bbox": b, "conf": s, "cls": c, "name": cname} for b, s in zip(j["boxes"], j["scores"])]
        ms += j.get("ms", 0)
    return filter_and_dedupe_detections(raw, conf, iou_thresh=DEDUPE_IOU), round(ms, 1) or round((time.perf_counter() - t0) * 1000, 1), None


@app.route("/api/detect", methods=["POST"])
def detect():
    conf = float(request.form.get("conf", DEFAULT_CONF))
    wanted = [m for m in request.form.get("models", "").split(",") if m]
    with _models_lock:
        entries = {n: e for n, e in _models.items() if not wanted or n in wanted}
    # external detectors only run when asked for by name (they are slower and not part of "all")
    external = {n: s for n, s in _external_models().items() if n in wanted}
    if not entries and not external:
        return jsonify(status="error", message="no models selected"), 400

    image_bytes = request.files["image"].read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(status="error", message="could not decode image"), 400

    # jobs: model name -> function returning (detections, ms, split)
    jobs = {n: (lambda e=e: _predict(e, image, conf)) for n, e in entries.items()}
    for n, spec in external.items():
        jobs[n] = (lambda spec=spec: _predict_yoloret(image_bytes, conf)) if spec["engine"] == "yoloret" \
            else (lambda spec=spec: _predict_sam3(spec, image_bytes, conf))

    timing = request.form.get("timing", "accurate")
    t_start = time.perf_counter()
    results, errors, model_ms, model_split = {}, {}, {}, {}

    def collect(n, run):
        try:
            results[n], model_ms[n], split = run()
            if split:
                model_split[n] = split
        except Exception as e:
            errors[n] = str(e)

    if timing == "parallel":
        # fast: the models run at once and share the CPU, so each model_ms is inflated by the others
        futures = {n: _pool.submit(collect, n, run) for n, run in jobs.items()}
        for fut in futures.values():
            fut.result()
    else:
        # accurate: one model at a time with the whole machine to itself, and only one image at a time
        # (a lock, in case requests arrive in parallel), so model_ms is what that model takes alone
        with _timing_lock:
            t_start = time.perf_counter()
            for n, run in jobs.items():
                collect(n, run)
    h, w = image.shape[:2]
    return jsonify(width=w, height=h, results=results, errors=errors, model_ms=model_ms, model_split_ms=model_split,
                   timing=timing, timing_ms=round((time.perf_counter() - t_start) * 1000))


os.makedirs(WORK_DIR, exist_ok=True)
if _dataset_available():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
_ensure_base_weights()
_load_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7870, threaded=True)
