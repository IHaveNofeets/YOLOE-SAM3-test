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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
import onnxruntime
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

import evaluate
import train
from dataset import Dataset, bbox_xyxy, flags_of
from detect_utils import filter_and_dedupe_detections


# The COCO-pretrained YOLO26 weights a fine-tune starts from are downloaded on first use into this
# container's own bind mount (dockers/batch-yolo26/modelData/yolo26); a model's ONNX file is kept in models/
WEIGHTS_DIR = "modelData/yolo26"
BASE_RE = re.compile(r"^yolo26[nsmlx](-seg)?$")  # yolo26l (boxes) or yolo26l-seg (boxes + masks, trains on the polygons)
DEFAULT_BASE = "yolo26l-seg"
MODELS_DIR = "models"
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")  # the dataset root, bind-mounted (compose.yaml)
OUTPUT_DIR = os.path.join(DATA_ROOT, "output", "YOLO26")  # the only place written to under the dataset root
WORK_DIR = os.environ.get("WORK_DIR", "/app/work")  # training datasets (symlinks + labels), inside the container
RAW_CONF = 0.01
DEFAULT_CONF = 0.02
DEDUPE_IOU = 0.9
MAX_PARALLEL = max(1, int(os.environ.get("MAX_PARALLEL", "8")))  # models running at once (/api/detect parallel)
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
_models = {}  # name -> {"model": YOLO, "lock": Lock, "meta": dict}
_timing_lock = threading.Lock()  # accurate timing: one image at a time, the models one after the other
_pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
DS = Dataset(DATA_ROOT)

KIND_LABELS = {"finetune": "Fine-tuned"}


def _model_dir(name):
    return os.path.join(MODELS_DIR, name)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_meta(name):
    with open(os.path.join(_model_dir(name), "meta.json")) as f:
        return {"type": "finetune", **json.load(f)}


def _load_model(name):
    """A model's registry entry. The ONNX model itself is loaded only when it is used (_ensure_loaded)
    and unloaded again around every job, so idle models take no memory away from training."""
    return {"model": None, "lock": threading.Lock(), "meta": _read_meta(name)}


def _ensure_loaded(entry):
    """Load the entry's ONNX model if it isn't loaded. The caller holds entry["lock"]."""
    if entry["model"] is None:
        model = YOLO(os.path.join(_model_dir(entry["meta"]["name"]), "model.onnx"))
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
    if TORCH_GPU:
        torch.cuda.empty_cache()


def _unload_model(entry):
    """Unload one model's ONNX model; it loads again the next time it is used."""
    with entry["lock"]:
        entry["model"] = None
    _release_memory()


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
    """Download an Ultralytics asset (the base weights) into modelData/yolo26 if it's missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {path}...")
        attempt_download_asset(path)
    if not os.path.exists(path):
        raise RuntimeError(f"could not download {path}")


def _public_meta(meta):
    """Model meta for API listings, with whether it is outdated (see _outdated)."""
    out = {"type": "finetune", **meta, "num_classes": len(meta.get("class_names", []))}
    out["outdated"] = _outdated(out)
    return out


SLOW_KINDS = ("finetune",)


def _outdated(meta):
    """A model trained on dataset images that have changed since (e.g. org.png, before the crops).
    Models built before meta.json recorded the images were built from org.png."""
    if "scope" not in meta:
        return False
    try:
        return meta.get("images", "org.png") != DS.scope_image_file(meta["scope"])
    except ValueError:  # its dataset is gone
        return False


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
    with entry["lock"]:  # one predictor object is not thread-safe; different models are
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
    return jsonify(models=sorted(metas, key=lambda m: m["name"]), max_parallel=MAX_PARALLEL, device=DEVICE,
                   cpu_threads=os.cpu_count())


def _model_files(name):
    """What a model was made from, as [(file name, bytes)]: the training summary."""
    meta = _read_meta(name)
    return [(f"{name}_training.json", json.dumps(meta.get("training", {}), indent=2).encode())]


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
    """Every model's files from _model_files(), plus its meta.json, one folder
    per model."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(MODELS_DIR)):
            d = _model_dir(name)
            if not NAME_RE.match(name) or not os.path.exists(os.path.join(d, "meta.json")):
                continue
            for fname, data in _model_files(name):
                z.writestr(f"{name}/{fname}", data)
            for extra in ("meta.json",):
                if os.path.exists(os.path.join(d, extra)):
                    z.write(os.path.join(d, extra), f"{name}/{extra}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="yolo26_models.zip")


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


def _build_finetune(job, name, spec, class_ids, names, tmp_dir):
    o = spec["options"]
    base = o.get("base") or DEFAULT_BASE
    if not BASE_RE.match(base):
        raise ValueError("base must be yolo26n / s / m / l / x, optionally with -seg (e.g. yolo26l-seg)")
    method = o.get("method", "full")
    if method not in ("full", "head"):
        raise ValueError("method must be 'full' or 'head'")
    epochs = int(o.get("epochs") or (50 if method == "full" else 30))
    imgsz = int(o.get("imgsz") or 640)
    batch = int(o.get("batch") or 4)
    patience = int(o.get("patience") or 20)
    workers = int(o.get("workers") if o.get("workers") is not None else 2)
    # train: a number of trainImages (+ seed); all: every trainImage; ref: every refImage
    source = o.get("source") if o.get("source") in train.SOURCES else "train"
    max_images = max(0, int(o.get("max_images") or 0)) if source == "train" else 0  # 0 = all
    seed = int(o.get("seed") or 0)
    focus = o.get("focus") if o.get("focus") in train.FOCUS else "fp"  # fp (default) / balanced / fn
    base_weights = os.path.join(WEIGHTS_DIR, base + ".pt")
    job.set(0.0, f"getting {base}.pt")
    _ensure_asset(base_weights)
    work = os.path.join(WORK_DIR, name)
    job.say(f"preparing the training data in {work}")
    data_yaml, stats = train.build_yolo_dataset(
        DS, spec["scope"], class_ids, names, spec["include_uncompleted"], spec["include_covered"], work,
        seed=seed, max_images=max_images, source=source, log=job.say, progress=lambda p, m: job.set(0.1 * p, m),
        cancelled=job.cancel.is_set)
    if source == "ref":
        job.say(f"training on the {stats['train_images']} reference image(s); validation uses the same images")
    else:
        job.say(f"{stats['used_images']} of {stats['available_images']} train images (seed {seed}): "
                f"{stats['train_images']} train, {stats['val_images']} val")
    job.say(f"training {base}: {method}, {epochs} epochs, imgsz {imgsz}, batch {batch}, focus {FOCUS_LABELS[focus]}, device "
            f"{('GPU ' + GPU_NAME) if TORCH_GPU else 'cpu'}")
    # training and the ONNX export run in a separate process: all their memory is returned when it exits,
    # and if it runs out of memory only that process is killed, not the server and its queue
    best, exported, summary = train.finetune_in_process(
        {"base_weights": os.path.abspath(base_weights),
         "data_yaml": data_yaml, "work": work, "method": method, "epochs": epochs, "imgsz": imgsz, "batch": batch,
         "patience": patience, "workers": workers, "device": TRAIN_DEVICE, "focus": focus},
        log=job.say, progress=lambda p, m: job.set(0.1 + 0.86 * p, m), cancelled=job.cancel.is_set)
    dst = os.path.join(tmp_dir, "model.onnx")
    shutil.move(exported, dst)
    shutil.copy(best, os.path.join(tmp_dir, "best.pt"))
    # the images (symlinks and greyed copies) are only needed for training
    shutil.rmtree(os.path.join(work, "images"), ignore_errors=True)
    return dst, [(os.path.join(tmp_dir, "best.pt"), "best.pt")], {
        "base": base,
        "training": {"base": base, "method": method, "epochs": epochs, "imgsz": imgsz, "batch": batch,
                     "patience": patience, "focus": focus, "source": source, "max_images": max_images, "seed": seed,
                     "data": stats, **summary}}


FOCUS_LABELS = {"fp": "fewer false positives", "balanced": "balanced", "fn": "fewer false negatives"}
BUILDERS = {"finetune": _build_finetune}


@app.route("/api/models", methods=["POST"])
def build_model():
    """Build a model from the dataset, as a background job.
    Body: {name, kind, scope, include_uncompleted, include_covered, options}."""
    try:
        job = _queue_build(request.get_json(force=True))
    except LookupError as e:
        return jsonify(status="error", message=str(e)), 409
    except ValueError as e:
        return jsonify(status="error", message=str(e)), 400
    return jsonify(status="ok", job=job.as_dict())


@app.route("/api/models/rebuild", methods=["POST"])
def rebuild_outdated():
    """Queue every outdated model's build again, with the settings it was built with, quick kinds first."""
    with _models_lock:
        metas = [e["meta"] for e in _models.values() if _outdated(e["meta"])]
    queued, skipped = [], []
    for m in sorted(metas, key=lambda m: (m.get("type", "finetune") in SLOW_KINDS, m["name"])):
        spec = {"name": m["name"], "kind": m.get("type", "finetune"), "scope": m["scope"],
                "include_uncompleted": m.get("include_uncompleted"), "include_covered": m.get("include_covered"),
                "options": dict(m.get("options") or {})}
        try:
            _queue_build(spec)
            queued.append(m["name"])
        except (ValueError, LookupError) as e:
            skipped.append({"name": m["name"], "reason": str(e)})
    return jsonify(status="ok", queued=queued, skipped=skipped)


def _queue_build(spec):
    """Check a build spec and queue the build; returns the job. ValueError: the spec is invalid,
    LookupError: a build of that name is already in the queue."""
    name = spec.get("name", "")
    if not NAME_RE.match(name):
        raise ValueError("name must be 1-64 chars of A-Z a-z 0-9 _ -")
    kind = spec.get("kind") or "finetune"
    if kind not in BUILDERS:
        raise ValueError(f"unknown kind '{kind}'")
    if not _dataset_available():
        raise ValueError(f"no dataset at {DATA_ROOT}")
    class_ids = DS.validate_scope(spec.get("scope"))
    if name in _queued_builds():
        raise LookupError(f"a build of '{name}' is already in the queue")
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
                "options": spec["options"], "images": DS.scope_image_file(spec["scope"]), "created": _now(), **info,
            }
            meta["build_seconds"] = round(time.perf_counter() - t0, 1)
            _save_model(name, exported, meta, extra)
        entry = _load_model(name)  # registered; its ONNX model loads when it is first used
        with _models_lock:
            _models[name] = entry
        job.say(f"done in {meta['build_seconds']} s")
        return {"model": _public_meta(meta)}

    return _enqueue("build", f"Build {name} ({KIND_LABELS[kind]}, {DS.describe_scope(spec['scope'], all_names)})", run, target=name)


# =====================================================================================================
# testing
# =====================================================================================================
def _model_test_plan(name):
    """(scope, global class ids that are ground truth, class-agnostic?, registry entry, None) for testing one model."""
    with _models_lock:
        entry = _models.get(name)
    if entry is None:
        raise ValueError(f"unknown model {name!r}")
    meta = entry["meta"]
    if "scope" not in meta:
        raise ValueError(f"model {name!r} has no dataset scope")
    return meta["scope"], meta["class_ids"], False, entry, None


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
    # models can still be waiting in the queue to be built: check now that each exists or will, and
    # work out what it is tested on when the test starts
    pending = _queued_builds()
    try:
        for n in model_names:
            if n not in pending:
                _model_test_plan(n)
    except ValueError as e:
        return jsonify(status="error", message=str(e)), 400
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", body.get("name") or "_".join(model_names))[:60].strip("_") or "test"

    def run(job):
        plans = {n: _model_test_plan(n) for n in model_names}  # fails if a queued build failed
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"
        return _run_test(job, run_id, model_names, plans)

    job = _enqueue("test", f"Test {', '.join(model_names)}", run)
    return jsonify(status="ok", job=job.as_dict())


def _run_test(job, run_id, model_names, plans):
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
    recs, truths = {}, {}  # img dir -> its record in results.json (filled in model by model) / truth
    done = 0
    # Model by model, each over all of its images and then unloaded: on a GPU every loaded ONNX model keeps
    # its own memory pool, so a test of many models, all kept loaded, runs out of GPU memory.
    for n in model_names:
        scope, class_ids, agnostic, entry, ext = plans[n]
        idx = {cid: i for i, cid in enumerate(class_ids)}
        for d in order:
            if n not in images[d]:
                continue
            if job.cancel.is_set():
                raise train.Cancelled()
            path = DS.image_path(d)
            image_bytes = open(path, "rb").read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if d not in recs:
                truths[d] = truth = DS.truth(d)
                h, w = image.shape[:2]
                recs[d] = {"path": DS.rel(path), "width": w, "height": h, "results": {}, "model_ms": {}, "model_split_ms": {},
                           "truth": [{"bbox": [round(v, 1) for v in bbox_xyxy(o)], "class_id": int(o["class"]),
                                      "name": o.get("class_name", all_names.get(int(o["class"]), "")),
                                      "uncompleted": bool(o.get("uncompleted")), "covered": bool(o.get("covered"))}
                                     for o in truth.get("objects", [])]}
            truth, rec = truths[d], recs[d]
            # accurate timing: one model at a time, nothing else running (the lock also holds off /api/detect)
            with _timing_lock:
                dets, ms, split = _predict(entry, image, RAW_CONF)
            gts = [(bbox_xyxy(o), idx[int(o["class"])], flags_of(o)) for o in DS.test_objects(truth, class_ids)]
            per_model[n].append({"dets": [(x["bbox"], x["conf"], x["cls"]) for x in dets], "gts": gts, "rec": rec, "raw": dets})
            rec["results"][n] = dets
            rec["model_ms"][n] = ms
            if split:
                rec["model_split_ms"][n] = split
            done += 1
            job.set(0.95 * done / total, f"{done}/{total}: {n} on {rec['path'].split('/')[-2]}")
        if entry is not None:
            _unload_model(entry)
    out_images = [recs[d] for d in order]

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
                  "n_images": len(per_model[n]),
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
                                                 "raw_conf": RAW_CONF,
                                                 "dedupe_iou": DEDUPE_IOU, "iou_thresholds": evaluate.IOU_THRESHOLDS,
                                                 "timing": timing, "data_root": DATA_ROOT})
    _write_csv(os.path.join(out, "metrics.csv"), metrics_models)
    job.say(f"written to output/YOLO26/{run_id}/")
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
# single-image detection (also for other programs)
# =====================================================================================================
@app.route("/api/detect", methods=["POST"])
def detect():
    conf = float(request.form.get("conf", DEFAULT_CONF))
    wanted = [m for m in request.form.get("models", "").split(",") if m]
    with _models_lock:
        entries = {n: e for n, e in _models.items() if not wanted or n in wanted}
    if not entries:
        return jsonify(status="error", message="no models selected"), 400

    image = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(status="error", message="could not decode image"), 400

    # jobs: model name -> function returning (detections, ms, split)
    jobs = {n: (lambda e=e: _predict(e, image, conf)) for n, e in entries.items()}

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
_load_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7872, threaded=True)
