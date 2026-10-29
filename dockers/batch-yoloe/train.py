"""Fine-tuning YOLOE on the dataset's trainImages.

build_yolo_dataset() writes an Ultralytics YOLO-seg dataset into a work folder inside the container:
images are symlinks to the originals, except where an in-scope object is left out because of its flags
(uncompleted / covered not ticked). Those images are copied with the left-out objects filled with
Ultralytics' padding grey (114), so the model is neither trained on them nor punished for detecting them.
Objects of classes outside the scope stay in the image unlabelled, so they act as background.
"""
import os
import random
import shutil
import sys

import cv2
import numpy as np

from dataset import Dataset, bbox_xyxy

GREY = (114, 114, 114)


class Cancelled(Exception):
    pass


def _polygon_points(obj):
    polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in obj.get("polygons") or [] if len(p) >= 3]
    if not polys:  # no polygon: the box as a 4-point polygon
        x1, y1, x2, y2 = bbox_xyxy(obj)
        polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])]
    return polys


def _label_line(obj, cls, w, h):
    polys = _polygon_points(obj)
    if len(polys) > 1:
        # several parts of one object: join them into one outline the way Ultralytics' COCO converter does
        from ultralytics.data.converter import merge_multi_segment
        pts = np.concatenate(merge_multi_segment([p.reshape(-1).tolist() for p in polys]), axis=0)
    else:
        pts = polys[0]
    pts = pts / np.array([w, h], dtype=np.float64)
    pts = np.clip(pts, 0, 1)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))


# "train": a number of trainImages picked with a seed; "all": every trainImage; "ref": every refImage
SOURCES = ("train", "all", "ref")


def select_train_images(ds: Dataset, scope, max_images=0, seed=0, source="train"):
    """The images to train on. source "train" / "all": the scope's trainImages in a random order fixed by
    `seed` (validation is held out from the start of it), cut to the first `max_images` for "train"
    (0 = all) - the same seed with a larger count gives a superset of a smaller one. source "ref": all of
    the scope's refImages. Returns (chosen dirs, number available)."""
    split = "ref" if source == "ref" else "train"
    dirs = ds.scope_images(scope, split)
    if not dirs:
        raise ValueError(f"no {'reference' if split == 'ref' else 'training'} images in this scope")
    if split == "ref":
        return sorted(dirs), len(dirs)
    order = list(dirs)
    random.Random(seed).shuffle(order)
    if source == "train" and max_images and max_images > 0:
        order = order[:max_images]
    return order, len(dirs)


def build_yolo_dataset(ds: Dataset, scope, class_ids, names, include_uncompleted, include_covered, work,
                       val_fraction=0.1, seed=0, max_images=0, source="train", log=print, progress=None,
                       cancelled=None):
    """Returns (data.yaml path, stats). With source "ref" (a handful of images) nothing is held out:
    the same images are used for training and validation."""
    model_idx = {cid: i for i, cid in enumerate(class_ids)}
    order, available = select_train_images(ds, scope, max_images, seed, source)
    n_val = max(1, round(len(order) * val_fraction)) if len(order) > 1 and source != "ref" else 0
    split_of = {d: ("val" if i < n_val else "train") for i, d in enumerate(order)}

    shutil.rmtree(work, ignore_errors=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(work, sub))
    stats = {"source": source, "available_images": available, "max_images": max_images or 0, "seed": seed,
             "used_images": len(order), "train_images": 0, "val_images": 0, "objects": 0, "excluded_objects": 0,
             "greyed_images": 0}
    dirs = sorted(order)
    for n, d in enumerate(dirs):
        if cancelled and cancelled():
            raise Cancelled()
        split = split_of[d]
        truth = ds.truth(d)
        w, h = truth["width"], truth["height"]
        keep, excluded = ds.split_objects(truth, class_ids, include_uncompleted, include_covered)
        stem = ds.rel(d).split("/", 1)[1].replace("/", "__")  # <dataset>__<image>
        src = os.path.join(d, "org.png")
        if excluded:
            img = cv2.imread(src)
            for o in excluded:
                cv2.fillPoly(img, [p.round().astype(np.int32) for p in _polygon_points(o)], GREY)
            cv2.imwrite(os.path.join(work, "images", split, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["greyed_images"] += 1
        else:
            os.symlink(src, os.path.join(work, "images", split, stem + ".png"))
        with open(os.path.join(work, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(_label_line(o, model_idx[int(o["class"])], w, h) for o in keep))
        stats[f"{split}_images"] += 1
        stats["objects"] += len(keep)
        stats["excluded_objects"] += len(excluded)
        if progress:
            progress((n + 1) / len(dirs), f"preparing images {n + 1}/{len(dirs)}")
    if n_val == 0:  # nothing held out (ref images, or a single image): validate on the training images
        for sub in ("images", "labels"):
            os.rmdir(os.path.join(work, sub, "val"))
            os.symlink("train", os.path.join(work, sub, "val"))
        stats["val_images"] = stats["train_images"]
        stats["val_is_train"] = True

    yaml_path = os.path.join(work, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name!r}\n")
    log(f"dataset: {stats}")
    return yaml_path, stats


def linear_probe_freeze(model):
    """Everything frozen except the last layer of the head's class-embedding branches (cv3 and the
    end-to-end head's one2one_cv3), which YOLOEPESegTrainer re-creates as trainable."""
    head = len(model.model.model) - 1
    freeze = [str(i) for i in range(head)]
    for name, _ in model.model.model[-1].named_children():
        if name not in ("cv3", "one2one_cv3"):
            freeze.append(f"{head}.{name}")
    for branch in ("cv3", "one2one_cv3"):
        if getattr(model.model.model[-1], branch, None) is not None:
            for i in range(len(getattr(model.model.model[-1], branch))):
                freeze += [f"{head}.{branch}.{i}.0", f"{head}.{branch}.{i}.1"]
    return freeze


def finetune(base_weights, data_yaml, work, method, epochs, imgsz, batch, patience, workers, device,
             log=print, progress=None, cancelled=None):
    """Train with YOLOEPESegTrainer. Returns (best weights path, train summary)."""
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

    model = YOLOE(base_weights)
    freeze = linear_probe_freeze(model) if method == "linear" else None
    state = {"epoch": 0}

    def on_batch_end(trainer):
        if cancelled and cancelled():
            raise Cancelled()
        if progress:
            nb = max(1, len(trainer.train_loader))
            i = getattr(trainer, "_batch_i", 0) + 1
            trainer._batch_i = i
            done = (trainer.epoch + min(i, nb) / nb) / trainer.epochs
            progress(done, f"epoch {trainer.epoch + 1}/{trainer.epochs}, batch {min(i, nb)}/{nb}")

    def on_epoch_start(trainer):
        trainer._batch_i = 0

    def on_fit_epoch_end(trainer):
        state["epoch"] = min(trainer.epoch + 1, trainer.epochs)
        m = {k: round(float(v), 4) for k, v in (trainer.metrics or {}).items()}
        state["last_metrics"] = m
        label = f"epoch {trainer.epoch + 1}/{trainer.epochs}" if trainer.epoch + 1 <= trainer.epochs else "final validation (best weights)"
        log(f"{label}: " + ", ".join(f"{k.split('/')[-1]} {v}" for k, v in m.items()
                                                                        if "mAP50" in k or "precision" in k or "recall" in k))

    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    args = dict(data=data_yaml, trainer=YOLOEPESegTrainer, epochs=epochs, imgsz=imgsz, batch=batch,
                patience=patience, workers=workers, device=device, project=os.path.join(work, "runs"),
                name="train", exist_ok=True, plots=False, verbose=False)
    if freeze:
        args["freeze"] = freeze
    model.train(**args)
    trainer = model.trainer
    best = trainer.best if os.path.exists(trainer.best) else trainer.last
    summary = {"epochs_done": state["epoch"], "final_val": state.get("last_metrics", {}),
               "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None}
    return str(best), summary


# ----------------------------------------------------------------------------- separate process
# Training runs in a child process started fresh for each fine-tune: PyTorch keeps the memory a
# training used, so back-to-back trainings in the server process pile up until the container is
# killed. A child gives everything back when it exits, and if it does run out of memory only the child
# is killed - the server, its queue and the other jobs carry on.

class OutOfMemory(RuntimeError):
    pass


def _child_main(args):
    """The training process (python train.py --child): train, export to ONNX. Talks to the server through
    one JSON message per line on stdout; a line "cancel" on stdin cancels."""
    import json
    import sys
    import threading

    out = sys.stdout
    sys.stdout = sys.stderr  # Ultralytics' own printing goes to stderr, stdout carries only our messages

    def send(*msg):
        out.write(json.dumps(msg) + chr(10))
        out.flush()

    cancel = threading.Event()

    def watch_stdin():
        for line in sys.stdin:
            if line.strip() == "cancel":
                cancel.set()
        cancel.set()  # the server went away: stop too

    threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        from ultralytics import YOLOE
        from ultralytics.utils import SETTINGS
        SETTINGS.update(weights_dir=args["weights_dir"])  # where the text encoder is (get_text_pe loads it by name)

        best, summary = finetune(
            args["base_weights"], args["data_yaml"], args["work"], args["method"], args["epochs"], args["imgsz"],
            args["batch"], args["patience"], args["workers"], args["device"],
            log=lambda m: send("log", m), progress=lambda p, m: send("progress", p, m), cancelled=cancel.is_set)
        send("progress", 1.0, "exporting")
        exported = YOLOE(best).export(format="onnx")
        send("done", str(best), str(exported), summary)
    except Cancelled:
        send("cancelled")
    except BaseException as e:  # noqa: BLE001 - every failure has to reach the server as a message
        import traceback
        send("error", f"{type(e).__name__}: {e}", traceback.format_exc())


def finetune_in_process(args, log=print, progress=None, cancelled=None):
    """Run the training in a new process (this file with --child) and relay its messages.
    Returns (best .pt, exported .onnx, summary). Raises Cancelled, OutOfMemory (the process was killed)
    or RuntimeError (the training failed)."""
    import json
    import subprocess
    import sys
    import threading

    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", json.dumps(args)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
                            cwd=os.getcwd())
    log(f"training process started (pid {proc.pid})")
    result = []

    def read():
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg[0] == "log":
                log(msg[1])
            elif msg[0] == "progress":
                if progress:
                    progress(msg[1], msg[2])
            else:
                result.append(msg)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    sent_cancel = False
    try:
        while proc.poll() is None:
            if cancelled and cancelled() and not sent_cancel:
                try:
                    proc.stdin.write("cancel" + chr(10))
                    proc.stdin.flush()
                except OSError:
                    pass
                sent_cancel = True
            reader.join(timeout=1.0)
        reader.join(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if not result:
        code = proc.returncode
        if code == -9 or code == 137:
            raise OutOfMemory("the training process ran out of memory and was killed (try a smaller batch or "
                              "image size, or give Docker more memory)")
        raise RuntimeError(f"the training process stopped unexpectedly (exit code {code})")
    msg = result[0]
    if msg[0] == "cancelled":
        raise Cancelled()
    if msg[0] == "error":
        log(msg[2].strip().splitlines()[-1])
        raise RuntimeError(f"training failed: {msg[1]}")
    _, best, exported, summary = msg
    return best, exported, summary


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    import json as _json
    _child_main(_json.loads(sys.argv[2]))
