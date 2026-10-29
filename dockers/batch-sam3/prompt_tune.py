"""Prompt tuning: learn a few prompt tokens per class with SAM 3 itself frozen.

Each class's prompt is its name's text tokens (frozen) followed by K learned 256-d tokens, initialised
near the text tokens' mean. Training images come from the scope's trainImages (count and seed chosen in
the form); in-scope objects left out because of their flags are greyed out (114), as for YOLOE's
fine-tuning, so the prompt is neither taught them nor punished for finding them.

Loss, per image and class, on SAM 3's DETR-style outputs (200 queries): Hungarian matching of queries to
ground truth (cost: class probability + L1 + GIoU), then sigmoid focal loss on every query's logit,
L1 + GIoU on the matched boxes, and BCE on the presence logit (does the class occur in the image).

The image backbone gets no gradient, so its output is computed once per image and cached on disk
(fp16, last feature level only - the only one the fusion encoder uses): every epoch after the first
skips the expensive part. Runs in its own process (python prompt_tune.py --child), like train.py in
batch-yoloe: all memory is returned when it exits, and running out of memory only kills the child.
"""
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

from dataset import Dataset, bbox_xyxy

GREY = (114, 114, 114)
W_CLS, W_L1, W_GIOU = 2.0, 5.0, 2.0  # matching cost and loss weights (DETR / SAM 3 style)
ACCUMULATE = 4  # images per optimiser step


class Cancelled(Exception):
    pass


class OutOfMemory(RuntimeError):
    pass


def _f(v):
    return "–" if v is None else f"{v:.3f}"


# "train": a number of trainImages picked with a seed; "all": every trainImage; "ref": every refImage
SOURCES = ("train", "all", "ref")


def select_train_images(ds: Dataset, scope, max_images=0, seed=0, source="train"):
    """The images to train on. source "train" / "all": the scope's trainImages in a random order fixed by
    `seed` (validation is held out from the start of it), cut to the first `max_images` for "train"
    (0 = all). source "ref": all of the scope's refImages. Returns (chosen dirs, number available)."""
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


def _polygon_points(obj):
    polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in obj.get("polygons") or [] if len(p) >= 3]
    if not polys:
        x1, y1, x2, y2 = bbox_xyxy(obj)
        polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])]
    return polys


def _image_and_targets(ds, d, class_ids, include_uncompleted, include_covered):
    """(BGR image with left-out objects greyed, {model class index: [xyxy, ...]}, greyed?)."""
    truth = ds.truth(d)
    img = cv2.imread(os.path.join(d, "org.png"))
    keep, excluded = ds.split_objects(truth, class_ids, include_uncompleted, include_covered)
    for o in excluded:
        cv2.fillPoly(img, [p.round().astype(np.int32) for p in _polygon_points(o)], GREY)
    idx = {cid: i for i, cid in enumerate(class_ids)}
    targets = {}
    for o in keep:
        targets.setdefault(idx[int(o["class"])], []).append(bbox_xyxy(o))
    return img, targets, bool(excluded)


def tune(args, log=print, progress=None, cancelled=None):
    """Returns (prompt.pt path, summary)."""
    import functools

    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from torch.utils.checkpoint import checkpoint
    from torchvision.ops import generalized_box_iou, sigmoid_focal_loss

    import evaluate
    from sam3_models import Sam3Engine

    ds = Dataset(args["data_root"])
    class_ids, names = args["class_ids"], args["names"]
    source = args.get("source", "train")
    order, available = select_train_images(ds, args["scope"], args["max_images"], args["seed"], source)
    if source == "ref":
        # a handful of images: nothing is held out, validation uses the same images
        train_dirs, val_dirs = sorted(order), []
        log(f"training on the {len(train_dirs)} reference image(s); validation uses the same images")
    else:
        n_val = max(1, round(len(order) * 0.1)) if len(order) > 1 else 0
        val_dirs, train_dirs = sorted(order[:n_val]), sorted(order[n_val:])
        log(f"{len(order)} of {available} train images (seed {args['seed']}): {len(train_dirs)} train, {len(val_dirs)} val")
    # (cache index, image) to validate on; nothing held out -> the training images (their features are cached)
    eval_items = ([(len(train_dirs) + i, d) for i, d in enumerate(val_dirs)] if val_dirs
                  else list(enumerate(train_dirs)))

    work = args["work"]
    cache = os.path.join(work, "features")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(cache)
    engine = Sam3Engine(args["weights"], args["device"])
    dev = engine.device
    use_amp = dev.type == "cuda"
    # activation checkpointing: each encoder / decoder layer recomputes its activations in the backward
    # pass instead of keeping them. Measured on CPU: peak memory of a step 10.5 -> 7.1 GB, and faster.
    for layers in (engine.model.transformer.encoder.layers, engine.model.transformer.decoder.layers):
        for layer in layers:
            layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)
    log(f"SAM 3 loaded on {dev}")

    # ---------------------------------------------------------------- cached backbone features
    pos_enc = {}
    sizes = {}

    def features(i, d):
        """The image's last-level backbone features, computed once and kept on disk (fp16)."""
        path = os.path.join(cache, f"{i}.pt")
        if not os.path.exists(path):
            img, _, _ = _image_and_targets(ds, d, class_ids, args["include_uncompleted"], args["include_covered"])
            f = engine.features(img)
            torch.save(f["backbone_fpn"][-1].half().cpu(), path)
            pos_enc.setdefault("p", f["vision_pos_enc"][-1])
            sizes[d] = img.shape[:2]
        fpn = torch.load(path, map_location=dev).float()
        return {"backbone_fpn": [fpn], "vision_pos_enc": [pos_enc["p"]]}

    targets = {}
    for d in train_dirs + val_dirs:
        t = ds.truth(d)
        keep, _ = ds.split_objects(t, class_ids, args["include_uncompleted"], args["include_covered"])
        idx = {cid: i for i, cid in enumerate(class_ids)}
        per = {}
        for o in keep:
            per.setdefault(idx[int(o["class"])], []).append(bbox_xyxy(o))
        targets[d] = per
        sizes[d] = (t["height"], t["width"])

    # ---------------------------------------------------------------- the learned tokens
    start_text = args.get("start_text") or names  # the frozen text part of each class's prompt
    log("starting prompts: " + ", ".join(f"{n}: '{t}'" for n, t in zip(names, start_text)))
    text = engine.text_tokens(start_text)
    k = args["tokens"]
    init = torch.stack([t.mean(0) for t, _ in text])  # (C, 256)
    gen = torch.Generator().manual_seed(args["seed"])
    learned = torch.nn.Parameter(init[:, None, :].repeat(1, k, 1).cpu()
                                 + 0.02 * init.std() * torch.randn(len(names), k, init.shape[-1], generator=gen))
    learned.data = learned.data.to(dev)
    opt = torch.optim.AdamW([learned], lr=args["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def prompts():
        out = []
        for c, (t, m) in enumerate(text):
            out.append((torch.cat([t, learned[c].to(t.dtype)]),
                        torch.cat([m, torch.zeros(k, dtype=torch.bool, device=m.device)])))
        return out

    def cxcywh_to_xyxy(b):
        return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)

    def loss_on(d, feats):
        h, w = sizes[d]
        scale = torch.tensor([w, h, w, h], device=dev, dtype=torch.float32)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            logits, boxes, presence = engine.ground(feats, prompts())
        logits, boxes, presence = logits.float(), boxes.float(), presence.float()
        total = 0.0
        for c in range(len(names)):
            gt = torch.tensor(targets[d].get(c, []), device=dev, dtype=torch.float32).reshape(-1, 4) / scale
            target = torch.zeros_like(logits[c])
            n = len(gt)
            if n:
                pred_xyxy = cxcywh_to_xyxy(boxes[c])
                with torch.no_grad():
                    cost = (-W_CLS * logits[c].sigmoid()[:, None] + W_L1 * torch.cdist(pred_xyxy, gt, p=1)
                            - W_GIOU * generalized_box_iou(pred_xyxy, gt))
                    rows, cols = linear_sum_assignment(cost.cpu().numpy())
                rows, cols = torch.as_tensor(rows, device=dev), torch.as_tensor(cols, device=dev)
                target[rows] = 1.0
                total = total + (W_L1 * F.l1_loss(pred_xyxy[rows], gt[cols], reduction="sum")
                                 + W_GIOU * (1 - torch.diag(generalized_box_iou(pred_xyxy[rows], gt[cols]))).sum()) / n
            total = total + W_CLS * sigmoid_focal_loss(logits[c], target, alpha=0.25, gamma=2.0, reduction="sum") / max(n, 1)
            total = total + F.binary_cross_entropy_with_logits(presence[c], torch.tensor(float(n > 0), device=dev))
        return total / len(names)

    @torch.no_grad()
    def validate():
        imgs = []
        for i, d in eval_items:
            if cancelled and cancelled():
                raise Cancelled()
            feats = features(i, d)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                dets = engine.detect(feats, prompts(), sizes[d], conf=0.01)
            truth = ds.truth(d)
            idx = {cid: j for j, cid in enumerate(class_ids)}
            from dataset import flags_of
            gts = [(bbox_xyxy(o), idx[int(o["class"])], flags_of(o)) for o in ds.test_objects(truth, class_ids)]
            imgs.append({"dets": [((x1, y1, x2, y2), s, c) for x1, y1, x2, y2, s, c in dets], "gts": gts})
        m, _, _ = evaluate.score_model(imgs, names, class_agnostic=False)
        return m

    # ---------------------------------------------------------------- training
    epochs = args["epochs"]
    steps_total = epochs * max(1, len(train_dirs))
    rng = random.Random(args["seed"])
    def fitness(m):  # how the best epoch is picked: Ultralytics' weighting of AP50 and AP50-95
        return 0.1 * (m["ap50"] or 0.0) + 0.9 * (m["ap50_95"] or 0.0)

    best = {"fitness": -1.0, "ap50": None, "epoch": 0, "tokens": learned.detach().clone()}
    history = []
    step = 0
    if eval_items:
        m = validate()
        log(f"before training (starting prompts): val AP50 {_f(m['ap50'])}, AP50-95 {_f(m['ap50_95'])}")
        best.update(fitness=fitness(m), ap50=m["ap50"])
        history.append({"epoch": 0, "val_ap50": m["ap50"], "val_ap50_95": m["ap50_95"]})
    for ep in range(epochs):
        idx = list(range(len(train_dirs)))
        rng.shuffle(idx)
        run_loss, n_loss = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(idx):
            if cancelled and cancelled():
                raise Cancelled()
            d = train_dirs[i]
            feats = features(i, d)
            loss = loss_on(d, feats)
            scaler.scale(loss / ACCUMULATE).backward()
            if (j + 1) % ACCUMULATE == 0 or j + 1 == len(idx):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            n_loss += 1
            step += 1
            if progress:
                progress(step / steps_total, f"epoch {ep + 1}/{epochs}, image {j + 1}/{len(idx)}, loss {run_loss / n_loss:.3f}")
        rec = {"epoch": ep + 1, "train_loss": round(run_loss / max(1, n_loss), 4)}
        if eval_items:
            m = validate()
            rec.update(val_ap50=m["ap50"], val_ap50_95=m["ap50_95"])
            if fitness(m) > best["fitness"]:
                best.update(fitness=fitness(m), ap50=m["ap50"], epoch=ep + 1, tokens=learned.detach().clone())
        else:
            best.update(epoch=ep + 1, tokens=learned.detach().clone())
        history.append(rec)
        log(f"epoch {ep + 1}/{epochs}: loss {rec['train_loss']}"
            + (f", val AP50 {_f(rec['val_ap50'])}, AP50-95 {_f(rec['val_ap50_95'])}" if eval_items else ""))

    out = os.path.join(work, "prompt.pt")
    torch.save({"text": list(start_text), "tokens": [t.cpu() for t in best["tokens"]]}, out)
    shutil.rmtree(cache, ignore_errors=True)
    summary = {"epochs_done": epochs, "best_epoch": best["epoch"], "best_val_ap50": best["ap50"],
               "history": history, "data": {"available_images": available, "max_images": args["max_images"] or 0,
                                            "seed": args["seed"], "train_images": len(train_dirs), "val_images": len(val_dirs) or len(train_dirs),
                                            "val_is_train": not val_dirs, "source": source, "used_images": len(order)}}
    if best["epoch"] == 0:
        log("no epoch beat the starting prompts on validation: the tokens are kept at their starting point")
    return out, summary


# ----------------------------------------------------------------------------- separate process
def _child_main(args):
    """The training process (python prompt_tune.py --child): one JSON message per line on stdout;
    a line "cancel" on stdin cancels."""
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
        path, summary = tune(args, log=lambda m: send("log", m), progress=lambda p, m: send("progress", p, m),
                             cancelled=cancel.is_set)
        send("done", path, summary)
    except Cancelled:
        send("cancelled")
    except BaseException as e:  # noqa: BLE001 - every failure has to reach the server as a message
        import traceback
        send("error", f"{type(e).__name__}: {e}", traceback.format_exc())


def tune_in_process(args, log=print, progress=None, cancelled=None):
    """Run tune() in a new process and relay its messages. Returns (prompt.pt path, summary).
    Raises Cancelled, OutOfMemory (the process was killed) or RuntimeError (the training failed)."""
    import subprocess
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
            raise OutOfMemory("the training process ran out of memory and was killed (try fewer train images, "
                              "or give Docker more memory)")
        raise RuntimeError(f"the training process stopped unexpectedly (exit code {code})")
    msg = result[0]
    if msg[0] == "cancelled":
        raise Cancelled()
    if msg[0] == "error":
        log(msg[2].strip().splitlines()[-1])
        raise RuntimeError(f"training failed: {msg[1]}")
    _, path, summary = msg
    return path, summary


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    _child_main(json.loads(sys.argv[2]))
