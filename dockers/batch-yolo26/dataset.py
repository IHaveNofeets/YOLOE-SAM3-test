"""Reading the labelled dataset root (bind-mounted at /data):

    <root>/{trainImages,testImages,refImages}/<dataset>/crop_offsets.json
    <root>/{trainImages,testImages,refImages}/<dataset>/<image>/{crop.png, org.png, truth.json}

truth.json: {image, width, height, objects: [{id, class, class_name, bbox: {x, y, w, h},
polygons: [[[x, y], ...]], uncompleted, covered?, letter?}]}, in org.png's pixels. Class ids are global
across datasets.

crop.png is org.png with x1 / x2 pixels cut off the left / right and y1 / y2 off the top / bottom
(crop_offsets.json: {"OrgToCrop": {x1, x2, y1, y2}}). When a dataset has crop_offsets.json, its images are
the crops and truth() returns the truth in crop pixels: boxes and polygons shifted and clipped to the
crop, objects the crop cuts flagged uncompleted (cut off by the image edge), objects outside it dropped.
A dataset without it uses org.png as it is.

A model's *scope* is either one class in one dataset, {"type": "single", "dataset": ..., "class_id": n},
or every class except "Other" in every dataset, {"type": "all"}.
"""
import json
import os
import threading

SPLITS = {"train": "trainImages", "test": "testImages", "ref": "refImages"}
IMAGE_FILE = "org.png"
CROP_FILE = "crop.png"
OFFSETS_FILE = "crop_offsets.json"
TRUTH_FILE = "truth.json"
FLAGS = ("uncompleted", "covered")


def is_other(class_name):
    return str(class_name).strip().lower() == "other"


def bbox_xyxy(obj):
    b = obj["bbox"]
    return [float(b["x"]), float(b["y"]), float(b["x"] + b["w"]), float(b["y"] + b["h"])]


def flags_of(obj):
    """Bitmask of the object's flags: 1 = uncompleted, 2 = covered."""
    return (1 if obj.get("uncompleted") else 0) | (2 if obj.get("covered") else 0)


def _clip_polygon(pts, w, h):
    """The part of a polygon inside the rectangle 0..w x 0..h (Sutherland-Hodgman)."""
    for inside, cut in ((lambda p: p[0] >= 0, lambda a, b: _cut_x(a, b, 0)),
                        (lambda p: p[0] <= w, lambda a, b: _cut_x(a, b, w)),
                        (lambda p: p[1] >= 0, lambda a, b: _cut_y(a, b, 0)),
                        (lambda p: p[1] <= h, lambda a, b: _cut_y(a, b, h))):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
        pts = out
        if not pts:
            break
    return pts


def _cut_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return [x, a[1] + t * (b[1] - a[1])]


def _cut_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return [a[0] + t * (b[0] - a[0]), y]


def crop_truth(truth, off):
    """truth.json (org.png pixels) -> the same in crop pixels, off = crop_offsets.json's OrgToCrop."""
    dx, dy = off["x1"], off["y1"]
    w, h = truth["width"] - off["x1"] - off["x2"], truth["height"] - off["y1"] - off["y2"]
    objects = []
    for o in truth.get("objects", []):
        b = o["bbox"]
        x1, y1, x2, y2 = b["x"] - dx, b["y"] - dy, b["x"] + b["w"] - dx, b["y"] + b["h"] - dy
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # outside the crop
        o = dict(o, bbox={"x": cx1, "y": cy1, "w": cx2 - cx1, "h": cy2 - cy1})
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            o["uncompleted"] = True  # the crop cuts it
        polys = (_clip_polygon([[px - dx, py - dy] for px, py in p], w, h) for p in o.get("polygons") or [])
        o["polygons"] = [p for p in polys if len(p) >= 3]
        objects.append(o)
    return dict(truth, width=w, height=h, objects=objects)


class Dataset:
    def __init__(self, root):
        self.root = root
        self._scan = None
        self._lock = threading.Lock()
        self._offsets = {}  # dataset folder -> OrgToCrop, or None when it has no crop_offsets.json

    # ---------- files ----------
    def split_dir(self, split, dataset=None):
        d = os.path.join(self.root, SPLITS[split])
        return os.path.join(d, dataset) if dataset else d

    def datasets(self):
        names = set()
        for split in SPLITS:
            d = self.split_dir(split)
            if os.path.isdir(d):
                names |= {n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))}
        return sorted(names)

    def image_dirs(self, split, dataset):
        """Image folders (with both the image and truth.json) of one dataset in one split, sorted."""
        d = self.split_dir(split, dataset)
        if not os.path.isdir(d):
            return []
        image_file = IMAGE_FILE if self.crop_offsets(d) is None else CROP_FILE
        out = []
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(os.path.join(p, image_file)) and os.path.isfile(os.path.join(p, TRUTH_FILE)):
                out.append(p)
        return out

    def rel(self, path):
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    def crop_offsets(self, dataset_dir):
        """The dataset folder's OrgToCrop offsets, or None when it has no crop_offsets.json."""
        dataset_dir = os.path.normpath(dataset_dir)
        if dataset_dir not in self._offsets:
            path = os.path.join(dataset_dir, OFFSETS_FILE)
            off = None
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    o2c = json.load(f)["OrgToCrop"]
                off = {k: int(o2c.get(k, 0)) for k in ("x1", "x2", "y1", "y2")}
            self._offsets[dataset_dir] = off
        return self._offsets[dataset_dir]

    def image_path(self, img_dir):
        """The image to use: crop.png when the dataset has crop_offsets.json, else org.png."""
        off = self.crop_offsets(os.path.dirname(os.path.normpath(img_dir)))
        return os.path.join(img_dir, IMAGE_FILE if off is None else CROP_FILE)

    def scope_image_file(self, scope):
        """The image file a model of this scope is built from: crop.png or org.png ("+"-joined if its
        datasets differ). Kept in a model's meta.json, to tell when the images have changed since."""
        files = {IMAGE_FILE if self.crop_offsets(self.split_dir(s, ds)) is None else CROP_FILE
                 for ds in self.scope_datasets(scope) for s in SPLITS if os.path.isdir(self.split_dir(s, ds))}
        return "+".join(sorted(files)) or IMAGE_FILE

    def truth(self, img_dir):
        """truth.json, in the pixels of image_path() (see the module docstring)."""
        with open(os.path.join(img_dir, TRUTH_FILE), encoding="utf-8") as f:
            truth = json.load(f)
        off = self.crop_offsets(os.path.dirname(os.path.normpath(img_dir)))
        return truth if off is None else crop_truth(truth, off)

    # ---------- overview ----------
    def scan(self, refresh=False):
        """Per dataset and split: image count and, per class, object / uncompleted / covered counts.
        Also the global class id -> name map."""
        with self._lock:
            if self._scan is not None and not refresh:
                return self._scan
            classes = {}
            out = []
            for ds in self.datasets():
                entry = {"name": ds, "splits": {}}
                for split in SPLITS:
                    dirs = self.image_dirs(split, ds)
                    per = {}
                    for d in dirs:
                        for o in self.truth(d).get("objects", []):
                            cid = int(o["class"])
                            classes.setdefault(cid, o.get("class_name", f"class{cid}"))
                            c = per.setdefault(cid, {"objects": 0, "uncompleted": 0, "covered": 0})
                            c["objects"] += 1
                            for fl in FLAGS:
                                c[fl] += bool(o.get(fl))
                    entry["splits"][split] = {"images": len(dirs), "classes": {str(k): v for k, v in sorted(per.items())}}
                out.append(entry)
            self._scan = {
                "root": self.root,
                "datasets": out,
                "classes": {str(k): v for k, v in sorted(classes.items())},
                "other_ids": [k for k, v in sorted(classes.items()) if is_other(v)],
            }
            return self._scan

    def class_names(self):
        return {int(k): v for k, v in self.scan()["classes"].items()}

    # ---------- scopes ----------
    def scope_class_ids(self, scope):
        """Global class ids a model is built for, in model-index order."""
        if scope["type"] == "single":
            cid = int(scope["class_id"])
            if is_other(self.class_names().get(cid, "")):
                raise ValueError('the "Other" class can not be used')
            return [cid]
        if scope["type"] == "all":
            return [cid for cid, name in sorted(self.class_names().items()) if not is_other(name)]
        raise ValueError(f"unknown scope type {scope.get('type')!r}")

    def scope_datasets(self, scope):
        if scope["type"] == "single":
            if scope["dataset"] not in self.datasets():
                raise ValueError(f"unknown dataset {scope['dataset']!r}")
            return [scope["dataset"]]
        return self.datasets()

    def scope_images(self, scope, split):
        return [d for ds in self.scope_datasets(scope) for d in self.image_dirs(split, ds)]

    def validate_scope(self, scope):
        if not isinstance(scope, dict) or scope.get("type") not in ("single", "all"):
            raise ValueError("scope must be {type: 'single', dataset, class_id} or {type: 'all'}")
        ids = self.scope_class_ids(scope)
        self.scope_datasets(scope)
        if scope["type"] == "single" and ids[0] not in self.class_names():
            raise ValueError(f"unknown class id {ids[0]}")
        return ids

    @staticmethod
    def describe_scope(scope, names):
        if scope["type"] == "single":
            return f"{names.get(int(scope['class_id']), scope['class_id'])} in {scope['dataset']}"
        return "all classes, all datasets"

    # ---------- objects ----------
    @staticmethod
    def split_objects(truth, class_ids, include_uncompleted=False, include_covered=False):
        """For building / training: (objects to use, in-scope objects left out because of their flags)."""
        ids = set(class_ids)
        keep, excluded = [], []
        for o in truth.get("objects", []):
            if int(o["class"]) not in ids:
                continue
            if (o.get("uncompleted") and not include_uncompleted) or (o.get("covered") and not include_covered):
                excluded.append(o)
            else:
                keep.append(o)
        return keep, excluded

    @staticmethod
    def test_objects(truth, class_ids):
        """For testing: every in-scope object, flagged ones included."""
        ids = set(class_ids)
        return [o for o in truth.get("objects", []) if int(o["class"]) in ids]
