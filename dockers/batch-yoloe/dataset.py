"""Reading the labelled dataset root (bind-mounted at /data):

    <root>/{trainImages,testImages,refImages}/<dataset>/<image>/{org.png, truth.json}

truth.json: {image, width, height, objects: [{id, class, class_name, bbox: {x, y, w, h},
polygons: [[[x, y], ...]], uncompleted, covered?, letter?}]}. Class ids are global across datasets.

A model's *scope* is either one class in one dataset, {"type": "single", "dataset": ..., "class_id": n},
or every class except "Other" in every dataset, {"type": "all"}.
"""
import json
import os
import threading

SPLITS = {"train": "trainImages", "test": "testImages", "ref": "refImages"}
IMAGE_FILE = "org.png"
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


class Dataset:
    def __init__(self, root):
        self.root = root
        self._scan = None
        self._lock = threading.Lock()

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
        """Image folders (with both org.png and truth.json) of one dataset in one split, sorted."""
        d = self.split_dir(split, dataset)
        if not os.path.isdir(d):
            return []
        out = []
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(os.path.join(p, IMAGE_FILE)) and os.path.isfile(os.path.join(p, TRUTH_FILE)):
                out.append(p)
        return out

    def rel(self, path):
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    @staticmethod
    def truth(img_dir):
        with open(os.path.join(img_dir, TRUTH_FILE), encoding="utf-8") as f:
            return json.load(f)

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
