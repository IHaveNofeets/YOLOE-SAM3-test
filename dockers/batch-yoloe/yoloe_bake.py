import numpy as np

from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


class _MaskSafePredictor(YOLOEVPSegPredictor):
    """Works around a shape bug in the base predictor's mask-prompt path.

    Letterboxing a single-channel mask (via the shared image pre_transform) leaves a
    stray trailing channel axis - (H, W, 1) instead of (H, W) - which get_visuals()
    doesn't expect (it wants plain (N, H, W)), and crashes with a spatial-dims mismatch.
    The bboxes path is untouched: it's delegated straight to the base implementation.
    """

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        if bboxes is not None and len(bboxes):
            return super()._process_single_image(dst_shape, src_shape, category, bboxes, masks)
        resized_masks = self.pre_transform(masks)
        resized_masks = [m[..., 0] if m.ndim == 3 else m for m in resized_masks]
        masks = np.stack(resized_masks)
        masks[masks == 114] = 0
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)


def bake_visual_prompts(model, refer_image, prompts, names=None, predictor_cls=_MaskSafePredictor):
    """Bake visual prompts (bboxes or masks) from a reference image into the model as fixed classes.

    Mirrors what YOLOE.predict(..., refer_image=...) does internally, except it also accepts
    mask prompts ({"masks": ..., "cls": ...}), which the public predict() API doesn't support -
    it hard-requires a 'bboxes' key even when a mask is what should actually drive the embedding.

    names (list[str], optional): one display name per class index (0..num_cls-1). Defaults to
    "object0", "object1", ... - what the exported model's detections get labeled with.
    """
    cls = prompts["cls"]
    num_cls = len(set(cls.tolist()))
    names = names if names is not None else [f"object{i}" for i in range(num_cls)]
    assert len(names) == num_cls, f"Expected {num_cls} names, got {len(names)}"

    predictor = predictor_cls(
        overrides={"task": model.task, "mode": "predict", "save": False, "verbose": False, "batch": 1},
        _callbacks=model.callbacks,
    )
    model.model.model[-1].nc = num_cls
    model.model.names = list(names)
    predictor.set_prompts(dict(prompts))
    predictor.setup_model(model=model.model, verbose=False)

    vpe = predictor.get_vpe(refer_image)
    model.set_classes(model.model.names, vpe)


def bake_visual_prompts_multi(model, images, names, predictor_cls=_MaskSafePredictor):
    """Bake visual prompts taken from several reference images into the model as fixed classes.

    images: [(image path, prompts)] where prompts is {"bboxes": (N, 4) xyxy, "cls": (N,)} or
    {"masks": (N, H, W), "cls": (N,)}, with cls the *model* class index (0..len(names)-1) of each prompt.
    Each image's embeddings are computed on their own (classes renumbered 0..k-1 inside the image, which
    get_vpe needs), then each class's embeddings are averaged over the images that show it and
    L2-normalised, like YOLOE does for several prompts of one class inside one image.
    """
    import torch

    num_cls = len(names)
    sums, counts = [None] * num_cls, [0] * num_cls
    predictor = predictor_cls(
        overrides={"task": model.task, "mode": "predict", "save": False, "verbose": False, "batch": 1},
        _callbacks=model.callbacks,
    )
    model.model.names = list(names)
    for path, prompts in images:
        cls = np.asarray(prompts["cls"], dtype=np.int64)
        present, local = np.unique(cls, return_inverse=True)
        local_prompts = {k: v for k, v in prompts.items() if k != "cls"}
        local_prompts["cls"] = local.astype(np.int64)
        model.model.model[-1].nc = len(present)
        predictor.set_prompts(local_prompts)
        predictor.setup_model(model=model.model, verbose=False)
        vpe = predictor.get_vpe(path)  # (1, k, D), one embedding per class present in this image
        vpe = vpe.reshape(len(present), -1).float().cpu()
        for row, c in enumerate(present):
            sums[c] = vpe[row] if sums[c] is None else sums[c] + vpe[row]
            counts[c] += 1
    missing = [names[c] for c in range(num_cls) if not counts[c]]
    if missing:
        raise ValueError(f"no reference object for class(es): {', '.join(missing)}")
    vpe = torch.stack([sums[c] / counts[c] for c in range(num_cls)])
    vpe = torch.nn.functional.normalize(vpe, dim=-1).unsqueeze(0)
    model.model.model[-1].nc = num_cls
    model.set_classes(list(names), vpe)
    return counts
