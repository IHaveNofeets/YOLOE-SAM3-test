"""SAM 3 for the dataset tool: one engine, four ways of prompting it.

Every model kind ends up as a list of prompts, one per class (or per synonym for text). A prompt is a
sequence of 256-d tokens with a padding mask - the same thing SAM 3's own predictor builds internally:

    text       the text encoder's tokens of the prompt
    exemplar   tokens of the word "visual" (or the class name) + the geometry encoder's tokens of
               exemplar boxes drawn on the *same* image - so the exemplar crops are pasted into a panel
               beside each test image (SAM 3 only searches the image the boxes are on)
    visual     "visual" (or the class name) + exemplar tokens encoded once on the reference images and
               stored; used on test images directly, without the panel (experimental: SAM 3 was trained
               with the exemplar on the searched image)
    tuned      the class name's text tokens + a few tokens learned by prompt_tune.py

The image backbone runs once per image; then all prompts of a model go through the fusion encoder and
decoder together as one batch. Only boxes are produced (the mask head is skipped: scoring is on boxes).
"""
import cv2
import numpy as np
import torch
import torchvision
from ultralytics.models.sam.modules.sam import SAM2Model
from ultralytics.models.sam.predict import SAM3SemanticPredictor

IMGSZ = 1008  # SAM 3's native input size
NMS_IOU = 0.7  # per prompt, as Ultralytics' SAM 3 predictor
PANEL_H = 1024  # exemplar panel: stored at this height, resized to the test image's height when used
PANEL_ASPECT = 0.32  # panel width / height: about a quarter of the 2112x1664 test images' width
CROP_MARGIN = 0.15  # context around an exemplar object in its crop, as a fraction of the box size
VISUAL_WORD = "visual"  # the text SAM 3 pairs with exemplar boxes when no class name is given


class Sam3Engine:
    def __init__(self, weights, device="cpu", imgsz=IMGSZ):
        self.device = torch.device("cuda:0" if device not in ("cpu", None) and torch.cuda.is_available() else "cpu")
        self.pred = SAM3SemanticPredictor(overrides=dict(
            conf=0.01, task="segment", mode="predict", model=weights, imgsz=imgsz, verbose=False, save=False,
            device=str(self.device.index if self.device.type == "cuda" else "cpu")))
        self.pred.setup_model(model=None, verbose=False)
        self.model = self.pred.model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._text_cache = {}

    # ------------------------------------------------------------------ image
    @torch.no_grad()
    def features(self, image):
        """Backbone features of a BGR image (the expensive part, once per image)."""
        self.pred.set_image(image)
        feats = self.pred.features
        self.pred.features = None
        return feats

    def _flat(self, features, batch=1):
        _, img_feats, img_pos, sizes = SAM2Model._prepare_backbone_features(self.model, dict(features), batch=batch)
        return img_feats, img_pos, sizes

    # ------------------------------------------------------------------ prompt tokens
    @torch.no_grad()
    def text_tokens(self, text):
        """[(tokens (L, 256), padding mask (L,))] for each string; cached."""
        missing = [t for t in dict.fromkeys(text) if t not in self._text_cache]
        if missing:
            out = self.model.backbone.forward_text(missing)
            feats, mask = out["language_features"], out["language_mask"]  # (L, n, 256), (n, L) True = padding
            for i, t in enumerate(missing):
                keep = ~mask[i]
                self._text_cache[t] = (feats[keep, i].clone(), mask[i, keep].clone())  # drop the padding
        return [self._text_cache[t] for t in text]

    @torch.no_grad()
    def box_tokens(self, features, src_shape, boxes, labels=None):
        """Geometry-encoder tokens of xyxy pixel boxes on the image `features` came from:
        (tokens (S, 256), mask (S,)). labels: 1 positive, 0 negative (default all positive)."""
        bboxes, lab = self.pred._prepare_geometric_prompts(src_shape[:2], np.asarray(boxes, np.float32),
                                                           None if labels is None else np.asarray(labels))
        geo = self.pred._get_dummy_prompt(1)
        for i in range(len(bboxes)):
            geo.append_boxes(bboxes[[i]], lab[[i]])
        img_feats, img_pos, sizes = self._flat(features)
        tok, mask = self.model._encode_prompt(img_feats, img_pos, sizes, geo)  # (S, 1, 256), (1, S)
        return tok[:, 0].clone(), mask[0].clone()

    # ------------------------------------------------------------------ detection
    def ground(self, features, prompts):
        """Run the fusion encoder + decoder for a batch of prompts [(tokens (S, 256), mask (S,)), ...].
        Returns (logits (n, Q), boxes cxcywh 0-1 (n, Q, 4), presence logits (n,))."""
        n = len(prompts)
        s_max = max(t.shape[0] for t, _ in prompts)
        tok = torch.zeros(s_max, n, prompts[0][0].shape[-1], device=self.device, dtype=prompts[0][0].dtype)
        mask = torch.ones(n, s_max, device=self.device, dtype=torch.bool)
        for i, (t, m) in enumerate(prompts):
            tok[: t.shape[0], i] = t
            mask[i, : t.shape[0]] = m
        img_feats, img_pos, sizes = self._flat(features, batch=n)
        m = self.model
        enc = m._run_encoder(img_feats, img_pos, sizes, tok, mask)
        out, _ = m._run_decoder(memory=enc["encoder_hidden_states"], pos_embed=enc["pos_embed"],
                                src_mask=enc["padding_mask"], out={}, prompt=tok, prompt_mask=mask, encoder_out=enc)
        return out["pred_logits"][..., 0], out["pred_boxes"], out["presence_logit_dec"].reshape(n)

    @torch.no_grad()
    def detect(self, features, prompts, src_shape, conf=0.01):
        """Detections [(x1, y1, x2, y2, score, prompt index)] in pixels of an image of src_shape (h, w).
        The score is SAM 3's: sigmoid(logit) x sigmoid(presence), as Ultralytics' predictor."""
        logits, boxes, presence = self.ground(features, prompts)
        scores = logits.sigmoid() * presence.sigmoid()[:, None]
        h, w = src_shape[:2]
        out = []
        for i in range(len(prompts)):
            keep = scores[i] > conf
            if not keep.any():
                continue
            b = boxes[i][keep]
            xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1) * torch.tensor([w, h, w, h], device=b.device)
            s = scores[i][keep]
            k = torchvision.ops.nms(xyxy.float(), s.float(), NMS_IOU)
            for (x1, y1, x2, y2), sc in zip(xyxy[k].tolist(), s[k].tolist()):
                out.append((x1, y1, x2, y2, sc, i))
        return out


# ---------------------------------------------------------------------- exemplar panel
def crop_object(image, box, margin=CROP_MARGIN):
    """A crop of `image` around box (xyxy) with some context; returns (crop, box inside the crop)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    cx1, cy1 = int(max(0, np.floor(x1 - mx))), int(max(0, np.floor(y1 - my)))
    cx2, cy2 = int(min(w, np.ceil(x2 + mx))), int(min(h, np.ceil(y2 + my)))
    return image[cy1:cy2, cx1:cx2].copy(), [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def build_panel(crops, height=PANEL_H, aspect=PANEL_ASPECT):
    """Pack crops [(crop, box in crop)] into one panel image (grey background) of the given height.
    The grid's column count is chosen so the crops are shown as large as possible.
    Returns (panel BGR, [box in panel coordinates])."""
    width = max(1, round(height * aspect))
    n = len(crops)
    best = None
    for cols in range(1, n + 1):
        rows = -(-n // cols)
        cw, ch = width / cols, height / rows
        scale = min(min(cw / c.shape[1], ch / c.shape[0]) for c, _ in crops)
        if best is None or scale > best[0]:
            best = (scale, cols, rows, cw, ch)
    _, cols, rows, cw, ch = best
    panel = np.full((height, width, 3), 114, np.uint8)
    boxes = []
    for i, (crop, box) in enumerate(crops):
        r, c = divmod(i, cols)
        s = min(cw / crop.shape[1], ch / crop.shape[0])
        nw, nh = max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))
        x0 = int(c * cw + (cw - nw) / 2)
        y0 = int(r * ch + (ch - nh) / 2)
        panel[y0:y0 + nh, x0:x0 + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        boxes.append([x0 + box[0] * s, y0 + box[1] * s, x0 + box[2] * s, y0 + box[3] * s])
    return panel, boxes


def compose(panel, panel_boxes, image):
    """The panel, resized to the image's height, on the left of the image.
    Returns (composite, panel boxes in composite pixels, x offset of the image)."""
    s = image.shape[0] / panel.shape[0]
    p = cv2.resize(panel, (max(1, round(panel.shape[1] * s)), image.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([p, image]), [[v * s for v in b] for b in panel_boxes], p.shape[1]


def from_composite(dets, offset, width, height):
    """Detections on the composite -> on the image: drop the ones centred on the panel, shift, clip."""
    out = []
    for x1, y1, x2, y2, sc, i in dets:
        if (x1 + x2) / 2 < offset:
            continue
        out.append((float(np.clip(x1 - offset, 0, width)), float(np.clip(y1, 0, height)),
                    float(np.clip(x2 - offset, 0, width)), float(np.clip(y2, 0, height)), sc, i))
    return out


# ---------------------------------------------------------------------- one built model
class Sam3Model:
    """A built model (its meta.json and stored prompt files) made runnable on an engine.
    predict(image) -> [(x1, y1, x2, y2, score, class index, prompt label)]."""

    def __init__(self, engine, meta, model_dir):
        self.engine, self.meta = engine, meta
        kind = meta["type"]
        names = meta["class_names"]
        self.kind = kind
        if kind == "text":
            prompts = meta["prompt_names"]
            self.prompts = engine.text_tokens(prompts)
            self.prompt_class = meta["prompt_class"]
            self.labels = prompts
        elif kind == "exemplar":
            self.panel = cv2.imread(f"{model_dir}/panel.png")
            p = meta["panel"]
            self.panel_boxes, self.panel_cls = p["boxes"], p["cls"]
            self.words = [n if meta["options"].get("with_text") else VISUAL_WORD for n in names]
            self.negatives = bool(meta["options"].get("negatives"))
            self.prompt_class = list(range(len(names)))
            self.labels = self.words
        else:  # visual / tuned: stored tokens per class
            store = torch.load(f"{model_dir}/prompt.pt", map_location=engine.device)
            words = store["text"]
            texts = engine.text_tokens(words)
            self.prompts = []
            for (t, m), extra in zip(texts, store["tokens"]):
                extra = extra.to(engine.device, t.dtype)
                self.prompts.append((torch.cat([t, extra]), torch.cat([m, torch.zeros(len(extra), dtype=torch.bool, device=m.device)])))
            self.prompt_class = list(range(len(names)))
            self.labels = words

    def predict(self, image, conf=0.01):
        h, w = image.shape[:2]
        e = self.engine
        if self.kind == "exemplar":
            comp, boxes, offset = compose(self.panel, self.panel_boxes, image)
            feats = e.features(comp)
            texts = e.text_tokens(self.words)
            prompts = []
            for c, (t, m) in enumerate(texts):
                pos = [b for b, k in zip(boxes, self.panel_cls) if k == c]
                neg = [b for b, k in zip(boxes, self.panel_cls) if k != c] if self.negatives else []
                bt, bm = e.box_tokens(feats, comp.shape, pos + neg, [1] * len(pos) + [0] * len(neg))
                prompts.append((torch.cat([t, bt]), torch.cat([m, bm])))
            dets = from_composite(e.detect(feats, prompts, comp.shape, conf), offset, w, h)
        else:
            feats = e.features(image)
            dets = e.detect(feats, self.prompts, image.shape, conf)
        return [(x1, y1, x2, y2, sc, self.prompt_class[i], self.labels[i]) for x1, y1, x2, y2, sc, i in dets]
