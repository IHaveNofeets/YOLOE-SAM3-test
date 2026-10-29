# YOLOE / SAM 3 dataset tools

Two web tools, each running in its own Docker container, that build object detectors from a labelled
dataset, test them on held-out images and compare the results. Both use the same dataset, the same
scoring, and write their results in the same format, so YOLOE and SAM 3 can be compared directly.

| Container | Port | Model | Kinds of model it builds |
|---|---|---|---|
| `batch-yoloe` | [7870](http://localhost:7870) | YOLOE (Ultralytics, ONNX) | visual prompt, text prompt, prompt-free, fine-tune |
| `batch-sam3` | [7871](http://localhost:7871) | SAM 3 (Meta, via Ultralytics) | text prompt, exemplar prompt, visual prompt, prompt tuning |

Both run on CPU, an NVIDIA GPU or (on Linux) an AMD GPU. SAM 3 is slow on a CPU (about 35–40 s per
image), so test and train it on a GPU where possible.

## The dataset

Both tools read one dataset folder, mounted into the containers at `/data`:

```
Dataset/
├── trainImages/<dataset>/<image>/org.png + truth.json   for fine-tuning / prompt tuning
├── testImages/<dataset>/<image>/org.png + truth.json    for testing (never used for building)
├── refImages/<dataset>/<image>/org.png + truth.json     reference images for visual / exemplar prompts
└── output/
    ├── YOLOE/<date>_<name>/                              test results of batch-yoloe
    └── SAM3/<date>_<name>/                               test results of batch-sam3
```

`truth.json` lists the objects of an image: `{image, width, height, objects: [{id, class, class_name,
bbox: {x, y, w, h}, polygons, uncompleted, covered?}]}`. Class ids are shared across datasets. The class
"Other" is never used for building or scoring.

Objects flagged **uncompleted** (cut off by the image edge) or **covered** (partly hidden) are left out
when building a model, unless their checkbox is ticked. In training images they are greyed out, so the
model is neither taught them nor punished for finding them. They are **always** part of the test ground
truth, and the results show the recall on them separately.

The tools only ever write to `output/YOLOE/` and `output/SAM3/`.

## Getting started

### Windows (CPU)

Needs Docker Desktop. The dataset is expected at `E:\Dataset`. To use another folder, put
`DATASET_ROOT=D:/path/to/Dataset` in a `.env` file next to `compose.yaml`.

```bash
docker compose up -d --build batch-yoloe batch-sam3
```

Then open http://localhost:7870 and http://localhost:7871. Start just one of them by naming only that
service.

Docker Desktop's memory limit matters: SAM 3 needs about 7 GB while loading. If containers get killed,
give WSL more memory in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=11GB`).

### Linux

On Linux, a GPU speeds both tools up a lot. An NVIDIA GPU also works on Windows with
`compose.gpu.yaml`; an AMD GPU needs Linux.

#### Ubuntu with an NVIDIA or AMD GPU

A setup script does all of it on Ubuntu:
```bash
bash tools/setup_ubuntu_nvidia.sh --dataset /path/to/Dataset   # NVIDIA
bash tools/setup_ubuntu_amd.sh --dataset /path/to/Dataset      # AMD
```

Both scripts install Docker, write `.env`, build and start both containers, and test the GPU inside
them. If a script installs a driver, reboot and run it again. Both are safe to run again; use `--help`
for the options.

- **NVIDIA:** installs the NVIDIA driver (CUDA 13 needs version 580 or newer) and the NVIDIA Container
  Toolkit, and uses `compose.gpu.yaml`. Training and testing both run on the GPU.
- **AMD (ROCm):** checks the AMD kernel driver, adds you to the render / video groups, and uses
  `compose.amd.yaml`. For cards ROCm doesn't officially support, such as the RX 6750 XT (gfx1031), it
  sets `HSA_OVERRIDE_GFX_VERSION=10.3.0`. batch-sam3 runs fully on the GPU; batch-yoloe trains on the
  GPU and tests on the CPU (ONNX Runtime).

### SAM 3 weights

`sam3.pt` (3.4 GB) is not downloaded automatically, because access to it on Hugging Face is gated. Put it
in `dockers/batch-sam3/modelData/sam3/sam3.pt`. The page shows a warning while it is missing. YOLOE's
weights are downloaded on first start.

## Using the tools

Both pages work the same way, from top to bottom.

1. **Dataset:** the datasets, image counts, and objects per class (with uncompleted / covered counts).
2. **Build a model:** pick a kind, a scope and the options, then press *Build*.
   - **Scope:** either one class in one dataset, or all classes (except "Other") in all datasets.
   - **Queue:** builds and tests wait in one queue and run one after another. The bar at the top shows
     progress, the log and the queue.
3. **Models:** the built models, with what they were made from. *Download* gives the reference images
   with their boxes, the prompts or the training summary.
4. **Test:** pick models and press *Run test*.
   - **Test images:** each model runs on the test images of its own scope, one model at a time, so the
     times are real.
   - **Confidence:** every detection is kept, and the confidence is chosen afterwards with the slider.
     Precision, recall, F1, the counts and the charts update live, and *best F1 @ conf* jumps to the best
     threshold.
   - **Quick look (batch-sam3 only):** *Test images* limits the run to the first N images.
5. **Images:** every test image with the detections, the ground truth (dashed) and TP / FP colouring.

### Kinds of model

**batch-yoloe**

| Kind | Made from | Notes |
|---|---|---|
| Visual prompt | refImages | each reference object's box (or polygon) becomes a prompt; averaged per class |
| Text prompt | words | one or more names per class (e.g. "2x4 brick", "lego brick"); a hit by any counts as the class |
| Prompt-free | nothing | built-in vocabulary; scored class-agnostically ("is there an object") |
| Fine-tune | trainImages (or refImages) | *linear probe* trains only the class layer (fast); *full* trains everything |

**batch-sam3**

| Kind | Made from | Notes |
|---|---|---|
| Text prompt | words | as YOLOE. Wording matters a lot, so add synonyms |
| Exemplar prompt | refImages | crops of the reference objects are placed beside each test image, and SAM 3 finds objects like them |
| Visual prompt | refImages | the reference boxes are encoded once into prompt tokens, used without the crops (faster, experimental) |
| Prompt tuning | trainImages (or refImages) | SAM 3 stays frozen; a few prompt tokens per class are learned after a starting text |

For exemplar and visual prompts, *add the class name as text* gives SAM 3 the class name next to the
boxes. Without it, SAM 3 gets its default word "visual".

### Training data (fine-tune and prompt tuning)

| Option | Uses |
|---|---|
| trainImages: a number of them (+ seed) | that many training images, picked at random with the seed. The same number and seed always give the same images, and a larger number includes the smaller set |
| trainImages: all of them | every training image in the scope; the seed picks which 10 % are held out for validation |
| refImages: all of them | every reference image in the scope; with so few, validation uses the same images |

## Test output

Each test writes a folder `output/YOLOE/<date>_<name>/` or `output/SAM3/<date>_<name>/`:

| File | Contents |
|---|---|
| `metrics.csv` | one row per model and class: AP50, AP50-95, P / R / F1 at the best F1 and at fixed confidences, recall on uncompleted / covered objects, median and mean ms per image |
| `metrics.json` | the same, plus the P / R / F1 curves |
| `results.json` | every detection on every image, with the ground truth (for the image viewer) |
| `matches.json` | each detection's confidence and whether it matched, so P / R / F1 can be recomputed at any confidence |
| `test.json` | the settings and the models' build information |

**How detections are scored:**
- **Matching:** greedy by confidence at IoU ≥ 0.5.
- **AP:** AP50 and AP50-95 are COCO-style and don't depend on the confidence.
- **Timing:** one model at a time per image, covering preprocessing, inference and postprocessing.

## Other programs

Both containers also take single images over HTTP:

```bash
curl -F image=@photo.png -F models=model_a,model_b -F conf=0.25 http://localhost:7870/api/detect
```

The answer lists each model's detections (`bbox` as x1, y1, x2, y2 in pixels, `conf`, `cls`, `name`) and
its time in ms.

## Project layout

```
compose.yaml              both services (CPU)
compose.gpu.yaml          NVIDIA override:  docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
compose.amd.yaml          AMD override (Linux only, needs .env from setup_ubuntu_amd.sh)
dockers/batch-yoloe/      app.py (server, jobs, building, testing), train.py (fine-tuning), yoloe_bake.py,
                          dataset.py, evaluate.py, static/ (the page)
dockers/batch-sam3/       app.py, sam3_models.py (SAM 3 and the prompt kinds), prompt_tune.py,
                          dataset.py, evaluate.py, static/
tools/                    setup_ubuntu_nvidia.sh, setup_ubuntu_amd.sh
```

`dataset.py`, `evaluate.py` and `detect_utils.py` are the same in both containers. They are separate
copies, so change both.

Built models are kept in `dockers/<container>/modelData/models/` and are not in git (`modelData/` is
ignored).
