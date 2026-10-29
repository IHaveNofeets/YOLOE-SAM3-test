# syntax=docker/dockerfile:1
FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# YOLO-ReT's own source (small, part of this project's existing checkout of prakharg24/yoloret) -
# not something to re-download, just copied straight from the build context.
COPY modelData/yoloret/yolo.py ./modelData/yoloret/yolo.py
COPY modelData/yoloret/yolo3/ ./modelData/yoloret/yolo3/
COPY modelData/yoloret/model_data/ ./modelData/yoloret/model_data/

# The actual model weight - fetched once at build time from the upstream repo (committed there
# directly, verified via the GitHub API: code/checkpoints/efficientnetb3_416_coco.h5) instead of
# requiring it to be pre-placed on the host.
RUN mkdir -p modelData/yoloret/checkpoints \
    && curl -fL --retry 3 -o modelData/yoloret/checkpoints/efficientnetb3_416_coco.h5 \
       https://raw.githubusercontent.com/prakharg24/yoloret/main/code/checkpoints/efficientnetb3_416_coco.h5

COPY service.py .

EXPOSE 8500
CMD ["python", "service.py"]
