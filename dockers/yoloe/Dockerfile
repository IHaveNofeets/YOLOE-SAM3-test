# syntax=docker/dockerfile:1
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# ultralytics pulls in regular opencv-python as its own transitive dependency, which can win the
# site-packages file-overwrite race against the headless build we actually want (no GUI libs
# installed) - force headless back in as an explicit final step so it's the one left on disk.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

# Base YOLOE weights - fetched once at build time via Ultralytics' own asset downloader (the same
# mechanism that would otherwise trigger on first request at runtime), so no pre-existing weight
# file needs to be provided on the host. Cached as a normal build layer: unchanged on rebuild
# unless this line or an earlier layer changes.
RUN mkdir -p modelData/yoloe \
    && python -c "from ultralytics import YOLOE; YOLOE('modelData/yoloe/yoloe-26l-seg.pt')"

COPY yoloe_bake.py service.py ./

EXPOSE 8600
CMD ["python", "service.py"]
