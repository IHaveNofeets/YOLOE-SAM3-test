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

COPY sam3_predict.py service.py ./

EXPOSE 8700
CMD ["python", "service.py"]
