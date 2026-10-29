# syntax=docker/dockerfile:1
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY main_web.py service_clients.py detect_utils.py yoloret_predict.py ./

EXPOSE 7860
CMD ["python", "main_web.py"]
