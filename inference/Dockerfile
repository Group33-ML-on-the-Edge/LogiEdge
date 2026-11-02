# LogiEdge inference service (Task D2)
#
# Layer order matters for OTA updates: things that change rarely come first, the model comes LAST.
# Changing only model.tflite therefore rebuilds (and re-sends) only the model layer.

FROM python:3.11-slim

# Settings that rarely change. MODEL_PATH can be overridden at run time (docker run -e MODEL_PATH=...)
# to switch model variants without rebuilding the image.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/model.tflite \
    STATS_PATH=/app/training_stats.npy \
    TRUCK_ID=T001 \
    MQTT_HOST=localhost \
    MQTT_PORT=1883

RUN useradd --system --uid 10001 --no-create-home logiedge
WORKDIR /app

# 1. Python dependencies: every pip install layer comes before the model
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Application code (including the PSI drift monitor) and the saved normalisation statistics
COPY preprocessing.py drift_monitor.py inference_service.py training_stats.npy ./

# 3. The model: the last layer that carries data
COPY model.tflite .

USER logiedge
CMD ["python", "inference_service.py"]
