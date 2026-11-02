#!/usr/bin/env python3
"""LogiEdge inference service (Task D2): preprocessing -> model inference -> MQTT publish.

Runs inside the Docker container. It subscribes to the truck's sensor topics, passes the
messages through the Task C2 pipeline (5-sample moving average, 30 s windows every 10 s,
6 features, normalisation with the saved training_stats.npy), runs the model on each
feature vector and publishes the result to

    logibridge/trucks/{truck_id}/inference

Configuration (environment variables):

    MODEL_PATH           model file, .tflite or .npz            default /app/model.tflite
                         Change it to switch model variants WITHOUT rebuilding the image
                         (mount a folder of models and point MODEL_PATH at one of them).
    STATS_PATH           normalisation statistics               default /app/training_stats.npy
    TRUCK_ID             truck identifier                       default T001
    MQTT_HOST, MQTT_PORT local Mosquitto broker                 default localhost, 1883
    MQTT_USERNAME, MQTT_PASSWORD   optional broker credentials
    SENSOR_TOPIC_PREFIX  where the sensors publish              default truck/{truck_id}
                         (the service reads <prefix>/temp and <prefix>/vib, as the Task C1 simulator does)
    REFERENCE_DIST_PATH  reference_dist.json for PSI drift monitoring (Task E1); monitoring is off if unset
    DRIFT_SCORE          score monitored: normal_confidence (default) or max_confidence
    LOG_LEVEL            logging level                          default INFO

Model files:
    .tflite   run with tflite-runtime (or ai-edge-litert / TensorFlow Lite if that is what is installed).
              The model takes the 6 normalised features and returns 3 class scores (logits or
              probabilities); float32 and int8-quantised inputs and outputs are both handled.
    .npz      the plain numpy weights saved by train_model.py (useful before conversion).

Drift monitoring (when REFERENCE_DIST_PATH is set): PSI over the last 100 inferences every 60 s is printed to
standard output (docker logs), a "[LOGIBRIDGE DRIFT ALERT] PSI=..." line is printed above 0.25, and each evaluation
is published on logibridge/trucks/{truck_id}/drift.

Result message (JSON, QoS 1):
    {"truck_id","window_end","class","label","confidence","probabilities","features","model":{"file","sha256"}}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

import numpy as np

from drift_monitor import DEFAULT_SCORE, PsiMonitor, load_reference
from preprocessing import FEATURE_NAMES, Normaliser, PreprocessingPipeline

CLASS_NAMES = ("Normal", "Warning", "Critical")
log = logging.getLogger("inference_service")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    model_path: str = "/app/model.tflite"
    stats_path: str = "/app/training_stats.npy"
    truck_id: str = "T001"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    sensor_topic_prefix: str = "truck/{truck_id}"
    reference_dist_path: Optional[str] = None
    drift_score: str = DEFAULT_SCORE
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Config":
        d = cls()
        return cls(
            model_path=env.get("MODEL_PATH", d.model_path),
            stats_path=env.get("STATS_PATH", d.stats_path),
            truck_id=env.get("TRUCK_ID", d.truck_id),
            mqtt_host=env.get("MQTT_HOST", d.mqtt_host),
            mqtt_port=int(env.get("MQTT_PORT", d.mqtt_port)),
            mqtt_username=env.get("MQTT_USERNAME") or None,
            mqtt_password=env.get("MQTT_PASSWORD") or None,
            sensor_topic_prefix=env.get("SENSOR_TOPIC_PREFIX", d.sensor_topic_prefix),
            reference_dist_path=env.get("REFERENCE_DIST_PATH") or None,
            drift_score=env.get("DRIFT_SCORE", d.drift_score),
            log_level=env.get("LOG_LEVEL", d.log_level).upper(),
        )

    @property
    def inference_topic(self) -> str:
        return f"logibridge/trucks/{self.truck_id}/inference"

    @property
    def status_topic(self) -> str:
        return f"logibridge/trucks/{self.truck_id}/status"

    @property
    def drift_topic(self) -> str:
        return f"logibridge/trucks/{self.truck_id}/drift"

    @property
    def sensor_topics(self) -> dict:
        prefix = self.sensor_topic_prefix.format(truck_id=self.truck_id)
        return {f"{prefix}/temp": "temp", f"{prefix}/vib": "vib"}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Accept either probabilities (already sum to 1) or logits."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if np.all(scores >= 0) and abs(scores.sum() - 1.0) < 1e-3:
        return scores
    return softmax(scores)


class NumpyModel:
    """MLP from the .npz weights written by train_model.py (ReLU hidden layers)."""

    def __init__(self, path: str):
        data = np.load(path)
        n = len(data["layers"]) - 1
        self.W = [data[f"W{i + 1}"] for i in range(n)]
        self.b = [data[f"b{i + 1}"] for i in range(n)]

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        a = np.asarray(z, dtype=float)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            a = a @ w + b
            if i < len(self.W) - 1:
                a = np.maximum(a, 0.0)
        return to_probabilities(a)


def _make_interpreter(path: str):
    """Create a TFLite interpreter from whichever runtime is installed."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError as exc:
                raise RuntimeError("No TFLite runtime found. Install tflite-runtime (see requirements.txt).") from exc
    return Interpreter(model_path=path)


class TFLiteModel:
    def __init__(self, path: str):
        self.interp = _make_interpreter(path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        x = np.asarray(z, dtype=np.float32).reshape(1, -1)
        dtype = self.inp["dtype"]
        if dtype != np.float32:                                   # quantised input
            scale, zero = self.inp["quantization"]
            info = np.iinfo(dtype)
            x = np.clip(np.round(x / scale + zero), info.min, info.max).astype(dtype)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = np.asarray(self.interp.get_tensor(self.out["index"]))[0]
        if self.out["dtype"] != np.float32:                       # quantised output
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return to_probabilities(y)


@dataclass
class LoadedModel:
    backend: object
    file: str
    sha256: str

    def predict_proba(self, z):
        return self.backend.predict_proba(z)


def load_model(path: str) -> LoadedModel:
    """Load the model at `path` (chosen by the MODEL_PATH variable)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tflite":
        backend = TFLiteModel(path)
    elif ext == ".npz":
        backend = NumpyModel(path)
    else:
        raise ValueError(f"Unsupported model type {ext!r}; use .tflite or .npz")
    return LoadedModel(backend, os.path.basename(path), digest)


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------
class InferenceService:
    def __init__(self, cfg: Config, model: LoadedModel, normaliser: Normaliser):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:                                  # pragma: no cover
            raise SystemExit("paho-mqtt is required (see requirements.txt)") from exc
        self.cfg, self.model = cfg, model
        self.pipeline = PreprocessingPipeline(normaliser)
        self.sensor_topics = cfg.sensor_topics
        self.published = 0
        self.errors = 0
        self.last_probs = None
        self.drift = None
        if cfg.reference_dist_path:
            try:
                self.drift = PsiMonitor(load_reference(cfg.reference_dist_path), cfg.drift_score)
                log.info("PSI drift monitoring on (%s, reference %s)", cfg.drift_score, cfg.reference_dist_path)
            except (OSError, ValueError, KeyError) as exc:
                log.error("Drift monitoring disabled: cannot use %s: %s", cfg.reference_dist_path, exc)
        client_id = f"logiedge-inference-{cfg.truck_id}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)
        if cfg.mqtt_username:
            self.client.username_pw_set(cfg.mqtt_username, cfg.mqtt_password)
        self.client.will_set(cfg.status_topic, "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # paho 1.x passes (client, userdata, flags, rc); 2.x adds properties. Accept both.
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if int(getattr(rc, "value", rc)) != 0:
            log.error("Broker refused connection: %s", rc)
            return
        for topic in self.sensor_topics:
            client.subscribe(topic, qos=0)
        client.publish(self.cfg.status_topic, "online", qos=1, retain=True)
        log.info("Connected to %s:%s, subscribed to %s", self.cfg.mqtt_host, self.cfg.mqtt_port,
                 ", ".join(self.sensor_topics))

    def _on_message(self, client, userdata, msg):
        try:
            body = json.loads(msg.payload)
            ts = datetime.fromisoformat(body["ts"].replace("Z", "+00:00")).timestamp()
            windows = self.pipeline.push(self.sensor_topics[msg.topic], ts, float(body["value"]))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.errors += 1
            log.warning("Ignoring malformed message on %s: %s", msg.topic, exc)
            return
        for window in windows:
            self.publish_result(client, window)

    def infer(self, window) -> dict:
        probs = self.model.predict_proba(window.z)
        self.last_probs = probs
        cls = int(np.argmax(probs))
        return {
            "truck_id": self.cfg.truck_id,
            "window_end": datetime.fromtimestamp(window.end, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "class": cls,
            "label": CLASS_NAMES[cls],
            "confidence": round(float(probs[cls]), 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "features": dict(zip(FEATURE_NAMES, (round(float(v), 4) for v in window.raw))),
            "model": {"file": self.model.file, "sha256": self.model.sha256},
        }

    def publish_result(self, client, window) -> None:
        try:
            result = self.infer(window)
        except Exception:                                           # keep the service alive on a bad window
            self.errors += 1
            log.exception("Inference failed for window ending %s", window.end)
            return
        client.publish(self.cfg.inference_topic, json.dumps(result, separators=(",", ":")), qos=1, retain=False)
        self.published += 1
        log.info("%s -> %s (%.1f%%) model=%s", result["window_end"], result["label"],
                 100 * result["confidence"], self.model.sha256)
        if self.drift is not None:
            for event in self.drift.add(window.end, self.last_probs):
                client.publish(self.cfg.drift_topic, json.dumps(
                    {"truck_id": self.cfg.truck_id, "psi": None if event["psi"] is None else round(event["psi"], 4),
                     "status": event["status"], "alert": event["alert"], "n": event["n"], "score": event["score"]},
                    separators=(",", ":")), qos=0, retain=False)

    def run(self, stop_event: threading.Event) -> None:
        self.client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, 30)
        self.client.loop_start()
        try:
            stop_event.wait()
        finally:
            try:
                info = self.client.publish(self.cfg.status_topic, "offline", qos=1, retain=True)
                info.wait_for_publish(timeout=2)
            except Exception:                                       # pragma: no cover
                pass
            self.client.disconnect()
            self.client.loop_stop()


def main(env: Mapping[str, str] = os.environ) -> int:
    cfg = Config.from_env(env)
    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    try:
        normaliser = Normaliser.load(cfg.stats_path)
    except (OSError, ValueError) as exc:
        log.error("Cannot load normalisation statistics from %s: %s", cfg.stats_path, exc)
        return 2
    try:
        model = load_model(cfg.model_path)
    except FileNotFoundError:
        log.error("MODEL_PATH=%s does not exist. Mount your model folder (for example -v ./models:/models:ro) "
                  "and point MODEL_PATH at a file in it.", cfg.model_path)
        return 2
    except (ValueError, RuntimeError) as exc:
        log.error("Cannot load model %s: %s", cfg.model_path, exc)
        return 2
    log.info("Loaded model %s (sha256 %s) for truck %s; publishing to %s",
             model.file, model.sha256, cfg.truck_id, cfg.inference_topic)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):                     # docker stop sends SIGTERM
        signal.signal(sig, lambda *_: stop.set())
    service = InferenceService(cfg, model, normaliser)
    service.run(stop)
    log.info("Stopped. Published %d results, %d errors", service.published, service.errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
