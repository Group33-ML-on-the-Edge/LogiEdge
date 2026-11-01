#!/usr/bin/env python3
"""LogiEdge preprocessing pipeline (Tasks C2 and C3).

Sequence, exactly as specified:

  1. Filtering        5-sample moving average on the temperature and vibration streams
  2. Windowing        30 s sliding windows, step 10 s
  3. Feature extraction (each sensor independently)
         temperature -> mean, standard deviation, rate of change (degC/min)
         vibration   -> RMS, peak, kurtosis
  4. Feature-level fusion (Task C3)
         the two 3-value blocks are concatenated into ONE 6-value joint vector
  5. Normalisation    z = (x - mean) / std, using statistics loaded from training_stats.npy

The normalisation statistics are computed ONCE, offline, from 10 minutes of clean
Normal-class output (`make-stats`), saved to training_stats.npy and only ever LOADED at
runtime. Nothing in the runtime path can recompute or update them.

Feature vector order (index: name):
  0 temp_mean   1 temp_std   2 temp_rate_c_per_min   3 vib_rms   4 vib_peak   5 vib_kurtosis

Command line:
  python3 preprocessing.py make-stats --out training_stats.npy
  python3 preprocessing.py run --stats training_stats.npy --host localhost --truck-id T001

Requires numpy; `run` also needs paho-mqtt.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

# --------------------------------------------------------------------------
# Specification constants
# --------------------------------------------------------------------------
MA_SAMPLES = 5            # moving-average length
WINDOW_S = 30.0           # window length
STEP_S = 10.0             # window step
TEMP_HZ = 1.0             # temperature stream rate (Task C1)
VIB_HZ = 0.5              # vibration RMS stream rate (Task C1)
MIN_FILL = 0.8            # a window needs >= 80% of its nominal samples
STATS_MINUTES = 10        # length of clean Normal-class data used for the statistics

FEATURE_NAMES = ("temp_mean", "temp_std", "temp_rate_c_per_min",
                 "vib_rms", "vib_peak", "vib_kurtosis")
N_FEATURES = len(FEATURE_NAMES)

log = logging.getLogger("preprocessing")


# --------------------------------------------------------------------------
# 1. Filtering
# --------------------------------------------------------------------------
class MovingAverage:
    """Causal 5-sample moving average.

    The first four outputs average the samples seen so far (warm-up), so no
    sample is lost at start-up.
    """

    def __init__(self, n: int = MA_SAMPLES):
        self.n = n
        self._buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return sum(self._buf) / len(self._buf)


# --------------------------------------------------------------------------
# 3. Feature extraction: each sensor independently
# --------------------------------------------------------------------------
def temperature_features(ts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """[mean, std, rate of change in degC/min] of one temperature window.

    std is the population standard deviation. The rate is the least-squares slope
    over the window (more robust to noise than last minus first).
    """
    ts = np.asarray(ts, dtype=float)
    x = np.asarray(x, dtype=float)
    t = ts - ts.mean()
    denom = float(np.sum(t * t))
    slope_per_s = float(np.sum(t * (x - x.mean())) / denom) if denom > 0 else 0.0
    return np.array([x.mean(), x.std(), slope_per_s * 60.0])


def vibration_features(x: np.ndarray) -> np.ndarray:
    """[RMS, peak, kurtosis] of one vibration window.

    Peak is max |x|. Kurtosis is the excess (Fisher) kurtosis, m4 / m2^2 - 3, so a
    Gaussian signal gives about 0; a constant window gives 0 instead of dividing by zero.
    """
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    d = x - x.mean()
    m2 = float(np.mean(d ** 2))
    kurt = float(np.mean(d ** 4) / (m2 * m2) - 3.0) if m2 > 1e-12 else 0.0
    return np.array([rms, peak, kurt])


# --------------------------------------------------------------------------
# 4. Feature-level fusion (Task C3)
# --------------------------------------------------------------------------
def fuse_features(temp_block: np.ndarray, vib_block: np.ndarray) -> np.ndarray:
    """Concatenate the per-sensor feature blocks into one joint 6-value vector."""
    temp_block, vib_block = np.asarray(temp_block), np.asarray(vib_block)
    if temp_block.shape != (3,) or vib_block.shape != (3,):
        raise ValueError("each sensor must contribute exactly 3 features")
    return np.concatenate([temp_block, vib_block])


# --------------------------------------------------------------------------
# 5. Normalisation with saved statistics (load only)
# --------------------------------------------------------------------------
class Normaliser:
    """z-score normalisation with fixed, pre-computed statistics.

    The statistics come from training_stats.npy (a (2, 6) array: row 0 means,
    row 1 standard deviations). They are read-only. This class has no fit or
    update method on purpose: live data must never change them.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        mean = np.array(mean, dtype=float)
        std = np.array(std, dtype=float)
        if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
            raise ValueError(f"mean and std must each have {N_FEATURES} values")
        if np.any(std <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("std must be positive and all statistics finite")
        mean.flags.writeable = False
        std.flags.writeable = False
        self.mean, self.std = mean, std

    @classmethod
    def load(cls, path: str) -> "Normaliser":
        arr = np.load(path)
        if arr.shape != (2, N_FEATURES):
            raise ValueError(f"{path}: expected shape (2, {N_FEATURES}), got {arr.shape}")
        return cls(arr[0], arr[1])

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.mean) / self.std

    def shifted(self, sigmas: float, features: Optional[list[int]] = None) -> "Normaliser":
        """Return a copy whose mean is moved by `sigmas` standard deviations.

        Only for the stats-shift experiment; the saved file is never touched.
        """
        idx = list(range(N_FEATURES)) if features is None else features
        mean = np.array(self.mean)
        mean[idx] = mean[idx] + sigmas * self.std[idx]
        return Normaliser(mean, self.std)


# --------------------------------------------------------------------------
# The streaming pipeline
# --------------------------------------------------------------------------
@dataclass
class FeatureWindow:
    start: float
    end: float
    raw: np.ndarray                              # 6-value joint feature vector
    z: Optional[np.ndarray] = None               # normalised vector (if a Normaliser is set)
    temp_samples: Optional[np.ndarray] = field(default=None, repr=False)   # filtered window samples
    vib_samples: Optional[np.ndarray] = field(default=None, repr=False)


class PreprocessingPipeline:
    """Streaming filter -> window -> features -> fuse -> normalise.

    Feed messages in time order with push(stream, ts, value); a list of completed
    FeatureWindow objects is returned. A window [s, s+30) is emitted as soon as a
    sample with timestamp >= s+30 arrives, then the next window starts 10 s later.
    """

    def __init__(self, normaliser: Optional[Normaliser] = None, window_s: float = WINDOW_S,
                 step_s: float = STEP_S, ma_samples: int = MA_SAMPLES, keep_samples: bool = False):
        self.normaliser = normaliser
        self.window_s, self.step_s = window_s, step_s
        self.keep_samples = keep_samples
        self._ma = {"temp": MovingAverage(ma_samples), "vib": MovingAverage(ma_samples)}
        self._buf: dict[str, list[tuple[float, float]]] = {"temp": [], "vib": []}
        self._origin: Optional[float] = None
        self._next_start: Optional[float] = None
        self._latest = -np.inf
        self._min = {"temp": int(MIN_FILL * window_s * TEMP_HZ), "vib": int(MIN_FILL * window_s * VIB_HZ)}
        self.dropped_out_of_order = 0
        self.skipped_windows = 0

    def push(self, stream: str, ts: float, value: float) -> list[FeatureWindow]:
        if stream not in self._buf:
            raise ValueError(f"unknown stream {stream!r}")
        if self._origin is None:
            self._origin = self._next_start = ts
        buf = self._buf[stream]
        if buf and ts <= buf[-1][0]:
            self.dropped_out_of_order += 1          # duplicate or late message
            return []
        buf.append((ts, self._ma[stream].update(value)))
        self._latest = max(self._latest, ts)

        out = []
        while self._latest >= self._next_start + self.window_s:
            win = self._emit(self._next_start, self._next_start + self.window_s)
            if win is not None:
                out.append(win)
            self._next_start += self.step_s
            cutoff = self._next_start                       # older samples can never be used again
            for b in self._buf.values():
                while b and b[0][0] < cutoff:
                    b.pop(0)
        return out

    def _emit(self, start: float, end: float) -> Optional[FeatureWindow]:
        t = [(ts, v) for ts, v in self._buf["temp"] if start <= ts < end]
        v = [(ts, x) for ts, x in self._buf["vib"] if start <= ts < end]
        if len(t) < self._min["temp"] or len(v) < self._min["vib"]:
            self.skipped_windows += 1
            log.warning("Window %.0f-%.0f skipped: %d temp / %d vib samples", start, end, len(t), len(v))
            return None
        t_ts, t_x = np.array([a for a, _ in t]), np.array([b for _, b in t])
        v_x = np.array([b for _, b in v])
        raw = fuse_features(temperature_features(t_ts, t_x), vibration_features(v_x))
        z = self.normaliser.transform(raw) if self.normaliser else None
        return FeatureWindow(start, end, raw, z,
                             t_x if self.keep_samples else None, v_x if self.keep_samples else None)


# --------------------------------------------------------------------------
# Training statistics: offline step, never called at runtime
# --------------------------------------------------------------------------
def compute_training_stats(clean_normal_windows: np.ndarray) -> np.ndarray:
    """(2, 6) array of per-feature mean and std from clean Normal-class windows."""
    w = np.asarray(clean_normal_windows, dtype=float)
    if w.ndim != 2 or w.shape[1] != N_FEATURES or len(w) < 10:
        raise ValueError("need an (N >= 10, 6) array of Normal-class feature vectors")
    return np.vstack([w.mean(axis=0), w.std(axis=0, ddof=1)])


def make_training_stats(out_path: str, seed: int = 2026, minutes: int = STATS_MINUTES) -> np.ndarray:
    """Generate `minutes` of clean Normal-class data with the Task C1 sensor models,
    run it through the pipeline (without normalisation) and save training_stats.npy."""
    import simulator as sim                                   # reuse the C1 sensor models
    rng_t, rng_v, _ = sim._make_rngs(seed)
    temp, vib = sim.TemperatureSensor(rng_t, False), sim.VibrationSensor(rng_v, False)
    pipe = PreprocessingPipeline(normaliser=None)
    feats = []
    for sec in range(int(minutes * 60)):
        feats += [w.raw for w in pipe.push("temp", float(sec), temp.read(False))]
        if sec % 2 == 0:
            feats += [w.raw for w in pipe.push("vib", float(sec), vib.read(False))]
    stats = compute_training_stats(np.array(feats))
    np.save(out_path, stats)
    log.info("Saved %s from %d Normal-class windows (%d min)", out_path, len(feats), minutes)
    return stats


# --------------------------------------------------------------------------
# MQTT runtime: subscribe -> preprocess -> publish features
# --------------------------------------------------------------------------
class MqttPreprocessor:
    """Subscribes to truck/<id>/temp and truck/<id>/vib, runs the pipeline and
    publishes each normalised feature vector on truck/<id>/features."""

    def __init__(self, pipeline: PreprocessingPipeline, host: str, port: int, truck_id: str,
                 on_window: Optional[Callable[[FeatureWindow], None]] = None):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:                                  # pragma: no cover
            raise SystemExit("paho-mqtt is required: pip install paho-mqtt") from exc
        self.pipeline, self.host, self.port, self.truck_id = pipeline, host, port, truck_id
        self.on_window = on_window
        self.topics = {f"truck/{truck_id}/temp": "temp", f"truck/{truck_id}/vib": "vib"}
        self.out_topic = f"truck/{truck_id}/features"
        client_id = f"logiedge-pre-{truck_id}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if int(getattr(rc, "value", rc)) == 0:
            for topic in self.topics:
                client.subscribe(topic, qos=0)
            log.info("Subscribed to %s", ", ".join(self.topics))

    def _on_message(self, client, userdata, msg):
        try:
            body = json.loads(msg.payload)
            ts = datetime.fromisoformat(body["ts"].replace("Z", "+00:00")).timestamp()
            windows = self.pipeline.push(self.topics[msg.topic], ts, float(body["value"]))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning("Ignoring malformed message on %s: %s", msg.topic, exc)
            return
        for w in windows:
            payload = json.dumps({"truck_id": self.truck_id, "window_end": w.end,
                                  "raw": np.round(w.raw, 4).tolist(),
                                  "z": None if w.z is None else np.round(w.z, 3).tolist()},
                                 separators=(",", ":"))
            client.publish(self.out_topic, payload, qos=0)
            if self.on_window:
                self.on_window(w)

    def run(self, stop_event: threading.Event) -> None:
        self.client.connect_async(self.host, self.port, 30)
        self.client.loop_start()
        try:
            stop_event.wait()
        finally:
            self.client.disconnect()
            self.client.loop_stop()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LogiEdge preprocessing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stats", help="create training_stats.npy from 10 min of clean Normal data (offline)")
    m.add_argument("--out", default="training_stats.npy")
    m.add_argument("--seed", type=int, default=2026)
    r = sub.add_parser("run", help="subscribe to the broker and publish normalised feature vectors")
    r.add_argument("--stats", default="training_stats.npy", help="saved statistics, loaded and never recomputed")
    r.add_argument("--host", default="localhost")
    r.add_argument("--port", type=int, default=1883)
    r.add_argument("--truck-id", default="T001")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if args.cmd == "make-stats":
        stats = make_training_stats(args.out, args.seed)
        for name, mu, sd in zip(FEATURE_NAMES, stats[0], stats[1]):
            print(f"{name:22s} mean={mu:9.4f}  std={sd:8.4f}")
        return 0

    try:
        normaliser = Normaliser.load(args.stats)
    except (OSError, ValueError) as exc:
        log.error("Cannot load statistics: %s. Run 'make-stats' first.", exc)
        return 1
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    MqttPreprocessor(PreprocessingPipeline(normaliser), args.host, args.port, args.truck_id,
                     on_window=lambda w: print(f"[{w.end:.0f}] z = {np.round(w.z, 2).tolist()}", flush=True)).run(stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
