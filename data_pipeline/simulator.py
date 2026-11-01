#!/usr/bin/env python3
"""LogiEdge cold-chain truck sensor simulator (Task C1).

Publishes three simulated sensor streams to a local Mosquitto broker:

    stream          topic                  rate            QoS  normal                    anomaly
    temperature     truck/<id>/temp        1 Hz            0    N(4.0 C, 0.3)             +0.08 C per reading (linear drift)
    vibration_rms   truck/<id>/vib         0.5 Hz          0    N(0.45 g, 0.05)           step to N(1.2 g, 0.15) (bearing wear)
    door_event      truck/<id>/door        discrete        1    'OPEN' / 'CLOSE' + time   continues in every mode

Anomaly modes (--anomaly); the anomaly can start late (--anomaly-start) and end early (--anomaly-end):
    none        all streams normal
    temp_drift  temperature drifts upward
    vibration   vibration steps up
    combined    both anomalies at once

A retained status message ("online" / "offline", with a last will) is published
on truck/<id>/status so subscribers can tell whether the simulator is alive.

Examples:
    python3 simulator.py --anomaly none
    python3 simulator.py --anomaly combined --anomaly-start 120 --duration 600
    python3 simulator.py --anomaly temp_drift --dry-run --duration 30 --speed 30

Requires:  pip install paho-mqtt   (not needed with --dry-run)
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Stream parameters (from the Task C1 table)
# --------------------------------------------------------------------------
SETPOINT_C = 4.0
TEMP_SIGMA_C = 0.3
TEMP_DRIFT_PER_READING_C = 0.08
VIB_NORMAL = (0.45, 0.05)          # mean g, sigma g
VIB_FAULT = (1.2, 0.15)            # bearing wear
TEMP_PERIOD_S = 1.0                # 1 Hz
VIB_PERIOD_S = 2.0                 # 0.5 Hz
DOOR_OPEN_MIN_S = 20.0             # door stays open between these bounds
DOOR_OPEN_MAX_S = 120.0

ANOMALY_CHOICES = ("none", "temp_drift", "vibration", "combined")
QOS = {"temp": 0, "vib": 0, "door": 1}   # matches the Task A2 topic design

log = logging.getLogger("simulator")


# --------------------------------------------------------------------------
# Sensor models (pure logic, no I/O)
# --------------------------------------------------------------------------
class TemperatureSensor:
    """Cargo temperature: N(4.0, 0.3) plus optional linear drift."""

    def __init__(self, rng: random.Random, drift_enabled: bool, cap_c: Optional[float] = None):
        self.rng = rng
        self.drift_enabled = drift_enabled
        self.cap_c = cap_c
        self.anomalous_readings = 0

    def read(self, anomaly_active: bool) -> float:
        value = self.rng.gauss(SETPOINT_C, TEMP_SIGMA_C)
        if self.drift_enabled and anomaly_active:
            self.anomalous_readings += 1          # first anomalous reading gets +0.08
            value += TEMP_DRIFT_PER_READING_C * self.anomalous_readings
        elif not anomaly_active:
            self.anomalous_readings = 0           # clean data restored: the drift starts again from zero next time
        if self.cap_c is not None:
            value = min(value, self.cap_c)
        return round(value, 2)


class VibrationSensor:
    """Compressor vibration RMS in g: N(0.45, 0.05), steps to N(1.2, 0.15) on a bearing fault."""

    def __init__(self, rng: random.Random, fault_enabled: bool):
        self.rng = rng
        self.fault_enabled = fault_enabled

    def read(self, anomaly_active: bool) -> float:
        mean, sigma = VIB_FAULT if (self.fault_enabled and anomaly_active) else VIB_NORMAL
        return round(max(0.0, self.rng.gauss(mean, sigma)), 3)   # RMS cannot be negative


class DoorSimulator:
    """Alternating OPEN/CLOSE events. Door starts closed; OPEN gaps are exponential."""

    def __init__(self, rng: random.Random, mean_gap_s: float):
        self.rng = rng
        self.mean_gap_s = mean_gap_s
        self.is_open = False
        self.next_time = self._gap()

    def _gap(self) -> float:
        return max(1.0, self.rng.expovariate(1.0 / self.mean_gap_s))

    def pop_event(self) -> tuple[float, str]:
        """Return (time, 'OPEN'|'CLOSE') for the pending event and schedule the next one."""
        when = self.next_time
        if self.is_open:
            self.is_open = False
            self.next_time = when + self._gap()
            return when, "CLOSE"
        self.is_open = True
        self.next_time = when + self.rng.uniform(DOOR_OPEN_MIN_S, DOOR_OPEN_MAX_S)
        return when, "OPEN"


# --------------------------------------------------------------------------
# Configuration and publishers
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    anomaly: str = "none"
    truck_id: str = "T001"
    duration_s: float = 0.0            # 0 = run until stopped
    speed: float = 1.0                 # simulated seconds per real second
    seed: Optional[int] = None
    anomaly_start_s: float = 0.0
    anomaly_end_s: Optional[float] = None   # None: the anomaly lasts until the run stops
    door_interval_s: float = 300.0     # mean gap between door openings
    temp_cap_c: Optional[float] = None

    def topic(self, kind: str) -> str:
        return f"truck/{self.truck_id}/{kind}"


class DryRunPublisher:
    """Prints messages instead of publishing them (no broker or paho needed)."""

    def __init__(self, out=None):
        self.out = out or sys.stdout

    def connect(self) -> None:
        pass

    def publish(self, topic: str, payload: str, qos: int) -> bool:
        print(f"{topic} qos={qos} {payload}", file=self.out, flush=True)
        return True

    def close(self) -> None:
        pass


class MqttPublisher:
    """Publishes to a Mosquitto broker with paho-mqtt (works with paho 1.x and 2.x)."""

    def __init__(self, host: str, port: int, truck_id: str, username: Optional[str] = None,
                 password: Optional[str] = None, keepalive: int = 30, connect_timeout: float = 10.0):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:                                  # pragma: no cover
            raise SystemExit("paho-mqtt is required: pip install paho-mqtt "
                             "(or use --dry-run)") from exc
        self.mqtt = mqtt
        self.host, self.port = host, port
        self.keepalive = keepalive
        self.connect_timeout = connect_timeout
        self.status_topic = f"truck/{truck_id}/status"
        self.connected = threading.Event()

        client_id = f"logiedge-sim-{truck_id}"
        if hasattr(mqtt, "CallbackAPIVersion"):                    # paho >= 2.0
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:                                                      # paho 1.x
            self.client = mqtt.Client(client_id=client_id)
        if username:
            self.client.username_pw_set(username, password)
        self.client.will_set(self.status_topic, "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    # paho 1.x passes (client, userdata, flags, rc); 2.x adds properties. Accept both.
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if int(getattr(rc, "value", rc)) == 0:
            log.info("Connected to broker %s:%s", self.host, self.port)
            client.publish(self.status_topic, "online", qos=1, retain=True)
            self.connected.set()
        else:
            log.error("Broker refused connection: %s", rc)

    def _on_disconnect(self, client, userdata, *args):
        self.connected.clear()
        log.warning("Disconnected from broker; paho will keep retrying")

    def connect(self) -> None:
        self.client.connect_async(self.host, self.port, self.keepalive)
        self.client.loop_start()
        if not self.connected.wait(self.connect_timeout):
            self.client.loop_stop()
            raise ConnectionError(f"Could not connect to MQTT broker at {self.host}:{self.port} "
                                  f"within {self.connect_timeout:.0f} s. Is Mosquitto running?")

    def publish(self, topic: str, payload: str, qos: int) -> bool:
        info = self.client.publish(topic, payload, qos=qos, retain=False)
        return info.rc == self.mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        try:
            info = self.client.publish(self.status_topic, "offline", qos=1, retain=True)
            info.wait_for_publish(timeout=2)
        except Exception:                                          # pragma: no cover
            pass
        self.client.disconnect()
        self.client.loop_stop()


# --------------------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------------------
def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _make_rngs(seed: Optional[int]) -> tuple[random.Random, random.Random, random.Random]:
    """Independent generators per stream so each stream is reproducible on its own."""
    if seed is None:
        return random.Random(), random.Random(), random.Random()
    return random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)


def run(cfg: SimConfig, publisher, stop_event: Optional[threading.Event] = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Optional[Callable[[float], object]] = None,
        start_epoch: Optional[float] = None) -> dict:
    """Generate data and publish it until the duration elapses or stop_event is set.

    Timestamps come from the scheduled simulated time, so they are evenly spaced
    regardless of scheduling jitter. Returns message counts per stream.
    """
    stop_event = stop_event or threading.Event()
    wait = wait or stop_event.wait
    start_epoch = time.time() if start_epoch is None else start_epoch

    rng_t, rng_v, rng_d = _make_rngs(cfg.seed)
    temp = TemperatureSensor(rng_t, cfg.anomaly in ("temp_drift", "combined"), cfg.temp_cap_c)
    vib = VibrationSensor(rng_v, cfg.anomaly in ("vibration", "combined"))
    door = DoorSimulator(rng_d, cfg.door_interval_s)

    counts = {"temp": 0, "vib": 0, "door": 0, "dropped": 0}
    seq = {"temp": 0, "vib": 0, "door": 0}
    events: list[tuple[float, int, str]] = [(0.0, 0, "temp"), (0.0, 1, "vib"), (door.next_time, 2, "door")]
    heapq.heapify(events)
    order = 3
    t0 = clock()

    def emit(kind: str, due: float) -> str:
        anomaly_active = (cfg.anomaly != "none" and due >= cfg.anomaly_start_s
                          and (cfg.anomaly_end_s is None or due < cfg.anomaly_end_s))
        seq[kind] += 1
        body = {"truck_id": cfg.truck_id, "ts": _iso(start_epoch + due), "seq": seq[kind]}
        if kind == "temp":
            body.update(value=temp.read(anomaly_active), unit="C")
        elif kind == "vib":
            body.update(value=vib.read(anomaly_active), unit="g_rms")
        payload = json.dumps(body, separators=(",", ":"))
        return payload

    while not stop_event.is_set():
        now = (clock() - t0) * cfg.speed
        finished = False
        while events and events[0][0] <= now:
            due, _, kind = heapq.heappop(events)
            if cfg.duration_s and due >= cfg.duration_s:
                finished = True
                break
            if kind == "door":
                when, state = door.pop_event()
                seq["door"] += 1
                payload = json.dumps({"truck_id": cfg.truck_id, "ts": _iso(start_epoch + when),
                                      "seq": seq["door"], "event": state}, separators=(",", ":"))
                heapq.heappush(events, (door.next_time, order, "door"))
            else:
                payload = emit(kind, due)
                heapq.heappush(events, (due + (TEMP_PERIOD_S if kind == "temp" else VIB_PERIOD_S), order, kind))
            order += 1
            if publisher.publish(cfg.topic(kind), payload, QOS[kind]):
                counts[kind] += 1
            else:
                counts["dropped"] += 1
                log.warning("Publish failed (broker unavailable?): %s", cfg.topic(kind))
            log.debug("%s %s", cfg.topic(kind), payload)
        if finished:
            break
        wait(max(0.0, (events[0][0] - now) / cfg.speed))
    return counts


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LogiEdge cold-chain truck sensor simulator. Publishes temperature, "
                    "vibration RMS and door events to a local Mosquitto broker.")
    p.add_argument("--anomaly", choices=ANOMALY_CHOICES, default="none",
                   help="anomaly mode (default: none)")
    p.add_argument("--host", default="localhost", help="MQTT broker host (default: localhost)")
    p.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    p.add_argument("--username", help="broker username, if the broker requires authentication")
    p.add_argument("--password", help="broker password")
    p.add_argument("--truck-id", default="T001", help="truck identifier used in topics (default: T001)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="simulated seconds to run; 0 runs until Ctrl-C (default: 0)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="simulated seconds per real second, for fast test runs (default: 1.0)")
    p.add_argument("--seed", type=int, help="random seed for reproducible data")
    p.add_argument("--anomaly-start", type=float, default=0.0,
                   help="simulated seconds before the anomaly begins (default: 0)")
    p.add_argument("--anomaly-end", type=float, default=None,
                   help="simulated seconds at which the anomaly stops and clean data returns (default: never)")
    p.add_argument("--door-interval", type=float, default=300.0,
                   help="mean seconds between door openings (default: 300)")
    p.add_argument("--temp-cap", type=float,
                   help="optional ceiling for temperature, e.g. ambient 35 (default: none)")
    p.add_argument("--connect-timeout", type=float, default=10.0,
                   help="seconds to wait for the broker (default: 10)")
    p.add_argument("--dry-run", action="store_true", help="print messages instead of publishing")
    p.add_argument("-v", "--verbose", action="store_true", help="log every message")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.speed <= 0 or args.duration < 0 or args.anomaly_start < 0 or args.door_interval <= 0:
        build_parser().error("--speed and --door-interval must be > 0; --duration and --anomaly-start must be >= 0")
    if args.anomaly_end is not None and args.anomaly_end <= args.anomaly_start:
        build_parser().error("--anomaly-end must be greater than --anomaly-start")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    cfg = SimConfig(anomaly=args.anomaly, truck_id=args.truck_id, duration_s=args.duration,
                    speed=args.speed, seed=args.seed, anomaly_start_s=args.anomaly_start,
                    anomaly_end_s=args.anomaly_end,
                    door_interval_s=args.door_interval, temp_cap_c=args.temp_cap)

    publisher = DryRunPublisher() if args.dry_run else MqttPublisher(
        args.host, args.port, args.truck_id, args.username, args.password,
        connect_timeout=args.connect_timeout)
    try:
        publisher.connect()
    except ConnectionError as exc:
        log.error("%s", exc)
        return 1

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    log.info("Simulating truck %s: anomaly=%s, start=%.0fs, speed=x%g, duration=%s",
             cfg.truck_id, cfg.anomaly, cfg.anomaly_start_s, cfg.speed,
             f"{cfg.duration_s:g}s" if cfg.duration_s else "until stopped")
    try:
        counts = run(cfg, publisher, stop)
    finally:
        publisher.close()
    log.info("Stopped. Published: temp=%d vib=%d door=%d (failed=%d)",
             counts["temp"], counts["vib"], counts["door"], counts["dropped"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
