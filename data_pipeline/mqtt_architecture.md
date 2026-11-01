# MQTT architecture

Each truck runs a **local Mosquitto broker**. Sensors, the preprocessing/inference service and the local alert log all talk
to it, so detection and alarms keep working with no cellular signal. Only compact results are sent to the operations centre
when coverage returns (see `scenario_architecture/system_architecture.png`).

```
sensors / simulator ──► truck/{id}/temp, /vib, /door ──► [ local Mosquitto broker ]
                                                               │
                         inference container (inference/)  ◄──┘
                         moving average → 30 s windows (10 s step) → 6 features → normalise → model
                               │                │
                               ▼                ▼
        logibridge/trucks/{id}/inference   logibridge/trucks/{id}/drift      (PSI drift monitor, QoS 0)
                               │
                               ▼
          local alert log + in-cab alarm ──► sync agent ──► cellular uplink (TLS 1.3, QoS 1) ──► operations centre
```

## Topics

`{id}` is the truck id (default `T001`).

| Topic | Publisher | Rate | QoS | Payload | Status |
|---|---|---|---|---|---|
| `truck/{id}/temp` | simulator / sensor service | 1 Hz | 0 | `{"truck_id","ts","seq","value","unit":"C"}` | implemented |
| `truck/{id}/vib` | simulator / sensor service | 0.5 Hz | 0 | `{"truck_id","ts","seq","value","unit":"g_rms"}` (RMS of the vibration signal) | implemented |
| `truck/{id}/door` | simulator / sensor service | on each event | 1 | `{"truck_id","ts","seq","event":"OPEN"\|"CLOSE"}` | implemented |
| `truck/{id}/status` | simulator | on connect/disconnect | 1, retained | `online` / `offline` (last will) | implemented |
| `truck/{id}/features` | `preprocessing.py run` | every 10 s | 0 | raw and normalised 6-value feature vector | implemented |
| `logibridge/trucks/{id}/inference` | inference service | every 10 s | 1 | class, label, confidence, probabilities, features, model file and hash | implemented |
| `logibridge/trucks/{id}/drift` | inference service | every 60 s | 0 | PSI, status, alert flag, window size, score name | implemented |
| `logibridge/trucks/{id}/status` | inference service | on connect/disconnect | 1, retained | `online` / `offline` (last will) | implemented |
| `truck/{id}/gps`, `.../state`, `.../alert` | | | | location tag, state summary, alerts | design only (Task A2) |

Example inference message:

```json
{"truck_id":"T001","window_end":"2027-01-15T08:00:30.000Z","class":2,"label":"Critical","confidence":0.9998,
 "probabilities":[0.0,0.0002,0.9998],"features":{"temp_mean":5.1,"vib_rms":1.29},
 "model":{"file":"model.tflite","sha256":"3f9c1a2b7d40"}}
```

## Design points

* **Offline first.** Everything up to the in-cab alarm and the local log is on the truck; only the sync agent needs the network.
* **QoS.** High-rate sensor topics use QoS 0 on the local link (no loss risk on a local broker, no overhead); door events and results use
  QoS 1; the uplink uses QoS 1 with retries until acknowledged.
* **Security.** The local broker accepts local connections only and is authenticated; the uplink uses TLS 1.3 with mutual authentication.
* **Last will and status.** Each client sets a last-will message so subscribers can tell if a service died.
* **Reconnection.** Clients subscribe inside `on_connect`, so subscriptions return after a reconnect (paho does not restore them itself).
* **Naming.** The sensors use `truck/{id}/...` (Task C1) while the results use the `logibridge/trucks/{id}/...` namespace the assignment
  specifies (Task D2). The inference service reads sensors from `SENSOR_TOPIC_PREFIX` (default `truck/{truck_id}`), so moving the sensors
  into the `logibridge` namespace is a one-variable change.
* **Message size.** The JSON payloads (83 B temperature, 88 B vibration, 74 B door event) stay on the truck, so they do not affect the
  cellular volume calculated in Task A1.

## Try it

```bash
make simulate        # prints simulator messages (no broker needed)
# with a broker:  mosquitto -v ;  mosquitto_sub -h localhost -t 'truck/#' -t 'logibridge/#' -v
python3 simulator.py --anomaly none               # in .work/ after `make stage`
```
