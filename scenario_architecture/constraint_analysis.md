# Constraint analysis: cold-chain truck monitoring (Task A1)

FreightBridge Logistics runs 85 refrigerated trucks in the pilot (265 at full scale). Each truck carries a temperature
probe (1 Hz), a compressor vibration sensor (500 Hz, 3 axes) and a door switch (discrete events). The system must
classify the cargo state as Normal, Warning or Critical and alert within **90 seconds** of a fault signature appearing in
the sensor data, **without depending on cellular coverage**.

The architecture is in [system_architecture.png](system_architecture.png). The detailed workings, with every step, are in
`reports/individual/LogiEdge_Task_A1_Constraint_Analysis.pdf`.

## 1. Latency

A refrigeration failure raises cargo temperature by 1 °C per minute.

| Quantity | Working | Result |
|---|---|---|
| Temperature rise at the 90 s deadline | 90 s × (1 °C ÷ 60 s) | 1.5 °C (still the Warning band, 1–3 °C) |
| Time to Critical (> 3 °C) | 3 °C ÷ 1 °C/min | 180 s |
| Cloud path, when connected | 10 s window + 4.8 s uplink + 1 s round trip + 0.5 s inference + 1 s alert push | 17.3 s (19% of the budget) |
| Edge path | 10 s window + 0.05 s inference + 0.1 s local alarm | 10.15 s (11% of the budget) |

Assumptions: rural 4G round trip 0.2–1 s (1 s used; up to about 3 s on 2G fallback), effective uplink 50 kbps, a 30 KB
vibration payload per 10 s window (10 s × 3,000 B/s ÷ 6,250 B/s = 4.8 s).

**Finding.** Cloud inference is feasible on latency alone, but only while a signal exists. Latency does not rule out the
cloud; connectivity does. (The later pipeline in Task C2 uses 30 s windows with a 10 s step; the simulation in Tasks D1/F
measured about 20 s from fault onset to first alert, still well inside 90 s.)

## 2. Bandwidth

Assumptions: temperature record 8 B (float32 value + 4 B timestamp), vibration 2 B per sample (16-bit), 50 door events a
day at 16 B, tariff ₹0.10 per MB, 1 MB = 1,000,000 B.

| Item | Working | Per truck per day |
|---|---|---|
| Vibration | 500 samples/s × 3 axes × 2 B × 86,400 s | 259.2 MB |
| Temperature | 8 B × 86,400 s | 0.69 MB |
| Door events | 50 × 16 B | 0.0008 MB |
| **Raw total** | | **259.89 MB**, costing **₹25.99** |
| Fleet, 85 trucks | 259.89 × 85 | 22.09 GB; ₹2,209.08 a day; about ₹8.06 lakh a year |
| Edge-processed (alerts + status) | 288 status packets × 100 B + 20 alerts × 256 B | 33,920 B = 0.034 MB; ₹0.0034 per truck per day; about ₹105 a year for the fleet |

Edge processing cuts volume and cost by a factor of **7,660 (99.99%)**.

## 3. Connectivity

The Nashik–Aurangabad route loses signal for 35–90 minutes at seven documented places: 245–630 minutes of cumulative dark
time (the cumulative figure looks long for that route; please confirm whether the 35–90 minutes applies to each place).
One gap is 23–60 times the 90 s budget.

| During a gap | Cloud-only | LogiEdge (edge) |
|---|---|---|
| Inference | None | Continues on the truck |
| Alerts | None; a fault goes unnoticed (it passes 3 °C after three minutes) | In-cab alarm within about 10 s |
| Data | Not recorded | Encrypted, hash-chained queue plus a 1 Hz temperature log; 90 minutes of raw data is only 16.2 MB |
| Signal returns | Nothing to recover | Ordered upload with retries and acknowledgements |

This is the failure that caused the ₹28 lakh vaccine-spoilage event on the same route.

## 4. Privacy

On-device inference keeps the 259 MB of daily raw data inside the cab; only class labels and signed alert metadata (about
34 KB a day) leave the truck. Data is encrypted at rest (AES-256, hardware-backed keys) and in transit (TLS 1.3 with mutual
authentication), and device-signed, hash-chained logs make tampering detectable. Contractually this supports a
data-minimisation commitment: raw cargo-condition data is processed and retained only on a key-protected device, and
disclosure is limited to defined alert fields. Device hardening and access policies are still required.

## Status

These are design calculations with stated assumptions, not field measurements. Measured round-trip times and uplink speeds
on the route would replace the assumed values.
