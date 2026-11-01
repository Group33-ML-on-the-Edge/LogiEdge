# Hardware selection and justification (Tasks B1 and B2)

Three options were considered for the on-truck edge node. The full reports are
`reports/individual/LogiEdge_Task_B1_Constraint_Triangle.pdf` and `..._B2_Roofline_Analysis.pdf`; the later benchmark
reasoning is in `..._F3_Deployment_Recommendation.pdf`.

| # | Hardware | India price | Power |
|---|---|---|---|
| 1 | Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L) | about ₹15,000 per truck | 7.5 W |
| 2 | Jetson Orin Nano Super Developer Kit (67 TOPS) | about ₹45,000 per truck | 15 W under moderate load |
| 3 | STM32H7-based custom MCU with sensor ICs | about ₹3,500 per truck | 0.4 W |

## B1: the Constraint Triangle (performance, power, cost)

**Dominant vertex: power.** The 10 W AI budget on a 12 V truck supply is a hard physical cap. Latency does not discriminate:
every option alerts in about 10.15 s against a 90 s limit.

| Option | Detection to alarm | Device power | Draw on 12 V rail (÷ 0.90) | Share of 10 W (device / rail) | Pilot (85) | Full scale (265) |
|---|---|---|---|---|---|---|
| 1 | 10.15 s | 7.5 W | 8.33 W (0.69 A) | 75% / 83% | ₹12.75 lakh | ₹39.75 lakh |
| 2 | 10.15 s | 15 W | 16.67 W (1.39 A) | 150% / 167% | ₹38.25 lakh | ₹1.19 crore |
| 3 | 10.15 s | 0.4 W | 0.44 W (0.04 A) | 4% / 4% | ₹2.98 lakh | ₹9.28 lakh |

**Recommendation: Option 1 (Raspberry Pi 5 + AI HAT+).**

* **For Option 1:** passes the 10 W cap under both readings (device and rail), meets the 90 s limit with a large margin,
  costs about 46% of one ₹28 lakh spoilage loss for the 85-truck pilot, and runs Linux, so the local MQTT broker, encrypted log,
  TLS 1.3 uplink and over-the-air updates use standard software.
* **Against Option 2:** 15 W is 150% of the budget (167% on the rail) and costs three times as much. NVIDIA lists a 7 W mode for the
  Orin Nano 8 GB, which could fit under 10 W at reduced performance, but the cost remains and the 67 TOPS would go unused.
* **Against Option 3 (narrowly):** compute is sufficient, but it has about 1 MB of RAM and no Linux, so the architecture would need custom
  firmware, and custom hardware carries one-off engineering cost. It saves ₹9.78 lakh at 85 trucks and ₹30.48 lakh at 265 against
  Option 1, so it wins on cost only if that engineering stays below those amounts.

Risks of Option 1 and mitigations: thin power margin (measure under load), not automotive-rated (industrial enclosure, 12 V
converter with surge protection, industrial storage), accelerator compatibility (run the small model on the CPU), parked-truck battery
drain (low-voltage cut-off).

## B2: arithmetic intensity and the Roofline model

Model: about 45 MFLOP and 18 MB of data per inference; Raspberry Pi 5 CPU about 16 GFLOP/s and 12 GB/s.

| Step | Working | Result |
|---|---|---|
| Arithmetic intensity | 45 MFLOP ÷ 18 MB | 2.5 FLOP/byte |
| Ridge point | 16 GFLOP/s ÷ 12 GB/s | 1.33 FLOP/byte |
| Classification | 2.5 > 1.33 | **compute-bound** |
| Latency lower bound | max(45 MFLOP ÷ 16 GFLOP/s, 18 MB ÷ 12 GB/s) = max(2.81 ms, 1.5 ms) | 2.81 ms |

What helps: cutting FLOPs or raising the compute ceiling (INT8 quantisation, pruning, a lighter model, more cores or the NPU).
What does not, for now: memory-traffic work. Compute-only gains are capped at 2.81 ÷ 1.5 = 1.875× before the model turns memory-bound;
INT8 avoids that cap because it also shrinks the bytes.

## What the later tasks found

* The trained classifier is far smaller than the 45 MFLOP example (803 parameters, 752 multiply-accumulates), so inference takes
  microseconds and the 90 s SLA is consumed by the 30 s window and 10 s step, not by inference (Task F3).
* FP32 on the Pi 5 is recommended; INT8 and pruning save nothing that matters here. On the STM32H743 fallback (up to 2 MB flash,
  up to 1 MB RAM, per ST's datasheet) every variant uses under 0.2% of flash.
