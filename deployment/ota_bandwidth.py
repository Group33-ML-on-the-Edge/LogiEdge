#!/usr/bin/env python3
"""Bandwidth saving of a layer-cached OTA model update across the fleet (Task D2).

Without layer caching a truck would download the whole image. With the model as the last
layer, a truck that already has the earlier layers downloads only the new model layer plus
a small manifest and config.

    full update per truck   = full image (compressed)
    layer update per truck  = model layer + manifest/config overhead
    saving                  = trucks x (full - layer)

1 MB = 1,000,000 bytes and the tariff is Rs 0.10 per MB, as in Task A1.

Examples:
  python3 ota_bandwidth.py                                   # illustrative sizes
  python3 ota_bandwidth.py --full-bytes 68000000 --layer-bytes 5200   # measured by ota_demo.sh
"""

from __future__ import annotations

import argparse
import sys

MB = 1_000_000


def ota_saving(full_bytes: float, layer_bytes: float, overhead_bytes: float = 10_000,
               trucks: int = 85, tariff_per_mb: float = 0.10) -> dict:
    layer_total = layer_bytes + overhead_bytes
    per_truck_saving = full_bytes - layer_total
    return {
        "trucks": trucks,
        "full_per_truck_mb": full_bytes / MB,
        "layer_per_truck_mb": layer_total / MB,
        "full_fleet_mb": trucks * full_bytes / MB,
        "layer_fleet_mb": trucks * layer_total / MB,
        "saving_fleet_mb": trucks * per_truck_saving / MB,
        "saving_percent": 100 * per_truck_saving / full_bytes,
        "full_cost": trucks * full_bytes / MB * tariff_per_mb,
        "layer_cost": trucks * layer_total / MB * tariff_per_mb,
        "saving_cost": trucks * per_truck_saving / MB * tariff_per_mb,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full-bytes", type=float, default=65 * MB, help="compressed full image (default 65 MB, illustrative)")
    p.add_argument("--layer-bytes", type=float, default=6_000, help="compressed model layer (default 6 KB, illustrative)")
    p.add_argument("--overhead-bytes", type=float, default=10_000, help="manifest + config per update (default 10 KB)")
    p.add_argument("--trucks", type=int, default=85)
    p.add_argument("--tariff", type=float, default=0.10, help="Rs per MB (default 0.10)")
    a = p.parse_args(argv)
    r = ota_saving(a.full_bytes, a.layer_bytes, a.overhead_bytes, a.trucks, a.tariff)
    print(f"Trucks: {r['trucks']}")
    print(f"Full image update:   {r['full_per_truck_mb']:.3f} MB per truck -> {r['full_fleet_mb']:,.2f} MB fleet, Rs {r['full_cost']:,.2f}")
    print(f"Model-layer update:  {r['layer_per_truck_mb']:.3f} MB per truck -> {r['layer_fleet_mb']:,.2f} MB fleet, Rs {r['layer_cost']:,.2f}")
    print(f"Saving:              {r['saving_fleet_mb']:,.2f} MB ({r['saving_percent']:.2f}%), Rs {r['saving_cost']:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
