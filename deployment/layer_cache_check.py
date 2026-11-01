#!/usr/bin/env python3
"""Predict which Dockerfile steps Docker would rebuild after a change (Task D2).

Docker's build cache rule: a step is reused only if its instruction text and, for COPY/ADD,
the content of the copied files are unchanged AND every step before it was reused. This
script applies that rule to two build contexts and prints what would be CACHED or REBUILT.

It is a static prediction of the design, not a substitute for a real `docker build`
(see ota_demo.sh for the real demonstration).

Usage:
  python3 layer_cache_check.py --dockerfile Dockerfile --context-a ./ctx_v1 --context-b ./ctx_v2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys


def parse_dockerfile(text: str) -> list[tuple[str, str]]:
    """Return [(INSTRUCTION, arguments)] with comments removed and continuation lines joined."""
    lines, buf = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if buf and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
        else:
            lines.append((buf + stripped).strip())
            buf = ""
    if buf:
        lines.append(buf.strip())
    out = []
    for line in lines:
        name, _, args = line.partition(" ")
        out.append((name.upper(), args.strip()))
    return out


def copy_sources(args: str) -> list[str]:
    parts = [p for p in args.split() if not p.startswith("--")]
    return parts[:-1]


def _hash_files(context: str, patterns: list[str]) -> str:
    h = hashlib.sha256()
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(context, pattern)))
        if not matches:
            raise FileNotFoundError(f"COPY source {pattern!r} not found in build context {context}")
        for path in matches:
            files = [path] if os.path.isfile(path) else sorted(
                os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs)
            for f in files:
                h.update(os.path.relpath(f, context).encode())
                with open(f, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def cache_keys(dockerfile_text: str, context: str) -> list[tuple[str, str, str]]:
    """[(instruction, arguments, cache key)] where each key also covers all earlier steps."""
    keys, parent = [], ""
    for name, args in parse_dockerfile(dockerfile_text):
        material = f"{parent}|{name} {args}"
        if name in ("COPY", "ADD"):
            material += "|" + _hash_files(context, copy_sources(args))
        key = hashlib.sha256(material.encode()).hexdigest()
        keys.append((name, args, key))
        parent = key
    return keys


# instructions that only change image metadata and add no filesystem data
METADATA = {"FROM", "ENV", "WORKDIR", "USER", "CMD", "ENTRYPOINT", "EXPOSE", "LABEL", "ARG", "HEALTHCHECK"}


def compare(dockerfile_text: str, ctx_a: str, ctx_b: str) -> list[dict]:
    a, b = cache_keys(dockerfile_text, ctx_a), cache_keys(dockerfile_text, ctx_b)
    rows = []
    for i, ((name, args, ka), (_, _, kb)) in enumerate(zip(a, b), start=1):
        rows.append({"step": i, "instruction": f"{name} {args}", "name": name,
                     "status": "CACHED" if ka == kb else "REBUILT",
                     "adds_data": name not in METADATA})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dockerfile", default="Dockerfile")
    p.add_argument("--context-a", required=True, help="build context before the change")
    p.add_argument("--context-b", required=True, help="build context after the change")
    args = p.parse_args(argv)
    with open(args.dockerfile) as f:
        rows = compare(f.read(), args.context_a, args.context_b)
    for r in rows:
        kind = "data layer" if r["adds_data"] else "metadata"
        print(f"{r['step']:2d}  {r['status']:8s} {kind:10s} {r['instruction'][:70]}")
    rebuilt = [r for r in rows if r["status"] == "REBUILT" and r["adds_data"]]
    print(f"\nData layers rebuilt: {len(rebuilt)} -> " + ", ".join(r["instruction"][:40] for r in rebuilt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
