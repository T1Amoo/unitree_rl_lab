#!/usr/bin/env python3
"""Fail-closed wall-clock preflight for the A1 camera deployment.

The camera stamps detections with the Jetson wall clock while the local ball
bridge extrapolates them with the deployment-computer wall clock.  Measuring
the offset here prevents a clock error from being mistaken for camera latency.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    offset_ms: float
    rtt_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.231")
    parser.add_argument("--user", default="jetson")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--max-offset-ms", type=float, default=10.0)
    parser.add_argument("--max-rtt-ms", type=float, default=40.0)
    parser.add_argument("--max-spread-ms", type=float, default=6.0)
    return parser.parse_args()


def measure_once(args: argparse.Namespace, control_path: str) -> Sample:
    command = [
        "ssh",
        "-S",
        control_path,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(args.connect_timeout_s))}",
        "-o",
        "ConnectionAttempts=1",
        f"{args.user}@{args.host}",
        "date +%s%N",
    ]
    epoch_before_ns = time.time_ns()
    mono_before_ns = time.monotonic_ns()
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.connect_timeout_s + 2.0,
    )
    mono_after_ns = time.monotonic_ns()
    epoch_after_ns = time.time_ns()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"ssh rc={result.returncode}"
        raise RuntimeError(detail)

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or not lines[-1].isdigit():
        raise RuntimeError(f"unexpected remote date output: {result.stdout!r}")
    remote_ns = int(lines[-1])
    midpoint_ns = (epoch_before_ns + epoch_after_ns) / 2.0
    return Sample(
        offset_ms=(remote_ns - midpoint_ns) / 1.0e6,
        rtt_ms=(mono_after_ns - mono_before_ns) / 1.0e6,
    )


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        print("CLOCK_SYNC_FAIL samples must be >= 3", file=sys.stderr)
        return 2
    if args.max_offset_ms <= 0.0 or args.max_rtt_ms <= 0.0 or args.max_spread_ms <= 0.0:
        print("CLOCK_SYNC_FAIL limits must be positive", file=sys.stderr)
        return 2

    samples: list[Sample] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="a1-clock-check-") as temp_dir:
        control_path = f"{temp_dir}/ssh.sock"
        master_command = [
            "ssh",
            "-M",
            "-S",
            control_path,
            "-fN",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(args.connect_timeout_s))}",
            "-o",
            "ConnectionAttempts=1",
            f"{args.user}@{args.host}",
        ]
        try:
            master = subprocess.run(
                master_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.connect_timeout_s + 2.0,
            )
            if master.returncode != 0:
                detail = master.stderr.strip() or master.stdout.strip() or f"ssh rc={master.returncode}"
                print(
                    f"CLOCK_SYNC_FAIL host={args.user}@{args.host} detail={detail}",
                    file=sys.stderr,
                )
                return 2
            for _ in range(args.samples):
                try:
                    samples.append(measure_once(args, control_path))
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    errors.append(str(exc))
        finally:
            subprocess.run(
                ["ssh", "-S", control_path, "-O", "exit", f"{args.user}@{args.host}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )

    usable = [sample for sample in samples if sample.rtt_ms <= args.max_rtt_ms]
    if len(usable) < 3:
        detail = errors[-1] if errors else "all SSH round trips exceeded the RTT limit"
        print(
            "CLOCK_SYNC_FAIL "
            f"host={args.user}@{args.host} usable={len(usable)}/{args.samples} "
            f"max_rtt_ms={args.max_rtt_ms:.1f} detail={detail}",
            file=sys.stderr,
        )
        return 2

    # The lowest-RTT samples have the smallest midpoint uncertainty.  Taking
    # their median is robust to one asymmetric SSH exchange.
    best = sorted(usable, key=lambda sample: sample.rtt_ms)[: min(5, len(usable))]
    offsets = [sample.offset_ms for sample in best]
    offset_ms = statistics.median(offsets)
    spread_ms = max(offsets) - min(offsets)
    best_rtt_ms = min(sample.rtt_ms for sample in best)

    fields = (
        f"host={args.user}@{args.host} offset_ms={offset_ms:+.3f} "
        f"best_rtt_ms={best_rtt_ms:.3f} spread_ms={spread_ms:.3f} "
        f"limits=offset:{args.max_offset_ms:.1f},rtt:{args.max_rtt_ms:.1f},spread:{args.max_spread_ms:.1f}"
    )
    if spread_ms > args.max_spread_ms:
        print(f"CLOCK_SYNC_FAIL {fields} reason=unstable_measurement", file=sys.stderr)
        return 2
    if abs(offset_ms) > args.max_offset_ms:
        print(f"CLOCK_SYNC_FAIL {fields} reason=clock_offset", file=sys.stderr)
        return 1

    print(f"CLOCK_SYNC_OK {fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
