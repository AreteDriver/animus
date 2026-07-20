#!/usr/bin/env python3
"""Profile kernel benchmarks and emit a hotspot report.

Usage::

    python scripts/profile_benchmarks.py [output_dir]

Runs the kernel benchmark suite under cProfile, then writes:
- ``kernel-benchmark.prof`` — raw cProfile dump
- ``kernel-benchmark-hotspots.txt`` — top 20 functions by cumulative time
- ``kernel-benchmark-callers.txt`` — top 20 callers by cumulative time

The ``.prof`` file can be visualised with snakeviz::

    pip install snakeviz
    snakeviz output/kernel-benchmark.prof
"""
from __future__ import annotations

import cProfile
import pstats
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(".benchmarks")


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    prof_path = output_dir / "kernel-benchmark.prof"
    hotspots_path = output_dir / "kernel-benchmark-hotspots.txt"
    callers_path = output_dir / "kernel-benchmark-callers.txt"

    # Run benchmarks under cProfile
    pr = cProfile.Profile()
    pr.enable()
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "packages/kernel/tests/test_benchmarks.py",
            "--benchmark-only", "-q",
        ],
        capture_output=True,
        text=True,
    )
    pr.disable()
    pr.dump_stats(str(prof_path))

    # Parse stats
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)

    with open(hotspots_path, "w") as fh:
        fh.write(f"Kernel Benchmark Hotspots (top 20 by cumulative time)\n")
        fh.write("=" * 60 + "\n")
        stats.stream = fh
        stats.print_stats(20)

    with open(callers_path, "w") as fh:
        fh.write(f"Kernel Benchmark Callers (top 20 by cumulative time)\n")
        fh.write("=" * 60 + "\n")
        stats.stream = fh
        stats.print_callers(20)

    print(f"Profiling complete:")
    print(f"  Raw profile:    {prof_path}")
    print(f"  Hotspots:       {hotspots_path}")
    print(f"  Callers:        {callers_path}")
    print(f"  Benchmark exit: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
