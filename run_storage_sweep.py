#!/usr/bin/env python3
"""DCS-CXL Storage Virtualization Sweep.

Systematically runs storage workload experiments across IO sizes,
queue depths, device pair counts, and host overhead levels.
Produces consolidated CSV for analysis and plotting.

Usage:
    python3 run_storage_sweep.py [--quick]
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time

ESF_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(ESF_DIR, "build", "Xerxes")
CONFIG_SCRIPT = os.path.join(ESF_DIR, "configs", "dcs-cxl-storage-virt.py")
OUT_DIR = os.path.join(ESF_DIR, "output")

FULL_SWEEP = {
    "io_size":        [4096, 65536, 131072],
    "queue_depth":    [1, 4, 8, 16, 32, 64, 128],
    "num_pairs":      [1, 2, 4],
    "host_overhead":  [0, 100, 500, 1000],
    "ssd_latency":    [5000],
}

QUICK_SWEEP = {
    "io_size":        [4096, 65536],
    "queue_depth":    [1, 16, 64],
    "num_pairs":      [1, 2],
    "host_overhead":  [0, 500],
    "ssd_latency":    [5000],
}

MODES = ["dcs", "baseline"]


def parse_stats(output):
    stats = {}
    m = re.search(r"Average op latency \(ns\): ([0-9.e+]+)", output)
    if m:
        stats["avg_latency_ns"] = float(m.group(1))

    m = re.search(r"Total operations completed: ([0-9.]+)", output)
    if m:
        stats["ops_completed"] = float(m.group(1))

    m = re.search(r"Effective bandwidth \(GB/s\): ([0-9.e+]+)", output)
    if m:
        stats["effective_bw_gbps"] = float(m.group(1))

    m = re.search(r"Total data transferred \(bytes\): ([0-9]+)", output)
    if m:
        stats["total_bytes"] = int(m.group(1))

    m = re.search(r"Total wall time \(ns\): ([0-9]+)", output)
    if m:
        stats["total_wall_ns"] = int(m.group(1))

    # Requester (baseline) aggregate stats
    agg = re.search(
        r"Aggregate:.*?Bandwidth \(GB/s\): ([0-9.e+]+).*?"
        r"Average latency \(ns\): ([0-9.e+]+)",
        output, re.DOTALL)
    if agg:
        if "effective_bw_gbps" not in stats:
            stats["effective_bw_gbps"] = float(agg.group(1))
        if "avg_latency_ns" not in stats:
            stats["avg_latency_ns"] = float(agg.group(2))

    m = re.search(r"Duration: (\d+) ms", output)
    if m:
        stats["sim_duration_ms"] = int(m.group(1))

    return stats


def run_single(mode, io_size, num_ios, queue_depth, num_pairs,
               host_overhead, ssd_latency, max_clock=500000000):
    tag = (f"{mode}-io{io_size}-qd{queue_depth}-"
           f"pairs{num_pairs}-oh{host_overhead}")

    cmd_gen = [
        sys.executable, CONFIG_SCRIPT,
        "-m", mode,
        "--io_size", str(io_size),
        "--num_ios", str(num_ios),
        "--queue_depth", str(queue_depth),
        "--num_pairs", str(num_pairs),
        "--ssd_latency", str(ssd_latency),
        "--host_overhead", str(host_overhead),
        "--log", os.path.join(OUT_DIR, f"storage-sweep-{tag}.csv"),
        "--max_clock", str(max_clock),
    ]

    gen_result = subprocess.run(cmd_gen, capture_output=True, text=True,
                                cwd=ESF_DIR)
    if gen_result.returncode != 0:
        return None

    toml_content = gen_result.stdout
    toml_path = os.path.join(OUT_DIR, f"storage-sweep-{tag}.toml")
    with open(toml_path, "w") as f:
        f.write(toml_content)

    sim_result = subprocess.run([BINARY, toml_path], capture_output=True,
                                text=True, cwd=ESF_DIR, timeout=600)
    combined = sim_result.stdout + "\n" + sim_result.stderr
    if sim_result.returncode != 0:
        print(f"  FAIL: {tag}: {sim_result.stderr[:200]}", file=sys.stderr)
        return None

    stats = parse_stats(combined)

    # For DCS: compute IOPS from ops and wall time
    if "total_wall_ns" in stats and stats["total_wall_ns"] > 0:
        stats["iops"] = (stats.get("ops_completed", num_ios) /
                         (stats["total_wall_ns"] / 1e9))
    elif "avg_latency_ns" in stats and stats["avg_latency_ns"] > 0:
        # Estimate: with queueing, effective IOPS depends on concurrency
        stats["iops"] = 0.0

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--num_ios", type=int, default=500,
                        help="IOs per pair per run")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    output_file = args.output or os.path.join(
        OUT_DIR, f"storage-sweep-{'quick' if args.quick else 'full'}-results.csv")

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(BINARY):
        print(f"ERROR: Binary not found at {BINARY}", file=sys.stderr)
        sys.exit(1)

    keys = list(sweep.keys())
    values = list(sweep.values())
    combos = list(itertools.product(*values))

    total_runs = len(combos) * len(MODES)
    print(f"Storage sweep: {len(combos)} combos x {len(MODES)} modes = "
          f"{total_runs} total runs")
    print(f"Params: {sweep}")
    print(f"Output: {output_file}\n")

    results = []
    completed = 0
    t0 = time.time()

    for combo in combos:
        params = dict(zip(keys, combo))
        for mode in MODES:
            completed += 1
            # Baseline only needs non-zero host_overhead to differ;
            # DCS ignores host_overhead so skip redundant DCS runs
            if mode == "dcs" and params["host_overhead"] != 0:
                print(f"[{completed}/{total_runs}] DCS oh={params['host_overhead']} "
                      f"... SKIP (DCS ignores overhead)")
                continue

            tag = (f"{mode} io={params['io_size']} qd={params['queue_depth']} "
                   f"pairs={params['num_pairs']} oh={params['host_overhead']}")
            print(f"[{completed}/{total_runs}] {tag} ... ", end="", flush=True)

            stats = run_single(
                mode=mode,
                io_size=params["io_size"],
                num_ios=args.num_ios,
                queue_depth=params["queue_depth"],
                num_pairs=params["num_pairs"],
                host_overhead=params["host_overhead"],
                ssd_latency=params["ssd_latency"],
            )

            if stats:
                row = {**params, "mode": mode, "num_ios": args.num_ios, **stats}
                results.append(row)
                bw = stats.get("effective_bw_gbps", 0)
                lat = stats.get("avg_latency_ns", 0)
                print(f"BW={bw:.2f} GB/s, lat={lat:.0f} ns")
            else:
                print("FAILED")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)}/{total_runs} runs in {elapsed:.1f}s")

    if results:
        fieldnames = ["mode", "io_size", "queue_depth", "num_pairs",
                      "host_overhead", "ssd_latency", "num_ios",
                      "avg_latency_ns", "ops_completed", "effective_bw_gbps",
                      "total_bytes", "total_wall_ns", "iops",
                      "sim_duration_ms"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
