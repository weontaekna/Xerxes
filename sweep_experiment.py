#!/usr/bin/env python3
"""DCS-CXL Parameter Sweep: Systematic comparison of DCS-CXL modes vs baseline.

Sweeps across transfer_size, switch_delay, max_outstanding, and num_ops.
Generates CSV results for analysis and plotting.

Usage:
    python3 sweep_experiment.py [--quick] [--output results.csv]
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
CONFIG_SCRIPT = os.path.join(ESF_DIR, "configs", "dcs-cxl-d2d.py")
GEN_DIR = os.path.join(ESF_DIR, "configs", "generated")
OUT_DIR = os.path.join(ESF_DIR, "output")

# Sweep parameter space
FULL_SWEEP = {
    "transfer_size": [64, 256, 1024, 4096, 16384, 65536],
    "switch_delay":  [10, 25, 50],
    "max_outstanding": [8, 16, 32, 64],
    "num_ops": [10, 100, 1000],
}

QUICK_SWEEP = {
    "transfer_size": [256, 4096, 65536],
    "switch_delay":  [10, 25],
    "max_outstanding": [16, 32],
    "num_ops": [10, 100],
}

MODES = ["dcs", "dcs-pipe", "baseline"]


def parse_stats(stderr_output):
    """Extract statistics from simulator stderr output."""
    stats = {}
    # Orchestrator or Requester stats
    m = re.search(r"Average op latency \(ns\): ([0-9.e+]+)", stderr_output)
    if m:
        stats["avg_latency_ns"] = float(m.group(1))

    m = re.search(r"Total operations completed: ([0-9.]+)", stderr_output)
    if m:
        stats["ops_completed"] = float(m.group(1))

    m = re.search(r"Total blocks transferred: ([0-9.]+)", stderr_output)
    if m:
        stats["blocks_transferred"] = float(m.group(1))

    m = re.search(r"Mode: (\w+)", stderr_output)
    if m:
        stats["orch_mode"] = m.group(1)

    # Requester stats (baseline mode) — parse Aggregate section
    m = re.search(r"Issued packets: (\d+)", stderr_output)
    if m:
        stats["total_requests"] = int(m.group(1))

    # Parse aggregate bandwidth and latency (appears after "Aggregate:" line)
    agg_match = re.search(
        r"Aggregate:.*?Bandwidth \(GB/s\): ([0-9.e+]+).*?"
        r"Average latency \(ns\): ([0-9.e+]+)",
        stderr_output, re.DOTALL)
    if agg_match:
        stats["aggregate_bw_gbps"] = float(agg_match.group(1))
        if "avg_latency_ns" not in stats:
            stats["avg_latency_ns"] = float(agg_match.group(2))

    # Duration
    m = re.search(r"Duration: (\d+) ms", stderr_output)
    if m:
        stats["sim_duration_ms"] = int(m.group(1))

    return stats


def get_total_sim_time_from_csv(csv_path):
    """Get last packet arrival time from CSV = total simulated time."""
    max_arrive = 0
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                arrive = int(row.get('arrive', 0))
                max_arrive = max(max_arrive, arrive)
    except Exception:
        pass
    return max_arrive


def run_single(mode, num_ops, transfer_size, switch_delay, max_outstanding,
               max_clock=50000000):
    """Run a single simulation and return parsed stats."""
    tag = f"{mode}-n{num_ops}-s{transfer_size}-sw{switch_delay}-mo{max_outstanding}"
    log_file = os.path.join(OUT_DIR, f"sweep-{tag}.csv")
    toml_file = os.path.join(GEN_DIR, f"sweep-{tag}.toml")

    # Generate config
    cmd_gen = [
        sys.executable, CONFIG_SCRIPT,
        "-m", mode,
        "-n", str(num_ops),
        "-s", str(transfer_size),
        "--switch_delay", str(switch_delay),
        "--max_outstanding", str(max_outstanding),
        "--log", log_file,
        "--max_clock", str(max_clock),
    ]

    with open(toml_file, "w") as f:
        result = subprocess.run(cmd_gen, stdout=f, stderr=subprocess.PIPE,
                                text=True, cwd=ESF_DIR)
        if result.returncode != 0:
            print(f"  CONFIG FAIL: {tag}: {result.stderr}", file=sys.stderr)
            return None

    # Run simulation
    cmd_sim = [BINARY, toml_file]
    try:
        result = subprocess.run(cmd_sim, capture_output=True, text=True,
                                cwd=ESF_DIR, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {tag}", file=sys.stderr)
        return None

    combined_output = result.stdout + "\n" + result.stderr
    stats = parse_stats(combined_output)

    if result.returncode != 0:
        print(f"  SIM FAIL: {tag}: {result.stderr[:200]}", file=sys.stderr)
        return None

    # Compute total simulated time and effective bandwidth from CSV
    if os.path.isfile(log_file):
        total_sim_ns = get_total_sim_time_from_csv(log_file)
        stats["total_sim_time_ns"] = total_sim_ns
        total_bytes = num_ops * transfer_size
        stats["total_bytes"] = total_bytes
        if total_sim_ns > 0:
            stats["effective_bw_gbps"] = total_bytes / total_sim_ns
        else:
            stats["effective_bw_gbps"] = 0.0

    return stats


def main():
    parser = argparse.ArgumentParser(description="DCS-CXL parameter sweep")
    parser.add_argument("--quick", action="store_true",
                        help="use reduced parameter space")
    parser.add_argument("--output", type=str, default="",
                        help="output CSV file")
    parser.add_argument("--modes", type=str, nargs="+", default=MODES,
                        choices=MODES, help="modes to sweep")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    output_file = args.output or os.path.join(
        OUT_DIR, f"sweep-{'quick' if args.quick else 'full'}-results.csv")

    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check binary exists
    if not os.path.isfile(BINARY):
        print(f"ERROR: Binary not found at {BINARY}. Run 'make' first.",
              file=sys.stderr)
        sys.exit(1)

    # Generate all parameter combinations
    keys = list(sweep.keys())
    values = list(sweep.values())
    combos = list(itertools.product(*values))

    total_runs = len(combos) * len(args.modes)
    print(f"Parameter sweep: {len(combos)} combos x {len(args.modes)} modes = "
          f"{total_runs} total runs")
    print(f"Sweep params: {sweep}")
    print(f"Output: {output_file}")
    print()

    results = []
    completed = 0
    start_time = time.time()

    for combo in combos:
        params = dict(zip(keys, combo))

        for mode in args.modes:
            completed += 1
            tag = (f"{mode} n={params['num_ops']} s={params['transfer_size']} "
                   f"sw={params['switch_delay']} mo={params['max_outstanding']}")
            print(f"[{completed}/{total_runs}] {tag} ... ", end="", flush=True)

            stats = run_single(
                mode=mode,
                num_ops=params["num_ops"],
                transfer_size=params["transfer_size"],
                switch_delay=params["switch_delay"],
                max_outstanding=params["max_outstanding"],
            )

            if stats:
                row = {**params, "mode": mode, **stats}
                results.append(row)
                lat = stats.get("avg_latency_ns", "N/A")
                print(f"avg_latency={lat:.1f} ns" if isinstance(lat, float)
                      else f"avg_latency={lat}")
            else:
                print("FAILED")

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(results)}/{total_runs} runs in {elapsed:.1f}s")

    # Write CSV
    if results:
        fieldnames = ["mode", "num_ops", "transfer_size", "switch_delay",
                      "max_outstanding", "avg_latency_ns", "ops_completed",
                      "blocks_transferred", "total_sim_time_ns",
                      "total_bytes", "effective_bw_gbps",
                      "aggregate_bw_gbps", "sim_duration_ms", "orch_mode"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_file}")
    else:
        print("No results to write!")


if __name__ == "__main__":
    main()
