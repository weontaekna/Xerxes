#!/usr/bin/env python3
"""Comprehensive DCS-CXL Results Analysis.

Consolidates all experiment data and produces publication-quality figures.
Reads from individual output CSVs and sweep result files.

Usage:
    python3 analyze_all_results.py [--figures-only] [--data-only]
"""

import argparse
import csv
import glob
import os
import re
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Data analysis only.", file=sys.stderr)


# =============================================================================
# Data Loading
# =============================================================================

def load_csv(path):
    """Load CSV file into list of dicts."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_scalability_results():
    """Load multi-switch scalability data."""
    return load_csv(os.path.join(OUT_DIR, "scalability-results.csv"))


def load_sweep_results():
    """Load parameter sweep data (full or quick)."""
    full = load_csv(os.path.join(OUT_DIR, "sweep-full-results.csv"))
    if full:
        return full
    return load_csv(os.path.join(OUT_DIR, "sweep-quick-results.csv"))


def load_sensitivity_results():
    """Load sensitivity sweep data."""
    return load_csv(os.path.join(OUT_DIR, "sensitivity-results.csv"))


def load_storage_results():
    """Load storage sweep results."""
    full = load_csv(os.path.join(OUT_DIR, "storage-sweep-full-results.csv"))
    if full:
        return full
    return load_csv(os.path.join(OUT_DIR, "storage-sweep-quick-results.csv"))


def load_ttft_results():
    """Load TTFT model results."""
    return load_csv(os.path.join(OUT_DIR, "ttft-all-models.csv"))


def load_streaming_results():
    """Load streaming TTFT results."""
    return load_csv(os.path.join(OUT_DIR, "streaming-ttft-results.csv"))


def load_llm_kv_stats():
    """Parse stats from LLM KV experiment stderr output files."""
    results = []
    for pattern in ["llm-dcs-*.csv", "llm-baseline-*.csv"]:
        for f in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
            basename = os.path.basename(f)
            # Parse config from filename: llm-{mode}-{model}-t{tokens}-d{decode}-b{block}.csv
            m = re.match(r"llm-(dcs|baseline)-(llama-\d+b)-t(\d+)-d(\d+)(?:-b(\d+))?\.csv", basename)
            if m:
                mode = m.group(1)
                model = m.group(2)
                tokens = int(m.group(3))
                decode_gpus = int(m.group(4))
                block_size = int(m.group(5)) if m.group(5) else 64
                # Get file size to check if it has data
                size = os.path.getsize(f)
                results.append({
                    "mode": mode, "model": model, "tokens": tokens,
                    "decode_gpus": decode_gpus, "block_size": block_size,
                    "file": basename, "size_bytes": size
                })
    return results


# =============================================================================
# Data Summary
# =============================================================================

def print_summary():
    """Print comprehensive data summary."""
    print("=" * 70)
    print("DCS-CXL Results Summary")
    print("=" * 70)

    # Scalability
    scal = load_scalability_results()
    if scal:
        print(f"\n## Multi-Switch Scalability ({len(scal)} data points)")
        modes = set(r.get("mode", r.get("config", "")) for r in scal)
        for mode in sorted(modes):
            rows = [r for r in scal if r.get("mode", r.get("config", "")) == mode]
            if rows and "effective_bw_gbps" in rows[0]:
                bws = [float(r["effective_bw_gbps"]) for r in rows if r.get("effective_bw_gbps")]
                if bws:
                    print(f"  {mode}: BW range {min(bws):.2f} - {max(bws):.2f} GB/s ({len(rows)} runs)")
    else:
        print("\n## Multi-Switch Scalability: No data")

    # Sweep
    sweep = load_sweep_results()
    if sweep:
        print(f"\n## Parameter Sweep ({len(sweep)} data points)")
        for mode in ["dcs", "dcs-pipe", "baseline"]:
            rows = [r for r in sweep if r.get("mode") == mode]
            if rows:
                bws = [float(r["effective_bw_gbps"]) for r in rows
                       if r.get("effective_bw_gbps") and float(r.get("effective_bw_gbps", 0)) > 0]
                lats = [float(r["avg_latency_ns"]) for r in rows
                        if r.get("avg_latency_ns") and float(r.get("avg_latency_ns", 0)) > 0]
                if bws:
                    print(f"  {mode}: BW {min(bws):.3f} - {max(bws):.3f} GB/s, "
                          f"Lat {min(lats):.0f} - {max(lats):.0f} ns ({len(rows)} runs)")
    else:
        print("\n## Parameter Sweep: No data")

    # Storage
    storage = load_storage_results()
    if storage:
        print(f"\n## Storage Virtualization ({len(storage)} data points)")
        for mode in ["dcs", "baseline"]:
            rows = [r for r in storage if r.get("mode") == mode]
            if rows:
                bws = [float(r["effective_bw_gbps"]) for r in rows
                       if r.get("effective_bw_gbps") and float(r.get("effective_bw_gbps", 0)) > 0]
                if bws:
                    print(f"  {mode}: BW {min(bws):.2f} - {max(bws):.2f} GB/s ({len(rows)} runs)")
    else:
        print("\n## Storage Virtualization: No data")

    # LLM KV
    llm = load_llm_kv_stats()
    if llm:
        print(f"\n## LLM KV Cache ({len(llm)} experiment files)")
        for mode in ["dcs", "baseline"]:
            rows = [r for r in llm if r["mode"] == mode]
            models = set(r["model"] for r in rows)
            for model in sorted(models):
                mrows = [r for r in rows if r["model"] == model]
                print(f"  {mode}/{model}: {len(mrows)} configs, "
                      f"tokens={set(r['tokens'] for r in mrows)}")
    else:
        print("\n## LLM KV Cache: No experiment files found")

    # TTFT
    ttft = load_ttft_results()
    if ttft:
        print(f"\n## TTFT Analysis ({len(ttft)} data points)")
        configs = set(r.get("config", "") for r in ttft)
        for cfg in sorted(configs):
            rows = [r for r in ttft if r.get("config") == cfg]
            if rows and "ttft_ms" in rows[0]:
                ttfts = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms")]
                if ttfts:
                    print(f"  {cfg}: TTFT {min(ttfts):.1f} - {max(ttfts):.1f} ms")
    else:
        print("\n## TTFT Analysis: No data")

    # Streaming
    stream = load_streaming_results()
    if stream:
        print(f"\n## Streaming TTFT ({len(stream)} data points)")
    else:
        print("\n## Streaming TTFT: No data")

    print("\n" + "=" * 70)


# =============================================================================
# Plotting
# =============================================================================

def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


COLORS = {
    'dcs': '#2196F3',
    'dcs-pipe': '#4CAF50',
    'baseline': '#F44336',
    'baseline-100': '#FF9800',
    'baseline-500': '#F44336',
    'baseline-1000': '#9C27B0',
}

MARKERS = {
    'dcs': 'o',
    'dcs-pipe': 's',
    'baseline': '^',
}


def plot_storage_results(storage_data):
    """Fig 9: Storage virtualization - IOPS and bandwidth vs queue depth."""
    if not storage_data:
        print("  Skipping storage plot: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Bandwidth vs queue depth for 4KB IOs
    ax = axes[0]
    for mode_label, oh_filter, color, marker in [
        ("DCS-CXL", ("dcs", None), COLORS['dcs'], 'o'),
        ("Baseline (0ns)", ("baseline", "0"), COLORS.get('baseline-100', '#666'), '^'),
        ("Baseline (500ns)", ("baseline", "500"), COLORS['baseline-500'], 'v'),
    ]:
        mode, oh = oh_filter
        rows = [r for r in storage_data
                if r["mode"] == mode
                and int(r["io_size"]) == 4096
                and int(r["num_pairs"]) == 1
                and (oh is None or str(r.get("host_overhead", "0")) == oh)]
        if not rows:
            continue
        qds = sorted(set(int(r["queue_depth"]) for r in rows))
        bws = []
        for qd in qds:
            qd_rows = [r for r in rows if int(r["queue_depth"]) == qd]
            if qd_rows:
                bws.append(float(qd_rows[0]["effective_bw_gbps"]))
            else:
                bws.append(0)
        ax.plot(qds, bws, marker=marker, label=mode_label, color=color,
                linewidth=2, markersize=6)

    ax.set_xlabel("Queue Depth")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("(a) 4KB Random IO")
    ax.set_xscale('log', base=2)
    ax.legend()

    # (b) Bandwidth vs queue depth for 64KB IOs
    ax = axes[1]
    for mode_label, oh_filter, color, marker in [
        ("DCS-CXL", ("dcs", None), COLORS['dcs'], 'o'),
        ("Baseline (0ns)", ("baseline", "0"), COLORS.get('baseline-100', '#666'), '^'),
        ("Baseline (500ns)", ("baseline", "500"), COLORS['baseline-500'], 'v'),
    ]:
        mode, oh = oh_filter
        rows = [r for r in storage_data
                if r["mode"] == mode
                and int(r["io_size"]) == 65536
                and int(r["num_pairs"]) == 1
                and (oh is None or str(r.get("host_overhead", "0")) == oh)]
        if not rows:
            continue
        qds = sorted(set(int(r["queue_depth"]) for r in rows))
        bws = []
        for qd in qds:
            qd_rows = [r for r in rows if int(r["queue_depth"]) == qd]
            if qd_rows:
                bws.append(float(qd_rows[0]["effective_bw_gbps"]))
            else:
                bws.append(0)
        ax.plot(qds, bws, marker=marker, label=mode_label, color=color,
                linewidth=2, markersize=6)

    ax.set_xlabel("Queue Depth")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("(b) 64KB Sequential IO")
    ax.set_xscale('log', base=2)
    ax.legend()

    plt.suptitle("Storage Virtualization: NVMe-over-CXL", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig9-storage-virt.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_storage_pairs(storage_data):
    """Fig 10: Storage multi-pair scalability."""
    if not storage_data:
        print("  Skipping storage pairs plot: no data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for mode_label, oh_filter, color, marker in [
        ("DCS-CXL", ("dcs", None), COLORS['dcs'], 'o'),
        ("Baseline (0ns)", ("baseline", "0"), COLORS.get('baseline-100', '#666'), '^'),
        ("Baseline (500ns)", ("baseline", "500"), COLORS['baseline-500'], 'v'),
    ]:
        mode, oh = oh_filter
        rows = [r for r in storage_data
                if r["mode"] == mode
                and int(r["io_size"]) == 4096
                and int(r["queue_depth"]) == 16
                and (oh is None or str(r.get("host_overhead", "0")) == oh)]
        if not rows:
            continue
        pairs = sorted(set(int(r["num_pairs"]) for r in rows))
        bws = []
        for p in pairs:
            p_rows = [r for r in rows if int(r["num_pairs"]) == p]
            if p_rows:
                bws.append(float(p_rows[0]["effective_bw_gbps"]))
            else:
                bws.append(0)
        ax.bar([x + 0.2 * (list(COLORS.keys()).index(mode) if mode in COLORS else 0)
                for x in range(len(pairs))],
               bws, width=0.25, label=mode_label, color=color)

    ax.set_xlabel("Number of SSD-NIC Pairs")
    ax.set_ylabel("Aggregate Bandwidth (GB/s)")
    ax.set_title("Storage: Multi-Pair Scalability (4KB, QD=16)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig10-storage-pairs.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_sweep_heatmap(sweep_data):
    """Fig 11: Parameter sweep heatmap - speedup of DCS over baseline."""
    if not sweep_data:
        print("  Skipping sweep heatmap: no data")
        return

    # Filter to n=100, sw=25 (representative)
    dcs_rows = [r for r in sweep_data
                if r["mode"] == "dcs"
                and int(r.get("num_ops", 0)) == 100
                and int(r.get("switch_delay", 0)) == 25]
    base_rows = [r for r in sweep_data
                 if r["mode"] == "baseline"
                 and int(r.get("num_ops", 0)) == 100
                 and int(r.get("switch_delay", 0)) == 25]

    if not dcs_rows or not base_rows:
        print("  Skipping sweep heatmap: insufficient data")
        return

    sizes = sorted(set(int(r["transfer_size"]) for r in dcs_rows))
    mos = sorted(set(int(r["max_outstanding"]) for r in dcs_rows))

    speedup_matrix = []
    for s in sizes:
        row = []
        for mo in mos:
            dcs = [r for r in dcs_rows
                   if int(r["transfer_size"]) == s and int(r["max_outstanding"]) == mo]
            base = [r for r in base_rows
                    if int(r["transfer_size"]) == s and int(r["max_outstanding"]) == mo]
            if dcs and base:
                dcs_lat = float(dcs[0].get("avg_latency_ns", 0))
                base_lat = float(base[0].get("avg_latency_ns", 0))
                if dcs_lat > 0 and base_lat > 0:
                    row.append(dcs_lat / base_lat)
                else:
                    row.append(1.0)
            else:
                row.append(1.0)
        speedup_matrix.append(row)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(speedup_matrix, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(mos)))
    ax.set_xticklabels(mos)
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels([f"{s}B" if s < 1024 else f"{s//1024}KB" for s in sizes])
    ax.set_xlabel("Max Outstanding Requests")
    ax.set_ylabel("Transfer Size")
    ax.set_title("DCS Latency / Baseline Latency\n(n=100, sw_delay=25ns)")
    plt.colorbar(im, ax=ax, label="Latency Ratio")

    # Annotate
    for i in range(len(sizes)):
        for j in range(len(mos)):
            val = speedup_matrix[i][j]
            ax.text(j, i, f"{val:.1f}x", ha="center", va="center",
                    fontsize=8, color="white" if val > 5 else "black")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig11-sweep-heatmap.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def plot_comprehensive_comparison(sweep_data, storage_data):
    """Fig 12: Comprehensive speedup summary across all workloads."""
    if not sweep_data and not storage_data:
        print("  Skipping comprehensive comparison: no data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    categories = []
    speedups = []
    colors = []

    # D2D transfer speedups (from sweep, n=100, sw=25, mo=32)
    if sweep_data:
        for size in [256, 4096, 16384]:
            dcs = [r for r in sweep_data
                   if r["mode"] == "dcs"
                   and int(r.get("num_ops", 0)) == 100
                   and int(r.get("switch_delay", 0)) == 25
                   and int(r.get("max_outstanding", 0)) == 32
                   and int(r.get("transfer_size", 0)) == size]
            base = [r for r in sweep_data
                    if r["mode"] == "baseline"
                    and int(r.get("num_ops", 0)) == 100
                    and int(r.get("switch_delay", 0)) == 25
                    and int(r.get("max_outstanding", 0)) == 32
                    and int(r.get("transfer_size", 0)) == size]
            if dcs and base:
                dcs_bw = float(dcs[0].get("effective_bw_gbps", 0))
                base_bw = float(base[0].get("effective_bw_gbps", 0))
                if base_bw > 0 and dcs_bw > 0:
                    label = f"D2D {size}B" if size < 1024 else f"D2D {size//1024}KB"
                    categories.append(label)
                    speedups.append(dcs_bw / base_bw if dcs_bw > base_bw else base_bw / dcs_bw)
                    colors.append(COLORS['dcs'])

    # Storage speedups
    if storage_data:
        for io_size, label in [(4096, "Storage 4KB"), (65536, "Storage 64KB")]:
            dcs = [r for r in storage_data
                   if r["mode"] == "dcs"
                   and int(r["io_size"]) == io_size
                   and int(r["queue_depth"]) == 16
                   and int(r["num_pairs"]) == 1]
            base = [r for r in storage_data
                    if r["mode"] == "baseline"
                    and int(r["io_size"]) == io_size
                    and int(r["queue_depth"]) == 16
                    and int(r["num_pairs"]) == 1
                    and int(r.get("host_overhead", 0)) == 500]
            if dcs and base:
                dcs_bw = float(dcs[0]["effective_bw_gbps"])
                base_bw = float(base[0]["effective_bw_gbps"])
                if base_bw > 0:
                    categories.append(label)
                    speedups.append(dcs_bw / base_bw)
                    colors.append('#FF9800')

    if categories:
        bars = ax.barh(range(len(categories)), speedups, color=colors)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_xlabel("DCS-CXL Speedup over Baseline")
        ax.set_title("DCS-CXL Speedup Summary (500ns host overhead)")
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)

        for i, (bar, val) in enumerate(zip(bars, speedups)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}x', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig12-speedup-summary.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def generate_all_figures():
    """Generate all publication figures."""
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    print("\nGenerating figures...")

    storage = load_storage_results()
    sweep = load_sweep_results()

    plot_storage_results(storage)
    plot_storage_pairs(storage)
    plot_sweep_heatmap(sweep)
    plot_comprehensive_comparison(sweep, storage)

    print(f"\nAll figures saved to {FIG_DIR}/")


# =============================================================================
# Numerical Results for Paper
# =============================================================================

def generate_paper_numbers():
    """Extract key numbers for paper text."""
    print("\n" + "=" * 70)
    print("Key Numbers for Paper")
    print("=" * 70)

    storage = load_storage_results()
    if storage:
        print("\n### Storage Virtualization (Section 4 / Section 6)")
        # QD sweep comparison
        for io_size in [4096, 65536]:
            io_label = f"{io_size//1024}KB" if io_size >= 1024 else f"{io_size}B"
            dcs_qd16 = [r for r in storage
                        if r["mode"] == "dcs" and int(r["io_size"]) == io_size
                        and int(r["queue_depth"]) == 16 and int(r["num_pairs"]) == 1]
            base_0 = [r for r in storage
                      if r["mode"] == "baseline" and int(r["io_size"]) == io_size
                      and int(r["queue_depth"]) == 16 and int(r["num_pairs"]) == 1
                      and int(r.get("host_overhead", 0)) == 0]
            base_500 = [r for r in storage
                        if r["mode"] == "baseline" and int(r["io_size"]) == io_size
                        and int(r["queue_depth"]) == 16 and int(r["num_pairs"]) == 1
                        and int(r.get("host_overhead", 0)) == 500]

            if dcs_qd16 and base_0:
                dcs_bw = float(dcs_qd16[0]["effective_bw_gbps"])
                base_bw_0 = float(base_0[0]["effective_bw_gbps"])
                speedup_0 = dcs_bw / base_bw_0 if base_bw_0 > 0 else 0
                print(f"  {io_label} QD=16: DCS {dcs_bw:.1f} GB/s vs Baseline {base_bw_0:.2f} GB/s "
                      f"({speedup_0:.0f}x, 0ns overhead)")

            if dcs_qd16 and base_500:
                dcs_bw = float(dcs_qd16[0]["effective_bw_gbps"])
                base_bw_500 = float(base_500[0]["effective_bw_gbps"])
                speedup_500 = dcs_bw / base_bw_500 if base_bw_500 > 0 else 0
                print(f"  {io_label} QD=16: DCS {dcs_bw:.1f} GB/s vs Baseline {base_bw_500:.2f} GB/s "
                      f"({speedup_500:.0f}x, 500ns overhead)")

    sweep = load_sweep_results()
    if sweep:
        print("\n### D2D Transfer (Section 6)")
        for size in [256, 1024, 4096, 16384]:
            s_label = f"{size}B" if size < 1024 else f"{size//1024}KB"
            dcs = [r for r in sweep
                   if r["mode"] == "dcs"
                   and int(r.get("transfer_size", 0)) == size
                   and int(r.get("num_ops", 0)) == 100
                   and int(r.get("switch_delay", 0)) == 25
                   and int(r.get("max_outstanding", 0)) == 32]
            base = [r for r in sweep
                    if r["mode"] == "baseline"
                    and int(r.get("transfer_size", 0)) == size
                    and int(r.get("num_ops", 0)) == 100
                    and int(r.get("switch_delay", 0)) == 25
                    and int(r.get("max_outstanding", 0)) == 32]
            if dcs and base:
                dcs_lat = float(dcs[0].get("avg_latency_ns", 0))
                base_lat = float(base[0].get("avg_latency_ns", 0))
                if dcs_lat > 0 and base_lat > 0:
                    ratio = dcs_lat / base_lat
                    print(f"  {s_label}: DCS {dcs_lat:.0f}ns vs Baseline {base_lat:.0f}ns "
                          f"(ratio={ratio:.1f}x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    if not args.figures_only:
        print_summary()
        generate_paper_numbers()

    if not args.data_only and HAS_MPL:
        generate_all_figures()
    elif not args.data_only:
        print("\nInstall matplotlib to generate figures: pip install matplotlib")


if __name__ == "__main__":
    main()
