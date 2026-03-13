#!/usr/bin/env python3
"""DCS-CXL Storage Virtualization Workload.

Models NVMe-over-CXL fabric: remote hosts access pooled SSDs through CXL
switches, with DCS-CXL Engine coordinating direct SSD↔NIC D2D transfers.

This demonstrates the second key workload from the paper (Section 3.2):
device pooling / storage virtualization where DCS-CXL replaces Oasis-style
host-polled device queues.

IO Pattern:
  - Multiple clients issue 4KB-128KB IO requests to remote SSDs
  - Baseline: Host reads SSD data, writes to NIC buffer (CPU on data path)
  - DCS: Orchestrator issues SSD read → NIC write directly through CXL switch

Topology:
  [Host] ---- [Switch-0] ---- [Switch-1]
               /   |   \\        /   |   \\
          [SSD-0] [Orch] [NIC-0]  [SSD-1]  [NIC-1]

Transfer: SSD-0 → NIC-0 (local switch), SSD-1 → NIC-1 (local switch)
With host: Host reads from SSD via switch, writes to NIC via switch.

Parameters modeled:
  - IO size: 4KB (random IOPS), 64KB (sequential), 128KB (large sequential)
  - Queue depth: 1-128 (concurrent IOs)
  - SSD latency: modeled via DRAM with higher process_time (~5μs read)
  - NIC buffer: modeled via DRAM with low latency
"""

from mkcfg import utils, devices
import argparse

parser = argparse.ArgumentParser(description="DCS-CXL storage virtualization")
parser.add_argument("-m", "--mode", type=str, default="dcs",
                    choices=["dcs", "baseline"],
                    help="dcs = orchestrated D2D, baseline = host-mediated")
parser.add_argument("--io_size", type=int, default=4096,
                    help="IO request size in bytes (4096, 65536, 131072)")
parser.add_argument("--num_ios", type=int, default=1000,
                    help="total number of IO operations")
parser.add_argument("--queue_depth", type=int, default=32,
                    help="max concurrent IO operations")
parser.add_argument("--num_pairs", type=int, default=1,
                    help="number of SSD-NIC pairs (1-4)")
parser.add_argument("--ssd_latency", type=int, default=5000,
                    help="SSD read latency in ns (default 5000=5μs for NVMe)")
parser.add_argument("--switch_delay", type=int, default=25,
                    help="per-switch port delay (ns)")
parser.add_argument("--host_overhead", type=int, default=0,
                    help="host CPU overhead per IO in ns (baseline only)")
parser.add_argument("--block_size", type=int, default=4096,
                    help="transfer block size")
parser.add_argument("--log", type=str, default="")
parser.add_argument("--max_clock", type=int, default=500000000)
args = parser.parse_args()

MODE = args.mode
IO_SIZE = args.io_size
NUM_IOS = args.num_ios
QD = args.queue_depth
NUM_PAIRS = args.num_pairs
SSD_LATENCY = args.ssd_latency
SWITCH_DELAY = args.switch_delay
HOST_OVERHEAD = args.host_overhead
BLOCK_SIZE = args.block_size
LOG = args.log or f"output/storage-{MODE}-io{IO_SIZE}-n{NUM_IOS}-qd{QD}-pairs{NUM_PAIRS}.csv"

import sys
blocks_per_io = (IO_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
total_blocks = NUM_IOS * blocks_per_io * NUM_PAIRS
print(f"# Storage Virtualization Workload", file=sys.stderr)
print(f"# Mode: {MODE}", file=sys.stderr)
print(f"# IO size: {IO_SIZE} bytes, Block size: {BLOCK_SIZE}", file=sys.stderr)
print(f"# IOs per pair: {NUM_IOS}, Pairs: {NUM_PAIRS}", file=sys.stderr)
print(f"# Queue depth: {QD}", file=sys.stderr)
print(f"# SSD latency: {SSD_LATENCY} ns", file=sys.stderr)
print(f"# Total blocks: {total_blocks}", file=sys.stderr)

addr_stride = 1 << 32

cfg = utils.Config()
cfg.max_clock = args.max_clock
cfg.log_name = LOG
cfg.log_level = "INFO"

sw0 = devices.Switch("Switch-0")
sw0.delay = SWITCH_DELAY

all_devs = [sw0]

if MODE == "dcs":
    ssds = []
    nics = []
    orchs = []
    for i in range(NUM_PAIRS):
        # SSD: high process_time models NVMe read latency
        ssd = devices.DRAMsim3Interface(f"SSD-{i}")
        ssd.capacity = addr_stride
        ssd.start = i * addr_stride * 2
        ssd.wr_ratio = 0.0  # Read source
        ssd.process_time = SSD_LATENCY // 10  # Approximate: DRAMsim3 process_time + DRAM latency ≈ SSD latency
        ssds.append(ssd)
        all_devs.append(ssd)

        # NIC buffer: fast write destination
        nic = devices.DRAMsim3Interface(f"NIC-{i}")
        nic.capacity = addr_stride
        nic.start = i * addr_stride * 2 + addr_stride
        nic.wr_ratio = 1.0  # Write destination
        nic.process_time = 40  # Fast buffer
        nics.append(nic)
        all_devs.append(nic)

        # Orchestrator per pair
        orch = devices.Orchestrator(f"Orch-{i}")
        orch.num_ops = NUM_IOS
        orch.transfer_size = IO_SIZE
        orch.src_device = f"SSD-{i}"
        orch.dst_device = f"NIC-{i}"
        orch.src_base_addr = i * addr_stride * 2
        orch.dst_base_addr = i * addr_stride * 2 + addr_stride
        orch.max_outstanding = QD
        orch.max_outstanding_reads = QD
        orch.max_outstanding_writes = QD
        orch.block_size = BLOCK_SIZE
        orch.pipeline = True
        orch.cmd_queue_capacity = QD * 2
        orchs.append(orch)
        all_devs.append(orch)

    cfg.add_devices(all_devs)
    for i in range(NUM_PAIRS):
        cfg.connect(sw0, ssds[i])
        cfg.connect(sw0, nics[i])
        cfg.connect(orchs[i], sw0)

elif MODE == "baseline":
    total_requests = NUM_PAIRS * NUM_IOS * blocks_per_io
    host = devices.Requester("Host")
    host.interleave_type = "stream"
    host.interleave_param = total_requests
    host.q_capacity = QD
    host.block_size = BLOCK_SIZE
    host.burst_size = 1
    host.issue_delay = HOST_OVERHEAD
    host.coherent = False

    ssds = []
    nics = []
    for i in range(NUM_PAIRS):
        ssd = devices.DRAMsim3Interface(f"SSD-{i}")
        ssd.capacity = addr_stride
        ssd.start = i * addr_stride * 2
        ssd.wr_ratio = 0.0
        ssd.process_time = SSD_LATENCY // 10
        ssds.append(ssd)
        all_devs.append(ssd)

        nic = devices.DRAMsim3Interface(f"NIC-{i}")
        nic.capacity = addr_stride
        nic.start = i * addr_stride * 2 + addr_stride
        nic.wr_ratio = 1.0
        nic.process_time = 40
        nics.append(nic)
        all_devs.append(nic)

    all_devs.append(host)
    cfg.add_devices(all_devs)
    cfg.connect(host, sw0)
    for i in range(NUM_PAIRS):
        cfg.connect(sw0, ssds[i])
        cfg.connect(sw0, nics[i])

print(cfg)
