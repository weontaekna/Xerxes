#!/usr/bin/env python3
"""DCS-CXL D2D Transfer Experiment: Orchestrator vs Host-Mediated Baseline.

Topology:
  [Host] --+
           |
       [Switch-0] --- [Switch-1]
           |               |
       [Mem-0]         [Mem-1]
           |
    [Orchestrator]

DCS-CXL mode: Orchestrator reads from Mem-0, writes to Mem-1 (D2D).
Baseline mode: Host reads from Mem-0, writes to Mem-1 (host-mediated).
"""

from mkcfg import utils, devices
import argparse
import os

parser = argparse.ArgumentParser(description="DCS-CXL D2D experiment")
parser.add_argument("-m", "--mode", type=str, default="dcs",
                    choices=["dcs", "baseline"],
                    help="dcs = orchestrator D2D, baseline = host-mediated")
parser.add_argument("-n", "--num_ops", type=int, default=100,
                    help="number of D2D transfers")
parser.add_argument("-s", "--transfer_size", type=int, default=4096,
                    help="bytes per transfer")
parser.add_argument("--switch_delay", type=int, default=25,
                    help="CXL switch port delay (ns)")
parser.add_argument("--log", type=str, default="",
                    help="output log file")
parser.add_argument("--max_clock", type=int, default=10000000,
                    help="max simulation clock ticks")
args = parser.parse_args()

MODE = args.mode
NUM_OPS = args.num_ops
TRANSFER_SIZE = args.transfer_size
SWITCH_DELAY = args.switch_delay
LOG = args.log or f"output/dcs-cxl-{MODE}-n{NUM_OPS}-s{TRANSFER_SIZE}.csv"

# Memories (representing CXL-attached device memory)
mem0 = devices.DRAMsim3Interface("Mem-0")
mem0.capacity = 1 << 30
mem0.wr_ratio = 0.0  # read-only from requester perspective

mem1 = devices.DRAMsim3Interface("Mem-1")
mem1.capacity = 1 << 30
mem1.start = 1 << 30  # different address space
mem1.wr_ratio = 1.0   # write-only target

# Switches (CXL switches)
sw0 = devices.Switch("Switch-0")
sw0.delay = SWITCH_DELAY

sw1 = devices.Switch("Switch-1")
sw1.delay = SWITCH_DELAY

cfg = utils.Config()
cfg.max_clock = args.max_clock
cfg.log_name = LOG
cfg.log_level = "INFO"

if MODE == "dcs":
    # DCS-CXL mode: Orchestrator issues D2D transfers
    orch = devices.Orchestrator("Orchestrator")
    orch.num_ops = NUM_OPS
    orch.transfer_size = TRANSFER_SIZE
    orch.src_device = "Mem-0"
    orch.dst_device = "Mem-1"
    orch.src_base_addr = 0
    orch.dst_base_addr = 1 << 30
    orch.max_outstanding = 32
    orch.block_size = 64

    cfg.add_devices([mem0, mem1, sw0, sw1, orch])

    # Orchestrator connects to Switch-0 (same switch as Mem-0)
    cfg.connect(orch, sw0)
    cfg.connect(sw0, mem0)
    cfg.connect(sw1, mem1)
    cfg.connect(sw0, sw1)

elif MODE == "baseline":
    # Host-mediated baseline: Host reads from Mem-0, writes to Mem-1
    # The host issues interleaved reads to Mem-0 and writes to Mem-1
    host = devices.Requester("Host")
    host.interleave_type = "stream"
    # Total requests = NUM_OPS * (TRANSFER_SIZE / 64) for reads
    # Each block: read from Mem-0, then host writes to Mem-1
    # We approximate by sending NUM_OPS * blocks_per_op requests total
    blocks_per_op = (TRANSFER_SIZE + 63) // 64
    host.interleave_param = NUM_OPS * blocks_per_op
    host.q_capacity = 32
    host.block_size = 64
    host.burst_size = 1
    host.issue_delay = 0
    host.coherent = False

    # For baseline, Mem-0 is read (wr_ratio=0), Mem-1 is write (wr_ratio=1)
    mem0.wr_ratio = 0.0
    mem1.wr_ratio = 1.0

    cfg.add_devices([mem0, mem1, sw0, sw1, host])

    cfg.connect(host, sw0)
    cfg.connect(sw0, mem0)
    cfg.connect(sw1, mem1)
    cfg.connect(sw0, sw1)

print(cfg)
