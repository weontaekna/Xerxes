#!/usr/bin/env python3
"""DCS-CXL Multi-Device Scalability Experiment.

Tests how DCS-CXL scales when coordinating transfers across N device pairs
concurrently, vs N host-mediated transfers that serialize through the CPU.

Topology (4 devices, 2 transfers):
  [Host] --+
           |
       [Switch-0] --- [Switch-1]
       /   |   \\       /   |   \\
  [Mem-0][Mem-1][Orch]  [Mem-2][Mem-3]

DCS mode: Orchestrator handles Transfer-0 (Mem-0→Mem-2) and Transfer-1 (Mem-1→Mem-3)
          concurrently. Host CPU does zero coordination work.
Baseline: Host handles both transfers sequentially (reads from Mem-0/1, writes to Mem-2/3).
"""

from mkcfg import utils, devices
import argparse

parser = argparse.ArgumentParser(description="DCS-CXL multi-device scalability")
parser.add_argument("-m", "--mode", type=str, default="dcs",
                    choices=["dcs", "baseline"],
                    help="dcs = orchestrated D2D, baseline = host-mediated")
parser.add_argument("-p", "--pairs", type=int, default=2,
                    help="number of concurrent device pairs (transfers)")
parser.add_argument("-n", "--num_ops", type=int, default=100,
                    help="number of transfer operations per pair")
parser.add_argument("-s", "--transfer_size", type=int, default=4096,
                    help="bytes per transfer")
parser.add_argument("--switch_delay", type=int, default=25,
                    help="CXL switch port delay (ns)")
parser.add_argument("--max_outstanding", type=int, default=32,
                    help="max outstanding per orchestrator / per requester")
parser.add_argument("--log", type=str, default="",
                    help="output log file")
parser.add_argument("--max_clock", type=int, default=50000000,
                    help="max simulation clock ticks")
args = parser.parse_args()

PAIRS = args.pairs
NUM_OPS = args.num_ops
TRANSFER_SIZE = args.transfer_size
SWITCH_DELAY = args.switch_delay
MAX_OUTSTANDING = args.max_outstanding
LOG = args.log or f"output/multi-{args.mode}-p{PAIRS}-n{NUM_OPS}-s{TRANSFER_SIZE}.csv"

# Create switches
sw0 = devices.Switch("Switch-0")
sw0.delay = SWITCH_DELAY
sw1 = devices.Switch("Switch-1")
sw1.delay = SWITCH_DELAY

cfg = utils.Config()
cfg.max_clock = args.max_clock
cfg.log_name = LOG
cfg.log_level = "INFO"

# Create device pairs
src_mems = []
dst_mems = []
addr_stride = 1 << 30  # 1GB per device

for i in range(PAIRS):
    src = devices.DRAMsim3Interface(f"SrcMem-{i}")
    src.capacity = addr_stride
    src.start = i * addr_stride * 2
    src.wr_ratio = 0.0

    dst = devices.DRAMsim3Interface(f"DstMem-{i}")
    dst.capacity = addr_stride
    dst.start = i * addr_stride * 2 + addr_stride
    dst.wr_ratio = 1.0

    src_mems.append(src)
    dst_mems.append(dst)

if args.mode == "dcs":
    # One orchestrator handles ALL transfers concurrently
    # (In a real system, the DCS-CXL Engine decomposes the op graph and
    # issues reads/writes to all device pairs in parallel)
    #
    # For ESF: we create one orchestrator per pair since the current
    # orchestrator model handles one src→dst pair.
    all_devs = list(src_mems) + list(dst_mems) + [sw0, sw1]
    orchs = []
    for i in range(PAIRS):
        orch = devices.Orchestrator(f"Orch-{i}")
        orch.num_ops = NUM_OPS
        orch.transfer_size = TRANSFER_SIZE
        orch.src_device = f"SrcMem-{i}"
        orch.dst_device = f"DstMem-{i}"
        orch.src_base_addr = i * addr_stride * 2
        orch.dst_base_addr = i * addr_stride * 2 + addr_stride
        orch.max_outstanding = MAX_OUTSTANDING
        orch.max_outstanding_reads = MAX_OUTSTANDING
        orch.max_outstanding_writes = MAX_OUTSTANDING
        orch.block_size = 64
        orch.pipeline = True
        orchs.append(orch)
        all_devs.append(orch)

    cfg.add_devices(all_devs)

    # Connect all source memories to Switch-0
    for src in src_mems:
        cfg.connect(sw0, src)
    # Connect all destination memories to Switch-1
    for dst in dst_mems:
        cfg.connect(sw1, dst)
    # Connect orchestrators to Switch-0
    for orch in orchs:
        cfg.connect(orch, sw0)
    # Inter-switch link
    cfg.connect(sw0, sw1)

elif args.mode == "baseline":
    # Host-mediated: Host issues reads to all sources, writes to all dests.
    # Total requests = PAIRS * NUM_OPS * blocks_per_op
    blocks_per_op = (TRANSFER_SIZE + 63) // 64
    total_requests = PAIRS * NUM_OPS * blocks_per_op

    host = devices.Requester("Host")
    host.interleave_type = "stream"
    host.interleave_param = total_requests
    host.q_capacity = MAX_OUTSTANDING
    host.block_size = 64
    host.burst_size = 1
    host.issue_delay = 0
    host.coherent = False

    all_devs = list(src_mems) + list(dst_mems) + [sw0, sw1, host]
    cfg.add_devices(all_devs)

    cfg.connect(host, sw0)
    for src in src_mems:
        cfg.connect(sw0, src)
    for dst in dst_mems:
        cfg.connect(sw1, dst)
    cfg.connect(sw0, sw1)

print(cfg)
