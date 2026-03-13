#!/usr/bin/env python3
"""DCS-CXL Multi-Switch Topology Experiment.

Models a CXL 3.0 fabric with multiple switches (rack-scale).
Demonstrates DCS-CXL advantage: orchestrators distributed across switches
can coordinate D2D transfers without routing through a central host.

Topology (example with 2 switches, 2 pairs):
  [Host] ---- [Switch-0] ---- [Switch-1]
               /    \\             /    \\
          [Mem-0]  [Orch-0]   [Mem-2]  [Orch-1]
          [Mem-1]             [Mem-3]

Transfer-0: Mem-0 → Mem-2 (crosses switches, orchestrated by Orch-0)
Transfer-1: Mem-1 → Mem-3 (crosses switches, orchestrated by Orch-1)

Baseline: Host reads ALL from Mem-0/1 (via SW-0), writes ALL to Mem-2/3 (via SW-0→SW-1).
DCS:      Orch-0 reads Mem-0 (local), writes Mem-2 (via SW-0→SW-1).
          Orch-1 reads Mem-1 (local SW-0→SW-1), writes Mem-3 (local).
"""

from mkcfg import utils, devices
import argparse

parser = argparse.ArgumentParser(description="DCS-CXL multi-switch topology")
parser.add_argument("-m", "--mode", type=str, default="dcs",
                    choices=["dcs", "baseline"],
                    help="dcs = distributed orchestrators, baseline = host-mediated")
parser.add_argument("--switches", type=int, default=2,
                    help="number of switches in chain topology")
parser.add_argument("-n", "--num_ops", type=int, default=50,
                    help="ops per transfer pair")
parser.add_argument("-s", "--transfer_size", type=int, default=4096,
                    help="bytes per transfer")
parser.add_argument("--switch_delay", type=int, default=25,
                    help="per-switch port delay (ns)")
parser.add_argument("--max_outstanding", type=int, default=32)
parser.add_argument("--host_overhead", type=int, default=0,
                    help="baseline: extra host CPU processing delay per request (ns)")
parser.add_argument("--log", type=str, default="")
parser.add_argument("--max_clock", type=int, default=100000000)
args = parser.parse_args()

NUM_SWITCHES = args.switches
NUM_OPS = args.num_ops
TRANSFER_SIZE = args.transfer_size
SWITCH_DELAY = args.switch_delay
MAX_OUTSTANDING = args.max_outstanding
MODE = args.mode
HOST_OVERHEAD = args.host_overhead
LOG = args.log or f"output/multi-sw-{MODE}-sw{NUM_SWITCHES}-n{NUM_OPS}-s{TRANSFER_SIZE}.csv"

addr_stride = 1 << 30

# Create switches in a chain: SW-0 -- SW-1 -- SW-2 -- ...
switches = []
for i in range(NUM_SWITCHES):
    sw = devices.Switch(f"Switch-{i}")
    sw.delay = SWITCH_DELAY
    switches.append(sw)

cfg = utils.Config()
cfg.max_clock = args.max_clock
cfg.log_name = LOG
cfg.log_level = "INFO"

# Each switch has a source memory and a destination memory
# Transfer pattern: Mem on switch-i → Mem on switch-(i+1 mod N)
# This creates cross-switch transfers that test fabric scalability.
src_mems = []
dst_mems = []
for i in range(NUM_SWITCHES):
    src = devices.DRAMsim3Interface(f"SrcMem-{i}")
    src.capacity = addr_stride
    src.start = i * addr_stride * 2
    src.wr_ratio = 0.0
    src_mems.append(src)

    dst = devices.DRAMsim3Interface(f"DstMem-{i}")
    dst.capacity = addr_stride
    dst.start = i * addr_stride * 2 + addr_stride
    dst.wr_ratio = 1.0
    dst_mems.append(dst)

all_devs = list(src_mems) + list(dst_mems) + switches

if MODE == "dcs":
    # Distributed orchestrators: one per switch, each handles transfers
    # from local source to remote destination on next switch.
    orchs = []
    for i in range(NUM_SWITCHES):
        dst_idx = (i + 1) % NUM_SWITCHES
        orch = devices.Orchestrator(f"Orch-{i}")
        orch.num_ops = NUM_OPS
        orch.transfer_size = TRANSFER_SIZE
        orch.src_device = f"SrcMem-{i}"
        orch.dst_device = f"DstMem-{dst_idx}"
        orch.src_base_addr = i * addr_stride * 2
        orch.dst_base_addr = dst_idx * addr_stride * 2 + addr_stride
        orch.max_outstanding = MAX_OUTSTANDING
        orch.max_outstanding_reads = MAX_OUTSTANDING
        orch.max_outstanding_writes = MAX_OUTSTANDING
        orch.block_size = 64
        orch.pipeline = True
        orchs.append(orch)
        all_devs.append(orch)

    cfg.add_devices(all_devs)

    # Connect devices to their local switches
    for i in range(NUM_SWITCHES):
        cfg.connect(switches[i], src_mems[i])
        cfg.connect(switches[i], dst_mems[i])
        cfg.connect(orchs[i], switches[i])

    # Chain the switches
    for i in range(NUM_SWITCHES - 1):
        cfg.connect(switches[i], switches[i + 1])

elif MODE == "baseline":
    # Host connects to Switch-0, must route to all devices through the chain.
    blocks_per_op = (TRANSFER_SIZE + 63) // 64
    total_requests = NUM_SWITCHES * NUM_OPS * blocks_per_op

    host = devices.Requester("Host")
    host.interleave_type = "stream"
    host.interleave_param = total_requests
    host.q_capacity = MAX_OUTSTANDING
    host.block_size = 64
    host.burst_size = 1
    host.issue_delay = HOST_OVERHEAD  # Model CPU processing overhead
    host.coherent = False

    all_devs.append(host)
    cfg.add_devices(all_devs)

    cfg.connect(host, switches[0])
    for i in range(NUM_SWITCHES):
        cfg.connect(switches[i], src_mems[i])
        cfg.connect(switches[i], dst_mems[i])
    for i in range(NUM_SWITCHES - 1):
        cfg.connect(switches[i], switches[i + 1])

print(cfg)
