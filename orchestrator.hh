#pragma once
#ifndef XERXES_ORCHESTRATOR_HH
#define XERXES_ORCHESTRATOR_HH

#include "device.hh"
#include "utils.hh"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xerxes {

class OrchestratorConfig {
  public:
    // Processing delay for decomposing an operation graph into D2D transfers
    Tick decompose_delay = 50; // ns

    // Per-transfer scheduling overhead (looking up device registry, setting up DMA)
    Tick schedule_delay = 10; // ns

    // Command queue capacity
    size_t cmd_queue_capacity = 64;

    // Max outstanding D2D transfers
    size_t max_outstanding = 32;

    // Block size for transfers (cache line)
    size_t block_size = 64;

    // Workload parameters (for self-driven mode)
    size_t num_ops = 0;           // Number of D2D transfers to auto-submit
    size_t transfer_size = 4096;  // Bytes per transfer
    std::string src_device = "";  // Source device name (resolved after config)
    std::string dst_device = "";  // Destination device name (resolved after config)
    size_t src_base_addr = 0;     // Base address on source device
    size_t dst_base_addr = 0;     // Base address on destination device
};
} // namespace xerxes

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(xerxes::OrchestratorConfig,
                                        decompose_delay, schedule_delay,
                                        cmd_queue_capacity, max_outstanding,
                                        block_size, num_ops, transfer_size,
                                        src_device, dst_device,
                                        src_base_addr, dst_base_addr);

namespace xerxes {

// DCS-CXL Orchestrator: A Type-2 CXL device that coordinates
// direct device-to-device transfers without host CPU mediation.
//
// In the baseline (host-mediated), a host Requester reads from device A,
// then writes to device B. The Orchestrator replaces this with:
//   1. Host submits an operation descriptor to the Orchestrator
//   2. Orchestrator decomposes it into D2D transfer(s)
//   3. Orchestrator issues read to source device, gets response
//   4. Orchestrator issues write to destination device
//   5. Orchestrator notifies host of completion
//
// This models the CXL 3.0 UIO mechanism where the Orchestrator
// accesses other devices' HDM-DB memory directly via the CXL switch.
class Orchestrator : public Device {
  public:
    // An operation descriptor submitted by the host.
    // Represents a D2D transfer: read from src_dev, write to dst_dev.
    struct OpDesc {
        PktID op_id;
        TopoID src_dev;    // Source device (e.g., GPU with KV cache)
        TopoID dst_dev;    // Destination device (e.g., another GPU)
        Addr src_addr;     // Source address
        Addr dst_addr;     // Destination address
        size_t size;       // Transfer size in bytes
        Tick submitted;    // When the host submitted this op
        Tick started;      // When orchestrator started processing
        Tick completed;    // When the transfer finished

        // For multi-op graphs: dependencies (op_ids that must complete first)
        std::vector<PktID> deps;
    };

  private:
    Tick decompose_delay;
    Tick schedule_delay;
    size_t cmd_queue_capacity;
    size_t max_outstanding;
    size_t block_size;

    // Workload config (for self-driven mode)
    size_t num_ops;
    size_t transfer_size;
    std::string src_device_name;
    std::string dst_device_name;
    size_t src_base_addr;
    size_t dst_base_addr;
    size_t total_ops_submitted = 0;

    // Pending operation descriptors (submitted but not yet started)
    std::queue<OpDesc> pending_ops;

    // Active operations (issued reads, waiting for responses)
    struct ActiveOp {
        OpDesc desc;
        size_t blocks_total;
        size_t blocks_read_issued;
        size_t blocks_read_done;
        size_t blocks_write_issued;
        size_t blocks_write_done;
        enum Phase { READING, WRITING, DONE } phase;
    };
    std::unordered_map<PktID, ActiveOp> active_ops;

    // Map from packet ID to the op that owns it
    std::unordered_map<PktID, PktID> pkt_to_op;

    // Completed operation IDs (for dependency tracking)
    std::unordered_set<PktID> completed_ops;

    // Outstanding transfer count
    size_t outstanding_count = 0;

    // Statistics
    double total_ops = 0;
    double total_latency = 0;
    double total_blocks = 0;
    double total_decompose_wait = 0;
    double total_schedule_wait = 0;

    // Op ID counter
    PktID next_op_id = 0;

    bool deps_satisfied(const OpDesc &op) {
        for (auto dep : op.deps) {
            if (completed_ops.find(dep) == completed_ops.end())
                return false;
        }
        return true;
    }

    void try_start_ops(Tick tick) {
        while (!pending_ops.empty() &&
               active_ops.size() < cmd_queue_capacity) {
            auto &op = pending_ops.front();
            if (!deps_satisfied(op)) break;

            // Decompose: add processing delay
            op.started = tick + decompose_delay;
            total_decompose_wait += decompose_delay;

            size_t num_blocks = (op.size + block_size - 1) / block_size;

            ActiveOp active;
            active.desc = op;
            active.blocks_total = num_blocks;
            active.blocks_read_issued = 0;
            active.blocks_read_done = 0;
            active.blocks_write_issued = 0;
            active.blocks_write_done = 0;
            active.phase = ActiveOp::READING;

            active_ops[op.op_id] = active;
            pending_ops.pop();

            // Issue reads to source device
            issue_reads(op.op_id, op.started);
        }
    }

    void issue_reads(PktID op_id, Tick tick) {
        auto &active = active_ops[op_id];
        auto &desc = active.desc;

        while (active.blocks_read_issued < active.blocks_total &&
               outstanding_count < max_outstanding) {
            Tick issue_time = tick + active.blocks_read_issued * schedule_delay;
            total_schedule_wait += schedule_delay;

            auto pkt = PktBuilder()
                           .type(PacketType::NT_RD) // UIO read (non-temporal via CXL.io)
                           .addr(desc.src_addr +
                                 active.blocks_read_issued * block_size)
                           .payload(0)
                           .burst(1)
                           .sent(issue_time)
                           .src(self)
                           .dst(desc.src_dev)
                           .build();

            pkt_to_op[pkt.id] = op_id;
            outstanding_count++;
            active.blocks_read_issued++;

            XerxesLogger::debug()
                << name() << " issue D2D read pkt " << pkt.id << " for op "
                << op_id << " to dev " << desc.src_dev << " at " << issue_time
                << std::endl;
            send_pkt(pkt);
        }
    }

    void issue_writes(PktID op_id, Tick tick) {
        auto &active = active_ops[op_id];
        auto &desc = active.desc;

        while (active.blocks_write_issued < active.blocks_total &&
               outstanding_count < max_outstanding) {
            Tick issue_time = tick + active.blocks_write_issued * schedule_delay;

            auto pkt = PktBuilder()
                           .type(PacketType::NT_WT) // UIO write
                           .addr(desc.dst_addr +
                                 active.blocks_write_issued * block_size)
                           .payload(block_size)
                           .burst(1)
                           .sent(issue_time)
                           .src(self)
                           .dst(desc.dst_dev)
                           .build();

            pkt_to_op[pkt.id] = op_id;
            outstanding_count++;
            active.blocks_write_issued++;

            XerxesLogger::debug()
                << name() << " issue D2D write pkt " << pkt.id << " for op "
                << op_id << " to dev " << desc.dst_dev << " at " << issue_time
                << std::endl;
            send_pkt(pkt);
        }
    }

    // Try to make progress on all active ops (issue reads/writes where possible)
    void try_progress_active_ops(Tick tick) {
        for (auto &pair : active_ops) {
            if (outstanding_count >= max_outstanding) break;
            auto &active = pair.second;
            if (active.phase == ActiveOp::READING &&
                active.blocks_read_issued < active.blocks_total) {
                issue_reads(pair.first, tick);
            } else if (active.phase == ActiveOp::WRITING &&
                       active.blocks_write_issued < active.blocks_total) {
                issue_writes(pair.first, tick);
            }
        }
    }

    void complete_op(PktID op_id, Tick tick) {
        auto &active = active_ops[op_id];
        active.desc.completed = tick;

        total_ops++;
        total_latency += (tick - active.desc.submitted);
        total_blocks += active.blocks_total;

        completed_ops.insert(op_id);

        XerxesLogger::debug()
            << name() << " completed op " << op_id << " at " << tick
            << " (latency: " << (tick - active.desc.submitted) << " ns)"
            << std::endl;

        active_ops.erase(op_id);

        // Try to start more pending ops and progress existing ones
        try_start_ops(tick);
        try_progress_active_ops(tick);
    }

  public:
    Orchestrator(Simulation *sim, const OrchestratorConfig &config,
                 std::string name = "Orchestrator")
        : Device(sim, name), decompose_delay(config.decompose_delay),
          schedule_delay(config.schedule_delay),
          cmd_queue_capacity(config.cmd_queue_capacity),
          max_outstanding(config.max_outstanding),
          block_size(config.block_size),
          num_ops(config.num_ops),
          transfer_size(config.transfer_size),
          src_device_name(config.src_device),
          dst_device_name(config.dst_device),
          src_base_addr(config.src_base_addr),
          dst_base_addr(config.dst_base_addr) {}

    // Initialize workload: resolve device names to IDs and submit ops.
    // Called after parse_config when name_to_id map is available.
    void init_workload(const std::map<std::string, TopoID> &name_to_id,
                       Tick start_tick = 0) {
        if (num_ops == 0 || src_device_name.empty() || dst_device_name.empty())
            return;

        auto src_it = name_to_id.find(src_device_name);
        auto dst_it = name_to_id.find(dst_device_name);
        ASSERT(src_it != name_to_id.end(),
               "Orchestrator src_device not found: " + src_device_name);
        ASSERT(dst_it != name_to_id.end(),
               "Orchestrator dst_device not found: " + dst_device_name);

        TopoID src_id = src_it->second;
        TopoID dst_id = dst_it->second;

        for (size_t i = 0; i < num_ops; i++) {
            Addr src_addr = src_base_addr + i * transfer_size;
            Addr dst_addr = dst_base_addr + i * transfer_size;
            submit_op(src_id, dst_id, src_addr, dst_addr,
                      transfer_size, start_tick);
            total_ops_submitted++;
        }
        XerxesLogger::debug()
            << name() << " submitted " << num_ops << " D2D ops ("
            << transfer_size << " B each) from " << src_device_name
            << " to " << dst_device_name << std::endl;
    }

    // Check if all submitted operations have completed
    bool all_done() const {
        return total_ops_submitted > 0 &&
               completed_ops.size() >= total_ops_submitted &&
               active_ops.empty() && pending_ops.empty();
    }

    bool has_workload() const { return num_ops > 0; }

    // Submit an operation descriptor (called externally or via packet)
    PktID submit_op(TopoID src_dev, TopoID dst_dev, Addr src_addr,
                    Addr dst_addr, size_t size, Tick tick,
                    std::vector<PktID> deps = {}) {
        OpDesc op;
        op.op_id = next_op_id++;
        op.src_dev = src_dev;
        op.dst_dev = dst_dev;
        op.src_addr = src_addr;
        op.dst_addr = dst_addr;
        op.size = size;
        op.submitted = tick;
        op.started = 0;
        op.completed = 0;
        op.deps = deps;

        pending_ops.push(op);
        try_start_ops(tick);

        return op.op_id;
    }

    void transit() override {
        auto pkt = receive_pkt();

        if (pkt.dst == self && pkt.is_rsp) {
            // Response from a device (read or write completion)
            auto it = pkt_to_op.find(pkt.id);
            if (it == pkt_to_op.end()) {
                XerxesLogger::warning()
                    << name() << " received unknown response pkt " << pkt.id
                    << std::endl;
                return;
            }

            PktID op_id = it->second;
            pkt_to_op.erase(it);
            outstanding_count--;

            auto op_it = active_ops.find(op_id);
            if (op_it == active_ops.end()) return;

            auto &active = op_it->second;

            if (active.phase == ActiveOp::READING) {
                active.blocks_read_done++;

                // If all reads done, transition to writing phase
                if (active.blocks_read_done == active.blocks_total) {
                    active.phase = ActiveOp::WRITING;
                    issue_writes(op_id, pkt.arrive);
                } else {
                    // Issue more reads if outstanding slots available
                    issue_reads(op_id, pkt.arrive);
                }
                // Also try to progress other active ops
                try_progress_active_ops(pkt.arrive);
            } else if (active.phase == ActiveOp::WRITING) {
                active.blocks_write_done++;

                if (active.blocks_write_done == active.blocks_total) {
                    active.phase = ActiveOp::DONE;
                    complete_op(op_id, pkt.arrive);
                } else {
                    issue_writes(op_id, pkt.arrive);
                    try_progress_active_ops(pkt.arrive);
                }
            }
        } else if (pkt.dst != self) {
            // Forward packet (shouldn't happen in normal operation)
            send_pkt(pkt);
        }
    }

    void log_stats(std::ostream &os) override {
        os << name() << " stats:" << std::endl;
        os << " * Total operations completed: " << total_ops << std::endl;
        os << " * Total blocks transferred: " << total_blocks << std::endl;
        if (total_ops > 0) {
            os << " * Average op latency (ns): " << total_latency / total_ops
               << std::endl;
            os << " * Average blocks per op: " << total_blocks / total_ops
               << std::endl;
            os << " * Average decompose overhead (ns): "
               << total_decompose_wait / total_ops << std::endl;
            os << " * Average schedule overhead per block (ns): "
               << total_schedule_wait / total_blocks << std::endl;
        }
    }
};
} // namespace xerxes

#endif // XERXES_ORCHESTRATOR_HH
