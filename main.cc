#include "def.hh"
#include "dramsim3_interface.hh"
#include "orchestrator.hh"
#include "requester.hh"
#include "xerxes_standalone.hh"

#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    // Initialize a global simulation object.
    auto sim = xerxes::Simulation{};
    xerxes::init_sim(&sim);

    // Parse the TOML configuration file.
    std::string config_file = "";
    if (argc > 1)
        config_file = argv[1];
    else
        config_file = "configs/test.toml";

    // check if the file exists
    std::ifstream f(config_file);
    if (!f.good()) {
        std::cerr << "File " << config_file << " does not exist." << std::endl;
        return 1;
    }
    std::cout << "Config file: " << config_file << std::endl;

    auto ctx = xerxes::parse_config(config_file);
    auto fout = std::fstream(ctx.general.log_name, std::ios::out);
    // Set packet logger, which logs the latency components of each request.
    xerxes::set_pkt_logger(fout,
                           xerxes::str_to_log_level(ctx.general.log_level));

    auto &config = ctx.general;
    auto &requesters = ctx.requesters;
    auto &mems = ctx.mems;
    auto &orchestrators = ctx.orchestrators;

    // Run the simulation.
    std::cout << "Start simulation." << std::endl;
    for (auto &requester : requesters)
        requester->register_issue_event(0);
    xerxes::Tick clock_cnt = 0;
    xerxes::Tick last_curt = INT_MAX;

    bool has_orchestrator_workload = false;
    for (auto &orch : orchestrators) {
        if (orch->has_workload()) {
            has_orchestrator_workload = true;
            break;
        }
    }

    // Helper functions to check whether all requesters have issued
    // all their requests.
    auto check_all_issued = [&requesters]() {
        for (auto &requester : requesters) {
            if (!requester->all_issued()) {
                return false;
            }
        }
        return true;
    };
    // Helper function to check whether all requesters' queues are empty.
    // If true, expect that all requests are finished.
    auto check_all_empty = [&requesters]() {
        for (auto &requester : requesters) {
            if (!requester->q_empty()) {
                return false;
            }
        }
        return true;
    };
    // Check if all orchestrators have completed their workloads.
    auto check_orchestrators_done = [&orchestrators]() {
        for (auto &orch : orchestrators) {
            if (orch->has_workload() && !orch->all_done())
                return false;
        }
        return true;
    };
    std::vector<xerxes::Tick> mems_tick(mems.size(), 0);
    // Ticking all memories and synchronizing them to the current tick.
    auto clock_all_mems_to_tick = [&mems, &mems_tick, &config](
                                      xerxes::Tick tick, bool not_changed) {
        for (size_t i = 0; i < mems.size(); ++i)
            mems_tick[i] = 0;
        for (size_t i = 0; i < mems.size(); ++i) {
            for (int g = 0;
                 g < config.clock_granu && (mems_tick[i] < tick || not_changed);
                 ++g) {
                mems_tick[i] = mems[i]->clock();
            }
        }
    };

    // Simulation.
    auto start = std::chrono::high_resolution_clock::now();

    if (has_orchestrator_workload && requesters.empty()) {
        // Orchestrator-only mode: run until all ops complete.
        // Must keep clocking DRAM even when event queue is empty,
        // because DRAM callbacks generate new events.
        while (clock_cnt < config.max_clock) {
            if (check_orchestrators_done())
                break;
            if (!xerxes::events_empty()) {
                auto curt = xerxes::step();
                bool not_changed = last_curt == curt;
                last_curt = curt;
                clock_all_mems_to_tick(curt, not_changed);
            } else {
                // No events but orchestrators not done — use clock_until()
                // which blocks until a DRAM callback fires or pending
                // packets get issued.
                bool any_active = false;
                for (size_t i = 0; i < mems.size(); ++i) {
                    if (mems[i]->clock_until())
                        any_active = true;
                }
                if (!any_active && xerxes::events_empty())
                    break; // Nothing left to do
            }
            clock_cnt++;
            if (clock_cnt % 100000 == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - start);
                std::cout << "Clock: " << clock_cnt
                          << " Duration: " << duration.count() << " ms"
                          << std::endl;
            }
        }
    } else {
        // Original requester-driven mode (with optional orchestrators)
        while (clock_cnt < config.max_clock) {
            if (check_all_issued()) {
                break;
            }
            auto curt = xerxes::step();
            bool not_changed = last_curt == curt;
            last_curt = curt;
            clock_all_mems_to_tick(curt, not_changed);
            clock_cnt++;
        }

        while (clock_cnt < config.max_clock) {
            auto curt = xerxes::step();
            bool not_changed = last_curt == curt;
            last_curt = curt;
            clock_all_mems_to_tick(curt, not_changed);
            clock_cnt++;
            bool done = check_all_empty() && check_orchestrators_done();
            if (done)
                break;
            if (clock_cnt % 10000 == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - start);
                std::cout << "Clock: " << clock_cnt
                          << " Duration: " << duration.count() << " ms"
                          << std::endl;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation finished." << std::endl;
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    xerxes::log_stats(std::cerr);
    return 0;
}
