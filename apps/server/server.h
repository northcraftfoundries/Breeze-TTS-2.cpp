#pragma once

#include <string>
#include <vector>

namespace breeze {

struct ServerOptions {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool use_gpu = true;
    bool webui = false;
    bool verbose = false;
    std::string voices_dir = "voices";
    int ws_port = 0; // 0 puts it on port + 1, negative turns it off
    int chunk_first = 4;
    int chunk_max = 0; // 0 picks for the backend
    int split_chars = 600; // 0 keeps long text in a single pass
    int n_threads = 0; // 0 picks the backend default
    std::vector<int> pinned_cpus; // empty when the process wasn't pinned; set by main() before run_server
};

int run_server(const ServerOptions & opts);

}
