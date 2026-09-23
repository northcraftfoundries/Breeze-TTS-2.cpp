#include "breeze/cpu_affinity.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <cerrno>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX // windows.h's min/max macros shadow std::min/std::max used below
#endif
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0A00
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00 // GetSystemCpuSetInformation needs Windows 10+
#endif
#include <windows.h>
#endif

namespace breeze {

bool parse_cpu_list(const std::string & spec, std::vector<int> & cpus, std::string & err) {
    cpus.clear();
    if (spec.empty()) {
        err = "cpu list is empty";
        return false;
    }

    auto parse_int = [](const std::string & s, int & v) -> bool {
        if (s.empty()) return false;
        for (char c : s) if (!isdigit((unsigned char) c)) return false;
        try {
            size_t idx = 0;
            const long l = std::stol(s, &idx);
            if (idx != s.size() || l < 0 || l > INT_MAX) return false;
            v = (int) l;
        } catch (...) {
            return false;
        }
        return true;
    };

    std::vector<int> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? spec.size() : comma + 1;
        if (tok.empty()) {
            err = "empty entry in cpu list '" + spec + "'";
            return false;
        }
        const size_t dash = tok.find('-');
        if (dash == std::string::npos) {
            int v;
            if (!parse_int(tok, v)) {
                err = "invalid cpu id '" + tok + "' in '" + spec + "'";
                return false;
            }
            out.push_back(v);
        } else {
            int lo, hi;
            if (!parse_int(tok.substr(0, dash), lo) || !parse_int(tok.substr(dash + 1), hi)) {
                err = "invalid range '" + tok + "' in '" + spec + "'";
                return false;
            }
            if (hi < lo) {
                err = "descending range '" + tok + "' in '" + spec + "'";
                return false;
            }
            for (int v = lo; v <= hi; v++) out.push_back(v);
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    cpus = std::move(out);
    return true;
}

#ifdef __linux__
static bool read_sysfs_cpu_list(const char * path, std::vector<int> & cpus) {
    std::ifstream f(path);
    if (!f.good()) return false;
    std::string line;
    std::getline(f, line);
    if (line.empty()) return false;
    std::string err;
    return parse_cpu_list(line, cpus, err);
}
#endif

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
// no /sys/devices/cpu_core topology (older kernel, or hidden by a VM/WSL): fall back to
// asking each cpu directly. leaf 7 tells us the cpu is hybrid at all, leaf 0x1a tells us
// what kind of core we're currently running on.
static std::vector<int> detect_pcores_cpuid() {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return {};
    if (!(edx & (1u << 15))) return {}; // hybrid flag, leaf 7 subleaf 0 edx bit 15

    cpu_set_t original;
    CPU_ZERO(&original);
    if (pthread_getaffinity_np(pthread_self(), sizeof(original), &original) != 0) return {};

    std::vector<int> pcores;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &original)) continue;
        cpu_set_t only;
        CPU_ZERO(&only);
        CPU_SET(cpu, &only);
        if (pthread_setaffinity_np(pthread_self(), sizeof(only), &only) != 0) continue;
        unsigned a, b, c, d;
        if (__get_cpuid_count(0x1a, 0, &a, &b, &c, &d)) {
            const unsigned core_type = (a >> 24) & 0xff; // 0x40 core/p, 0x20 atom/e
            if (core_type == 0x40) pcores.push_back(cpu);
        }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(original), &original);

    std::sort(pcores.begin(), pcores.end());
    return pcores;
}
#endif

#ifdef _WIN32
static std::vector<int> detect_pcores_windows() {
    DWORD len = 0;
    GetSystemCpuSetInformation(nullptr, 0, &len, nullptr, 0);
    if (len == 0) return {};
    std::vector<uint8_t> buf(len);
    if (!GetSystemCpuSetInformation((PSYSTEM_CPU_SET_INFORMATION) buf.data(), len, &len, nullptr, 0)) {
        return {};
    }

    std::vector<std::pair<uint8_t, int>> entries;
    uint8_t max_class = 0, min_class = 0xff;
    size_t offset = 0;
    while (offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= len) {
        auto * info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + offset);
        if (info->Size == 0) break;
        if (info->Type == CpuSetInformation) {
            const uint8_t cls = info->CpuSet.EfficiencyClass;
            entries.emplace_back(cls, (int) info->CpuSet.LogicalProcessorIndex);
            max_class = std::max(max_class, cls);
            min_class = std::min(min_class, cls);
        }
        offset += info->Size;
    }
    if (entries.empty() || max_class == min_class) return {}; // no heterogeneity reported

    std::vector<int> pcores;
    for (const auto & e : entries) if (e.first == max_class) pcores.push_back(e.second);
    std::sort(pcores.begin(), pcores.end());
    return pcores;
}
#endif

std::vector<int> detect_pcores() {
#ifdef __linux__
    std::vector<int> cpus;
    if (read_sysfs_cpu_list("/sys/devices/cpu_core/cpus", cpus)) return cpus;
#if defined(__x86_64__) || defined(__i386__)
    return detect_pcores_cpuid();
#else
    return {};
#endif
#elif defined(_WIN32)
    return detect_pcores_windows();
#else
    return {};
#endif
}

// pins the process rather than ggml's own threadpool cpumask because this build uses
// OpenMP: ggml ignores the threadpool cpumask under GGML_USE_OPENMP, and the OpenMP
// worker threads inherit whatever affinity the process has when they're first spawned
// on the initial graph compute. so pinning has to happen before the model runs anything.
bool pin_process(const std::vector<int> & cpus, std::string & err) {
    if (cpus.empty()) {
        err = "cpu list is empty";
        return false;
    }
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            err = "cpu id " + std::to_string(cpu) + " is out of range (max " + std::to_string(CPU_SETSIZE - 1) + ")";
            return false;
        }
        CPU_SET(cpu, &set);
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        err = std::string("sched_setaffinity failed: ") + strerror(errno);
        return false;
    }
    return true;
#elif defined(_WIN32)
    DWORD_PTR mask = 0;
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= 64) {
            err = "cpu id " + std::to_string(cpu) + " is out of range (max 63 on Windows)";
            return false;
        }
        mask |= (DWORD_PTR(1) << cpu);
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
        err = "SetProcessAffinityMask failed";
        return false;
    }
    return true;
#else
    (void) cpus;
    err = "cpu pinning is not supported on this platform";
    return false;
#endif
}

std::string format_cpu_list(const std::vector<int> & cpus) {
    if (cpus.empty()) return "";
    std::vector<int> sorted = cpus;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::string out;
    size_t i = 0;
    while (i < sorted.size()) {
        size_t j = i;
        while (j + 1 < sorted.size() && sorted[j + 1] == sorted[j] + 1) j++;
        if (!out.empty()) out += ",";
        out += std::to_string(sorted[i]);
        if (j != i) out += "-" + std::to_string(sorted[j]);
        i = j + 1;
    }
    return out;
}

}
