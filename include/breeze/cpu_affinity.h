#pragma once

#include <string>
#include <vector>

namespace breeze {

// parses "0-15" or "0,2,4-7" style specs into sorted, de-duplicated cpu ids.
// rejects empty input, malformed tokens, negative ids, descending ranges and ids past the
// platform limit (checked before a range is expanded, so a huge range can't loop or allocate).
bool parse_cpu_list(const std::string & spec, std::vector<int> & cpus, std::string & err);

// logical cpu ids of the performance cores on a hybrid cpu, sorted.
// empty when the cpu isn't hybrid or the topology can't be read (e.g. under a VM).
std::vector<int> detect_pcores();

// pins to the given logical cpus. on linux that is the calling thread and every thread it
// starts afterwards, so call it from main before anything spawns; on windows the whole process.
bool pin_process(const std::vector<int> & cpus, std::string & err);

// parses spec (a cpu list like parse_cpu_list, or the literal "pcores" to auto detect the
// performance cores), pins the process to it, then reads back the cpus the kernel actually
// kept: it silently drops any outside the online/allowed set, so the requested list can
// overstate what pinning got. applied is what to report and to size the thread count from.
// fails if nothing is left pinned.
bool resolve_and_pin(const std::string & spec, std::vector<int> & applied, std::string & err);

// compact form like "0-7,16" for log lines.
std::string format_cpu_list(const std::vector<int> & cpus);

// number of cpus in the current process's affinity mask, 0 if it can't be read.
int process_cpu_count();

}
