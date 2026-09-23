// plain-main unit test for cpu_affinity.cpp; no test framework, none of this needs one.
#include "breeze/cpu_affinity.h"

#include <cstdio>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h> // CPU_SETSIZE, for the exact boundary tests below
#endif

using namespace breeze;

static int g_failures = 0;

// prints the failing expression and line rather than just "false", so a failure
// says what broke without needing a debugger
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, __LINE__); \
            g_failures++; \
        } \
    } while (0)

static void expect_parse_ok(const char * spec, const std::vector<int> & expected) {
    std::vector<int> cpus;
    std::string err;
    CHECK(parse_cpu_list(spec, cpus, err));
    CHECK(cpus == expected);
}

static void expect_parse_err(const char * spec) {
    std::vector<int> cpus;
    std::string err;
    CHECK(!parse_cpu_list(spec, cpus, err));
    CHECK(!err.empty());
}

int main() {
    expect_parse_ok("0-3", {0, 1, 2, 3});
    expect_parse_ok("3,1,1,0-1", {0, 1, 3});
    expect_parse_ok("0,2,4-7", {0, 2, 4, 5, 6, 7});

    expect_parse_err("");
    expect_parse_err(",");
    expect_parse_err("1,");
    expect_parse_err("a");
    expect_parse_err("-1"); // dash makes this a malformed range token, not a negative-id test
    expect_parse_err("3-1");
    expect_parse_err("1-");
    expect_parse_err("0-2147483647");
    expect_parse_err("99999999");

#ifdef __linux__
    // exact boundary at CPU_SETSIZE (1024 on glibc); derive it instead of hardcoding so this
    // still matches if glibc ever changes it
    expect_parse_ok(std::to_string(CPU_SETSIZE - 1).c_str(), {CPU_SETSIZE - 1});
    expect_parse_err(std::to_string(CPU_SETSIZE).c_str());
    expect_parse_ok((std::to_string(CPU_SETSIZE - 4) + "-" + std::to_string(CPU_SETSIZE - 1)).c_str(),
                     {CPU_SETSIZE - 4, CPU_SETSIZE - 3, CPU_SETSIZE - 2, CPU_SETSIZE - 1});
    expect_parse_err((std::to_string(CPU_SETSIZE - 4) + "-" + std::to_string(CPU_SETSIZE)).c_str());
#endif
#ifdef _WIN32
    // exact boundary at 63, the highest bit a DWORD_PTR affinity mask can hold
    expect_parse_ok("63", {63});
    expect_parse_err("64");
#endif

    CHECK(format_cpu_list({0, 1, 2, 3, 5, 7, 8}) == "0-3,5,7-8");
    CHECK(format_cpu_list({}) == "");
    CHECK(format_cpu_list({4}) == "4");
    CHECK(format_cpu_list({3, 1, 2, 2}) == "1-3"); // public function: don't assume sorted, deduped input

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
