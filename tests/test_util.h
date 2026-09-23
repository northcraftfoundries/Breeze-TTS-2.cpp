// shared bits of the plain-main tests. each test is its own executable, so the static counter
// is per test and the header needs no matching source file
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

// deterministic, non-trivial embeds so a coding mistake in the path under test can't hide behind
// all-zero or all-equal inputs
inline std::vector<float> synthetic_embeds(int hidden, int n) {
    std::vector<float> e((size_t) hidden * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < hidden; j++)
            e[(size_t) i * hidden + j] = 0.01f * std::sin(i * 0.01f + j);
    return e;
}

// infinite on a size mismatch, empty output or a non finite value, so broken output can't score 0
inline double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}
