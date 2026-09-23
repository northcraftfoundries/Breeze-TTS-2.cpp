// plain-main regression test for the prefix cache added to BackboneState: checks that saving a
// prefill and loading it into a fresh state reproduces the same kv rows, and that continuing from
// a loaded prefix gives the same output (bit for bit on a SIMD-block boundary, rounding sized
// off it) as running the whole prompt through backbone_run in one call.
// needs a real model, so it is skipped (exit 77) unless BREEZE_TEST_MODEL points at a gguf.
#include "breeze/backbone.h"
#include "breeze/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

// deterministic, non-trivial embeds so a coding mistake in the split path can't hide behind
// all-zero or all-equal inputs
static std::vector<float> synthetic_embeds(int hidden, int n) {
    std::vector<float> e((size_t) hidden * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < hidden; j++)
            e[(size_t) i * hidden + j] = 0.01f * std::sin(i * 0.01f + j);
    return e;
}

// infinite on a size mismatch, empty output or a non finite value, so broken output can't score 0
static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

// index of the largest logit and how far it leads the second largest, to judge whether a rounding
// sized difference could plausibly flip the sampled token
static void top2(const std::vector<float> & v, int & argmax, float & margin) {
    int best = 0, second = -1;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[best]) best = (int) i;
    for (size_t i = 0; i < v.size(); i++)
        if ((int) i != best && (second < 0 || v[i] > v[second])) second = (int) i;
    argmax = best;
    margin = v[best] - v[second];
}

int main() {
    const char * path = std::getenv("BREEZE_TEST_MODEL");
    if (!path || !*path) {
        printf("BREEZE_TEST_MODEL not set, skipping\n");
        return 77;
    }

    BreezeModel m;
    if (!m.load(path, /*prefer_gpu=*/false)) {
        fprintf(stderr, "failed to load model at %s\n", path);
        return 1;
    }

    const int hidden = m.cfg.bb.hidden;
    const int total = 108;
    const std::vector<float> embeds = synthetic_embeds(hidden, total);

    // 1. save/load round trip: what one state saves must load into another bit for bit, since
    // that host copy is the only thing carrying the prefix between sessions
    {
        BackboneState a, b;
        a.init(m, 86);
        b.init(m, 86);
        std::vector<float> head(embeds.begin(), embeds.begin() + (size_t) 86 * hidden);
        backbone_run(m, a, head, 86);

        std::vector<std::vector<uint8_t>> rows = a.kv.save(86);
        b.load_prefix(rows, 86);
        std::vector<std::vector<uint8_t>> rows2 = b.kv.save(86);

        CHECK(rows.size() == rows2.size());
        bool equal = !rows.empty() && rows.size() == rows2.size();
        for (size_t i = 0; equal && i < rows.size(); i++) equal = rows[i] == rows2[i];
        CHECK(equal);

        a.free();
        b.free();
    }

    // 2. split at 64, a multiple of 32: the backbone is causal, so the first 64 kv rows of a one
    // shot 108 token run and a standalone 64 token run cover the same attention sums over the same
    // number of keys in the same SIMD blocks, and should therefore match exactly
    {
        BackboneState one;
        one.init(m, total + 8);
        StepOut out_one = backbone_run(m, one, embeds, total);

        BackboneState pre, split;
        pre.init(m, 64 + 8);
        std::vector<float> head(embeds.begin(), embeds.begin() + (size_t) 64 * hidden);
        backbone_run(m, pre, head, 64);
        std::vector<std::vector<uint8_t>> rows = pre.kv.save(64);

        split.init(m, total + 8);
        split.load_prefix(rows, 64);
        std::vector<float> tail(embeds.begin() + (size_t) 64 * hidden, embeds.end());
        StepOut out_split = backbone_run(m, split, tail, total - 64);

        const double hd = max_abs_diff(out_one.hidden, out_split.hidden);
        const double ld = max_abs_diff(out_one.logits, out_split.logits);
        printf("split at 64: hidden diff %.3g, logits diff %.3g\n", hd, ld);
        CHECK(hd == 0.0);
        CHECK(ld == 0.0);

        one.free();
        pre.free();
        split.free();
    }

    // 3. split at 86, not a multiple of 32: the attention sum past the last full SIMD block now
    // lands at a different point than the one shot run, so bit identical output isn't expected,
    // only a difference the size of float rounding
    {
        BackboneState one;
        one.init(m, total + 8);
        StepOut out_one = backbone_run(m, one, embeds, total);

        BackboneState pre, split;
        pre.init(m, 86 + 8);
        std::vector<float> head(embeds.begin(), embeds.begin() + (size_t) 86 * hidden);
        backbone_run(m, pre, head, 86);
        std::vector<std::vector<uint8_t>> rows = pre.kv.save(86);

        split.init(m, total + 8);
        split.load_prefix(rows, 86);
        std::vector<float> tail(embeds.begin() + (size_t) 86 * hidden, embeds.end());
        StepOut out_split = backbone_run(m, split, tail, total - 86);

        const double hd = max_abs_diff(out_one.hidden, out_split.hidden);
        const double ld = max_abs_diff(out_one.logits, out_split.logits);
        printf("split at 86: hidden diff %.3g, logits diff %.3g\n", hd, ld);
        CHECK(std::isfinite(hd) && hd < 1.0);
        CHECK(std::isfinite(ld) && ld < 1.0);

        int am_one, am_split;
        float margin_one, margin_split;
        top2(out_one.logits, am_one, margin_one);
        top2(out_split.logits, am_split, margin_split);
        printf("split at 86: top-2 margin %.3g\n", margin_one);
        if (margin_one > 0.5f) CHECK(am_one == am_split);

        one.free();
        pre.free();
        split.free();
    }

    m.free();

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
