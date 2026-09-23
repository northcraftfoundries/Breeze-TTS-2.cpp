// plain-main regression test for the prefix cache added to BackboneState: checks that saving a
// prefill and loading it into a fresh state reproduces the same kv rows, and that continuing from
// a loaded prefix gives the same output (bit for bit on a SIMD-block boundary, rounding sized
// off it) as running the whole prompt through backbone_run in one call.
// needs a real model, so it is skipped (exit 77) unless BREEZE_TEST_MODEL points at a gguf.
#include "breeze/backbone.h"
#include "breeze/model.h"
#include "test_util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace breeze;

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

// prefills the first `at` embeds on their own, loads that prefix into a fresh state and runs the
// rest on top, then compares against one run over the whole prompt. tol is the largest hidden or
// logits difference allowed, 0 demanding a bit identical result
static void check_split(BreezeModel & m, const std::vector<float> & embeds, int total, int at,
                        double tol) {
    const int hidden = m.cfg.bb.hidden;

    BackboneState one;
    one.init(m, total + 8);
    StepOut out_one = backbone_run(m, one, embeds, total);

    BackboneState pre, split;
    pre.init(m, at + 8);
    std::vector<float> head(embeds.begin(), embeds.begin() + (size_t) at * hidden);
    backbone_run(m, pre, head, at);
    std::vector<std::vector<uint8_t>> rows = pre.kv.save(at);

    split.init(m, total + 8);
    split.load_prefix(rows, at);
    CHECK(split.pos == at);
    std::vector<float> tail(embeds.begin() + (size_t) at * hidden, embeds.end());
    StepOut out_split = backbone_run(m, split, tail, total - at);

    const double hd = max_abs_diff(out_one.hidden, out_split.hidden);
    const double ld = max_abs_diff(out_one.logits, out_split.logits);
    printf("split at %d: hidden diff %.3g, logits diff %.3g\n", at, hd, ld);
    // max_abs_diff is infinite on broken output, so this also fails on a size mismatch or nan
    CHECK(hd <= tol);
    CHECK(ld <= tol);

    int am_one, am_split;
    float margin_one, margin_split;
    top2(out_one.logits, am_one, margin_one);
    top2(out_split.logits, am_split, margin_split);
    printf("split at %d: top-2 margin %.3g\n", at, margin_one);
    // a rounding sized difference can legitimately swap two near tied logits
    if (margin_one > 0.5f)
        CHECK(am_one == am_split);
    else
        printf("split at %d: top-2 margin too small, argmax check skipped\n", at);

    one.free();
    pre.free();
    split.free();
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

    // 2. split at 64, a multiple of the F32 SIMD step (32 floats with AVX2, 64 with AVX-512): the
    // backbone is causal, so the first 64 kv rows of a one shot 108 token run and a standalone 64
    // token run cover the same attention sums over the same number of keys in the same SIMD
    // blocks, and should therefore match exactly
    check_split(m, embeds, total, 64, 0.0);

    // 3. split at 86, not a multiple of the SIMD step: the attention sum past the last full SIMD
    // block now lands at a different point than the one shot run, so bit identical output isn't
    // expected, only a difference the size of float rounding. measured 0.0317 on hidden and
    // 0.0731 on logits, so 0.25 leaves headroom without letting a real mistake through
    check_split(m, embeds, total, 86, 0.25);

    m.free();

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
