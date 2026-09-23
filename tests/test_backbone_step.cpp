// plain-main regression test for backbone_step: checks that batching cfg branches into one
// graph gives the same numbers as running each branch through backbone_run separately.
// needs a real model, so it is skipped (exit 77) unless BREEZE_TEST_MODEL points at a gguf.
#include "breeze/backbone.h"
#include "breeze/model.h"

#include <algorithm>
#include <cmath>
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

// deterministic, non-trivial embeds so a coding mistake in the batched path can't hide behind
// all-zero or all-equal inputs
static std::vector<float> synthetic_embeds(int hidden, int n) {
    std::vector<float> e((size_t) hidden * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < hidden; j++)
            e[(size_t) i * hidden + j] = 0.01f * std::sin(i * 0.01f + j);
    return e;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) d = std::max(d, (double) std::fabs(a[i] - b[i]));
    return d;
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
    const int nc = m.cfg.num_codebooks;
    const int codebook_size = m.cfg.codec_codebook_size;

    // pair 1 goes through backbone_step batched; pair 2 runs the pre-batching path, branch by
    // branch, starting from an identical prefill so the two pairs stay comparable
    BackboneState A, B, A2, B2;
    A.init(m, 64);
    B.init(m, 64);
    A2.init(m, 64);
    B2.init(m, 64);

    const std::vector<float> embeds_a = synthetic_embeds(hidden, 5);
    const std::vector<float> embeds_b = synthetic_embeds(hidden, 9);
    backbone_run(m, A, embeds_a, 5);
    backbone_run(m, B, embeds_b, 9);
    backbone_run(m, A2, embeds_a, 5);
    backbone_run(m, B2, embeds_b, 9);
    CHECK(A.pos == A2.pos);
    CHECK(B.pos == B2.pos);

    std::vector<BackboneState *> branches = { &A, &B };
    double max_hidden_diff = 0.0, max_logits_diff = 0.0;

    for (int f = 0; f < 3; f++) {
        std::vector<int> frame(nc);
        for (int cb = 0; cb < nc; cb++) frame[cb] = (17 * f + 3 * cb) % codebook_size;

        std::vector<StepOut> batched = backbone_step(m, branches, frame);

        // the old, unbatched path: build the embedding once, then feed each branch its own step
        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        StepOut oa = backbone_run(m, A2, ae, 1);
        StepOut ob = backbone_run(m, B2, ae, 1);

        max_hidden_diff = std::max(max_hidden_diff, max_abs_diff(batched[0].hidden, oa.hidden));
        max_hidden_diff = std::max(max_hidden_diff, max_abs_diff(batched[1].hidden, ob.hidden));
        max_logits_diff = std::max(max_logits_diff, max_abs_diff(batched[0].logits, oa.logits));
        max_logits_diff = std::max(max_logits_diff, max_abs_diff(batched[1].logits, ob.logits));

        CHECK(A.pos == A2.pos);
        CHECK(B.pos == B2.pos);
    }

    printf("max hidden diff: %.3g, max logits diff: %.3g\n", max_hidden_diff, max_logits_diff);
    CHECK(max_hidden_diff <= 1e-5);
    CHECK(max_logits_diff <= 1e-5);

    A.free();
    B.free();
    A2.free();
    B2.free();
    m.free();

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
