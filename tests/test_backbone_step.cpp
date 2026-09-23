// plain-main regression test for backbone_step: checks that batching cfg branches into one
// graph gives the same numbers as running each branch through backbone_run separately. both
// paths share the embedding row lookup and the layer code, so this catches batching mistakes
// (column slicing, per-branch caches and positions), not a wrong embedding offset.
// needs a real model, so it is skipped (exit 77) unless BREEZE_TEST_MODEL points at a gguf.
#include "breeze/backbone.h"
#include "breeze/model.h"
#include "test_util.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace breeze;

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
    // C and C2 cover the single branch path that every frame takes at cfg 1
    BackboneState A, B, C, A2, B2, C2;
    for (BackboneState * s : { &A, &B, &C, &A2, &B2, &C2 }) s->init(m, 64);

    const std::vector<float> embeds_a = synthetic_embeds(hidden, 5);
    const std::vector<float> embeds_b = synthetic_embeds(hidden, 9);
    backbone_run(m, A, embeds_a, 5);
    backbone_run(m, B, embeds_b, 9);
    backbone_run(m, A2, embeds_a, 5);
    backbone_run(m, B2, embeds_b, 9);
    backbone_run(m, C, embeds_b, 9);
    backbone_run(m, C2, embeds_b, 9);
    CHECK(A.pos == A2.pos);
    CHECK(B.pos == B2.pos);
    CHECK(C.pos == C2.pos);

    std::vector<BackboneState *> branches = { &A, &B };
    double max_hidden_diff = 0.0, max_logits_diff = 0.0;

    for (int f = 0; f < 3; f++) {
        std::vector<int> frame(nc);
        for (int cb = 0; cb < nc; cb++) frame[cb] = (17 * f + 3 * cb) % codebook_size;

        std::vector<StepOut> batched = backbone_step(m, branches, frame);
        std::vector<StepOut> single = backbone_step(m, { &C }, frame);
        CHECK(batched.size() == 2);
        CHECK(single.size() == 1);
        if (batched.size() != 2 || single.size() != 1) break;

        // the old, unbatched path: build the embedding once, then feed each branch its own step
        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        StepOut oa = backbone_run(m, A2, ae, 1);
        StepOut ob = backbone_run(m, B2, ae, 1);
        StepOut oc = backbone_run(m, C2, ae, 1);

        max_hidden_diff = std::max(max_hidden_diff, max_abs_diff(batched[0].hidden, oa.hidden));
        max_hidden_diff = std::max(max_hidden_diff, max_abs_diff(batched[1].hidden, ob.hidden));
        max_logits_diff = std::max(max_logits_diff, max_abs_diff(batched[0].logits, oa.logits));
        max_logits_diff = std::max(max_logits_diff, max_abs_diff(batched[1].logits, ob.logits));
        max_hidden_diff = std::max(max_hidden_diff, max_abs_diff(single[0].hidden, oc.hidden));
        max_logits_diff = std::max(max_logits_diff, max_abs_diff(single[0].logits, oc.logits));

        CHECK(A.pos == A2.pos);
        CHECK(B.pos == B2.pos);
        CHECK(C.pos == C2.pos);
    }

    printf("max hidden diff: %.3g, max logits diff: %.3g\n", max_hidden_diff, max_logits_diff);
    CHECK(max_hidden_diff <= 1e-5);
    CHECK(max_logits_diff <= 1e-5);

    for (BackboneState * s : { &A, &B, &C, &A2, &B2, &C2 }) s->free();
    m.free();

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
