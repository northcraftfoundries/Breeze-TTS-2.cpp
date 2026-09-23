#pragma once

#include "breeze/model.h"

#include <vector>

namespace breeze {

struct BackboneState {
    KVCache kv;
    int pos = 0;
    void init(BreezeModel & m, int max_seq);
    void reset() { kv.reset(); pos = 0; }
    void free() { kv.free(); }

    // loads a saved prefix into the cache and moves pos past it in one call, so a caller can't
    // load the rows and forget to advance pos to match
    void load_prefix(const std::vector<std::vector<uint8_t>> & rows, int n);
};

struct StepOut {
    std::vector<float> hidden; // [hidden_size]
    std::vector<float> logits; // [audio_vocab_size + 1]
};

// sum of the 16 codebook embeddings per frame; codes laid out frame-major [f*16 + cb]
std::vector<float> audio_embed_forward(BreezeModel & m, const std::vector<int> & codes, int n_frames);

// run a chunk of inputs_embeds through the backbone, appending to the kv cache
StepOut backbone_run(BreezeModel & m, BackboneState & st, const std::vector<float> & embeds, int n_tokens);

// one decode step feeding the same frame (num_codebooks codes) to every branch in a single graph,
// so the weights are read once however many branches there are. each branch keeps its own cache
std::vector<StepOut> backbone_step(BreezeModel & m, const std::vector<BackboneState *> & states,
                                   const std::vector<int> & frame);

}
