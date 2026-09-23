#include "breeze/backbone.h"

#include <cmath>
#include <string>

namespace breeze {

void BackboneState::init(BreezeModel & m, int max_seq) {
    kv.init(m.backend, m.cfg.bb.n_layer, m.cfg.bb.head_dim, m.cfg.bb.n_kv_head, max_seq);
    pos = 0;
}

static ggml_tensor * build_audio_embed(ggml_context * ctx, BreezeModel & m, ggml_tensor * idx, int n) {
    const int nc = m.cfg.num_codebooks;
    const int hidden = m.cfg.hidden_size;
    // get_rows only indexes along one axis, so the frames stay flat until after the lookup
    ggml_tensor * rows = ggml_get_rows(ctx, m.w("audio_embd.weight"), idx);    // [hidden, nc*n]
    rows = ggml_reshape_3d(ctx, rows, hidden, nc, n);
    ggml_tensor * perm = ggml_cont(ctx, ggml_permute(ctx, rows, 1, 0, 2, 3));  // [nc, hidden, n]
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);                           // [1, hidden, n]
    return ggml_reshape_2d(ctx, summed, hidden, n);
}

// row per code in the shared audio_embd table: codebook cb's vocab occupies rows
// [cb*audio_vocab_size, (cb+1)*audio_vocab_size), and codes are frame-major [f*nc + cb]
static std::vector<int32_t> audio_embed_rows(BreezeModel & m, const std::vector<int> & codes, int n) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    GGML_ASSERT(codes.size() >= (size_t) nc * n);
    std::vector<int32_t> idx((size_t) nc * n);
    for (int f = 0; f < n; f++)
        for (int cb = 0; cb < nc; cb++) {
            const int code = codes[(size_t) f * nc + cb];
            // an out of range code would silently read the next codebook's rows
            GGML_ASSERT(code >= 0 && code < vs);
            idx[(size_t) f * nc + cb] = code + cb * vs;
        }
    return idx;
}

std::vector<float> audio_embed_forward(BreezeModel & m, const std::vector<int> & codes, int n) {
    const int nc = m.cfg.num_codebooks;
    Graph g(256);
    ggml_tensor * t = g.input_i32(audio_embed_rows(m, codes, n), nc * n);
    ggml_tensor * out = build_audio_embed(g.ctx, m, t, n);
    g.compute(m.backend, out);
    return tensor_to_f32(out);
}

// append this branch's k/v to its own cache and attend over everything it has seen
static ggml_tensor * bb_attend(ggml_context * ctx, BreezeModel & m, Graph & g, BackboneState & st, int il,
                               ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) {
    const BackboneConfig & c = m.cfg.bb;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    ggml_tensor * kfull = cache_append(ctx, g, st.kv.k[il], k, st.pos);
    ggml_tensor * vfull = cache_append(ctx, g, st.kv.v[il], v, st.pos);
    return attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
}

// one state takes all n tokens; several states take one token each, column b belonging to states[b]
static ggml_tensor * bb_layer(ggml_context * ctx, BreezeModel & m, Graph & g,
                              const std::vector<BackboneState *> & states, ggml_tensor * x, int il,
                              ggml_tensor * pos, const std::vector<ggml_tensor *> & masks, int n) {
    const BackboneConfig & c = m.cfg.bb;
    const std::string p = "bb.blk." + std::to_string(il);
    // one state may take any number of tokens, but several states take exactly one each
    GGML_ASSERT(states.size() == 1 || n == (int) states.size());

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);

    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = rms_norm(ctx, q, m.w(p + ".attn_q_norm.weight"), c.rms_eps);
    k = rms_norm(ctx, k, m.w(p + ".attn_k_norm.weight"), c.rms_eps);
    q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * a = nullptr;
    if (states.size() == 1) {
        a = bb_attend(ctx, m, g, *states[0], il, q, k, v, masks[0]);
    } else {
        // the projections above ran on every branch at once; attention can't, each cache is its own length
        for (size_t b = 0; b < states.size(); b++) {
            ggml_tensor * qb = ggml_view_3d(ctx, q, c.head_dim, c.n_head, 1, q->nb[1], q->nb[2], b * q->nb[2]);
            ggml_tensor * kb = ggml_view_3d(ctx, k, c.head_dim, c.n_kv_head, 1, k->nb[1], k->nb[2], b * k->nb[2]);
            ggml_tensor * vb = ggml_view_3d(ctx, v, c.head_dim, c.n_kv_head, 1, v->nb[1], v->nb[2], b * v->nb[2]);
            ggml_tensor * ab = bb_attend(ctx, m, g, *states[b], il, qb, kb, vb, masks[b]);
            a = a ? ggml_concat(ctx, a, ab, 1) : ab;
        }
    }
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

StepOut backbone_run(BreezeModel & m, BackboneState & st, const std::vector<float> & embeds, int n) {
    const BackboneConfig & c = m.cfg.bb;
    const int total = st.pos + n;
    Graph g(8192);

    ggml_tensor * x = g.input_f32(embeds, c.hidden, n);
    std::vector<int32_t> pos_i(n);
    for (int i = 0; i < n; i++) pos_i[i] = st.pos + i;
    ggml_tensor * pos = g.input_i32(pos_i, n);
    std::vector<float> mask_v = build_causal_mask(n, total, st.pos, 0);
    ggml_tensor * mask = g.input_f32(mask_v, total, n);

    for (int il = 0; il < c.n_layer; il++) x = bb_layer(g.ctx, m, g, { &st }, x, il, pos, { mask }, n);
    x = rms_norm(g.ctx, x, m.w("bb.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_view_2d(g.ctx, x, c.hidden, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    last = ggml_cont(g.ctx, last);
    ggml_tensor * logits = linear(g.ctx, m.w("bb.lm_head.weight"), last);

    ggml_set_output(last);
    g.write(last);
    g.compute(m.backend, logits);

    StepOut out;
    out.hidden = tensor_to_f32(last);
    out.logits = tensor_to_f32(logits);
    st.pos += n;
    return out;
}

std::vector<StepOut> backbone_step(BreezeModel & m, const std::vector<BackboneState *> & states,
                                   const std::vector<int> & frame) {
    const BackboneConfig & c = m.cfg.bb;
    const int nc = m.cfg.num_codebooks;
    const int nb = (int) states.size();
    GGML_ASSERT(nb >= 1);
    // the frame embedding is built at the shared hidden size and fed straight into the backbone
    GGML_ASSERT(m.cfg.hidden_size == c.hidden);
    // the shared layer stack fits in 8192 nodes; every branch adds its own cache append and
    // attention, about 20 nodes per layer
    Graph g(8192 + 32 * c.n_layer * nb);

    ggml_tensor * x = build_audio_embed(g.ctx, m, g.input_i32(audio_embed_rows(m, frame, 1), nc), 1);
    // every branch is fed the same frame
    x = ggml_repeat_4d(g.ctx, x, c.hidden, nb, 1, 1);

    std::vector<int32_t> pos_i(nb);
    std::vector<ggml_tensor *> masks(nb);
    for (int b = 0; b < nb; b++) {
        const int p = states[b]->pos;
        pos_i[b] = p;
        masks[b] = g.input_f32(build_causal_mask(1, p + 1, p, 0), p + 1, 1);
    }
    ggml_tensor * pos = g.input_i32(pos_i, nb);

    for (int il = 0; il < c.n_layer; il++) x = bb_layer(g.ctx, m, g, states, x, il, pos, masks, nb);
    x = rms_norm(g.ctx, x, m.w("bb.output_norm.weight"), c.rms_eps);
    ggml_tensor * logits = linear(g.ctx, m.w("bb.lm_head.weight"), x);

    ggml_set_output(x);
    g.write(x);
    g.compute(m.backend, logits);

    const std::vector<float> hidden = tensor_to_f32(x);
    const std::vector<float> all_logits = tensor_to_f32(logits);
    const size_t n_logits = all_logits.size() / nb;
    std::vector<StepOut> out(nb);
    for (int b = 0; b < nb; b++) {
        out[b].hidden.assign(hidden.begin() + (size_t) b * c.hidden, hidden.begin() + (size_t) (b + 1) * c.hidden);
        out[b].logits.assign(all_logits.begin() + b * n_logits, all_logits.begin() + (b + 1) * n_logits);
        states[b]->pos++;
    }
    return out;
}

}
