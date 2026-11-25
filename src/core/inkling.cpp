#include "inkling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

#include "quant.h"
#include "threads.h"

namespace gfx {

bool is_inkling(const GGUF &g) { return g.architecture == "inkling2" || g.architecture == "inkling2_loop"; }

namespace {

// A metadata value that is either one number or one number per layer.
double number_at(const GGUF &g, const std::string &suffix, int layer, bool required = true, double def = 0) {
    const MetaValue *v = g.find(g.arch_key(suffix));
    if (!v) {
        if (required) throw std::runtime_error("model is missing metadata " + g.arch_key(suffix));
        return def;
    }
    if (v->type != M_ARR) return v->is_int() ? (double)v->as_int() : v->as_float();
    if ((uint64_t)layer >= v->count) throw std::runtime_error("metadata " + g.arch_key(suffix) + " has too few entries");
    const bool integral = v->elem_type <= M_I32 || v->elem_type == M_BOOL || v->elem_type == M_U64 || v->elem_type == M_I64;
    return integral ? (double)v->arr_int(layer) : v->arr_float(layer);
}

// The same for all layers, or this implementation does not apply.
double uniform(const GGUF &g, const std::string &suffix, int n_layer) {
    double first = number_at(g, suffix, 0);
    for (int i = 1; i < n_layer; i++)
        if (number_at(g, suffix, i) != first) throw std::runtime_error("layers with different " + suffix + " are not supported");
    return first;
}

inline float dot(const float *a, const float *b, size_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; k++) acc[k] += a[i + k] * b[i + k];
    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// x / sqrt(mean(x^2) + eps), optionally times a weight
void rms_norm(const float *x, const float *w, float *y, int n, float eps) {
    float ss = dot(x, x, n);
    float s = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * (w ? w[i] : 1.0f);
}

// Scale a vector down to an L2 norm of at most `limit`, after an optional per-element scale.
void clip(float *x, const float *scale, int n, float limit) {
    if (scale) for (int i = 0; i < n; i++) x[i] *= scale[i];
    float norm = std::sqrt(dot(x, x, n)) + 1e-6f;
    float f = std::min(std::max(limit / norm, 0.0f), 1.0f);
    for (int i = 0; i < n; i++) x[i] *= f;
}

}  // namespace

InklingDesc describe_inkling(const GGUF &g) {
    if (!is_inkling(g)) throw std::runtime_error("not an Inkling 2 model");
    InklingDesc d;
    d.name = g.get_str("general.name");
    auto key = [&](const char *s) { return g.arch_key(s); };
    d.n_layer = (int)g.get_int(key("block_count"), 0);
    d.n_embd = (int)g.get_int(key("embedding_length"), 0);
    d.n_ctx_train = (int)g.get_int(key("context_length"), 0);
    if (d.n_layer <= 0 || d.n_embd <= 0) throw std::runtime_error("model metadata lacks the layer count or embedding size");
    d.n_head = (int)uniform(g, "attention.head_count", d.n_layer);
    d.n_head_kv = (int)uniform(g, "attention.head_count_kv", d.n_layer);
    d.head_dim = (int)g.get_int(key("attention.key_length"), 0);
    d.norm_eps = (float)g.get_float(key("attention.layer_norm_rms_epsilon"), 1e-6);
    d.qk_clip = (float)number_at(g, "attention.qk_clip", 0);
    d.rel_clip = (float)number_at(g, "attention.relative_clip", 0);
    d.n_rel = (int)number_at(g, "attention.relative_length", 0);
    d.rel_extent = (int)number_at(g, "attention.relative_extent", 0);
    d.window = (int)number_at(g, "attention.sliding_window", 0);
    d.n_conv = (int)number_at(g, "shortconv.l_cache", 0);
    d.block = (int)number_at(g, "attn_res.block_size", 0);
    d.log_floor = (int)number_at(g, "attention.log_scale_floor", 0);
    d.log_alpha = (float)number_at(g, "attention.log_scale_alpha", 0);
    d.n_expert = (int)number_at(g, "expert_count", 0);
    d.n_expert_used = (int)uniform(g, "expert_used_count", d.n_layer);
    d.n_ff = (int)uniform(g, "expert_feed_forward_length", d.n_layer);
    d.expert_scale = (float)number_at(g, "expert_weights_scale", 0, false, 1.0);
    d.situ_beta = (float)number_at(g, "activation.situ_beta", 0);
    d.situ_linear_beta = (float)number_at(g, "activation.situ_linear_beta", 0);
    d.softcap = (float)number_at(g, "final_logit_softcapping", 0, false, 0.0);
    d.n_vocab_padded = (int)number_at(g, "vocab_size", 0);
    d.n_vocab = (int)number_at(g, "vocab_size_unpadded", 0);
    if (g.architecture == "inkling2_loop") {
        d.loop_start = (int)number_at(g, "loop.start", 0);
        d.loop_end = (int)number_at(g, "loop.end", 0);
        d.loop_count = (int)number_at(g, "loop.count", 0);
        d.gate_rank = (int)number_at(g, "loop.gate_rank", 0);
        if (number_at(g, "loop.varying", 0, false, 0) != 0) throw std::runtime_error("models with a variable number of loop passes are not supported");
        if (d.loop_count < 2 || d.loop_start <= 0 || d.loop_start >= d.loop_end || d.loop_end > d.n_layer || d.loop_start % d.block || d.loop_end % d.block)
            throw std::runtime_error("unexpected loop layout");
    }
    if (d.head_dim <= 0 || d.n_head % d.n_head_kv || d.n_conv < 2 || d.block <= 0 || d.n_expert_used > d.n_expert || d.n_expert_used <= 0)
        throw std::runtime_error("unexpected attention or expert dimensions");

    auto get = [&](const std::string &name, bool required = true) {
        const TensorInfo *t = g.tensor(name);
        if (!t && required) throw std::runtime_error("model is missing tensor " + name);
        return t;
    };
    auto shape = [](const TensorInfo *t, uint64_t d0, uint64_t d1, uint64_t d2 = 1) {
        if (t && (t->dim(0) != d0 || t->dim(1) != d1 || t->dim(2) != d2))
            throw std::runtime_error("tensor " + t->name + " has an unexpected shape");
    };
    const uint64_t E = d.n_embd, HQ = (uint64_t)d.n_head * d.head_dim, HK = (uint64_t)d.n_head_kv * d.head_dim;
    d.tok_embd = get("token_embd.weight"); d.tok_norm = get("token_embd_norm.weight");
    d.output = get("output.weight"); d.output_norm = get("output_norm.weight"); d.output_query = get("output_res_query.weight");
    shape(d.tok_embd, E, d.n_vocab_padded); shape(d.output, E, d.n_vocab_padded);
    d.layers.resize(d.n_layer);
    for (int i = 0; i < d.n_layer; i++) {
        InklingLayer &l = d.layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        l.window = number_at(g, "attention.sliding_window_pattern", i) != 0;
        l.wq = get(p + "attn_q.weight"); l.wk = get(p + "attn_k.weight"); l.wv = get(p + "attn_v.weight"); l.wo = get(p + "attn_output.weight");
        l.wgate = get(p + "attn_gate.weight"); l.q_clip = get(p + "attn_q_clip.weight"); l.sink = get(p + "attn_sink_proj.weight");
        l.rel = get(p + "attn_rel.weight"); l.rel_bias = get(p + "attn_rel_bias.weight"); l.rel_clip = get(p + "attn_rel_clip.weight");
        l.k_conv = get(p + "attn_k_conv.weight"); l.v_conv = get(p + "attn_v_conv.weight");
        l.attn_conv = get(p + "attn_post_conv.weight"); l.ffn_conv = get(p + "ffn_post_conv.weight");
        if (l.window) l.attn_norm = get(p + "attn_norm.weight");
        else {
            l.q_norm = get(p + "attn_q_inp_norm.weight"); l.k_norm = get(p + "attn_k_inp_norm.weight"); l.v_norm = get(p + "attn_v_inp_norm.weight");
            l.k_taps = get(p + "attn_k_conv_proj.weight"); l.v_taps = get(p + "attn_v_conv_proj.weight");
            shape(l.k_taps, E, (uint64_t)d.n_conv * d.n_head_kv); shape(l.v_taps, E, (uint64_t)d.n_conv * d.n_head_kv);
        }
        if (i > 0) { l.attn_query = get(p + "attn_res_query.weight"); shape(l.attn_query, E, l.window ? 1 : 3); }
        l.ffn_query = get(p + "ffn_res_query.weight");
        l.ffn_norm = get(p + "ffn_norm.weight"); l.router = get(p + "ffn_gate_inp.weight"); l.router_bias = get(p + "exp_probs_b.bias", false);
        l.gate_up = get(p + "ffn_gate_up_exps.weight"); l.gate_up_scale = get(p + "ffn_gate_up_exps.scale", false);
        l.down = get(p + "ffn_down_exps.weight"); l.down_scale = get(p + "ffn_down_exps.scale", false);
        shape(l.wq, E, HQ); shape(l.wk, E, HK); shape(l.wv, E, HK); shape(l.wo, HQ, E); shape(l.wgate, E, HQ);
        shape(l.q_clip, d.head_dim, d.n_head); shape(l.sink, E, d.n_head); shape(l.rel, E, (uint64_t)d.n_rel * d.n_head);
        shape(l.rel_bias, d.n_rel, l.window ? d.window : d.rel_extent); shape(l.rel_clip, d.n_rel, d.n_head);
        shape(l.k_conv, d.n_conv, HK); shape(l.v_conv, d.n_conv, HK); shape(l.attn_conv, d.n_conv, E); shape(l.ffn_conv, d.n_conv, E);
        shape(l.router, E, d.n_expert); shape(l.gate_up, E, 2ull * d.n_ff, d.n_expert); shape(l.down, d.n_ff, E, d.n_expert);
        if (i >= d.loop_start && i < d.loop_end && d.loop_count > 1) {
            static const char *who[4] = {"q", "k", "v", "ffn"};
            l.msg_query = get(p + "loop_msg_query.weight"); shape(l.msg_query, E, 4);
            for (int c = 0; c < 4; c++) {
                const std::string q = p + "loop_" + who[c] + "_ctrl_";
                l.ctrl_down[c] = get(q + "down.weight"); l.ctrl_gain[c] = get(q + "gain.weight"); l.ctrl_forget[c] = get(q + "forget.weight");
                shape(l.ctrl_down[c], 2 * E, d.gate_rank); shape(l.ctrl_gain[c], d.gate_rank, E); shape(l.ctrl_forget[c], d.gate_rank, E);
            }
        }
    }
    return d;
}

InklingRef::InklingRef(const GGUF &g, const InklingDesc &desc, int n_ctx, int n_threads) : g_(g), d_(desc), n_ctx_(n_ctx) {
    pool_ = std::make_unique<ThreadPool>(n_threads < 1 ? 1 : n_threads);
    const size_t kv = (size_t)n_ctx_ * d_.n_head_kv * d_.head_dim;
    k_cache_.assign(d_.n_slots(), Vec(kv)); v_cache_.assign(d_.n_slots(), Vec(kv));
    conv_.assign(d_.n_slots(), Vec((size_t)d_.conv_channels() * (d_.n_conv - 1)));
    experts_.resize(d_.n_slots());
    logits_.resize(d_.n_vocab);
}
InklingRef::~InklingRef() = default;

void InklingRef::reset() {
    n_past_ = 0;
    for (Vec &c : conv_) std::fill(c.begin(), c.end(), 0.0f);
}

const float *InklingRef::full(const TensorInfo *t) {
    auto it = decoded_.find(t);
    if (it != decoded_.end()) return it->second.data();
    Vec v(t->nelems);
    if (!dequantize(t->type, t->data, v.data(), t->nelems)) throw std::runtime_error("no decoder for tensor " + t->name);
    return decoded_.emplace(t, std::move(v)).first->second.data();
}

// y = scale * W[row0 .. row0 + rows) x, decoding the rows as it goes.
void InklingRef::matvec(const TensorInfo *w, const float *x, float *y, uint64_t row0, uint64_t rows, float scale) {
    const uint64_t n_in = w->dim(0);
    if (rows == 0) rows = w->nelems / n_in;
    const TypeInfo *ti = type_info(w->type);
    if (!ti || n_in % ti->block_elems) throw std::runtime_error("tensor " + w->name + ": rows are not block aligned");
    const uint64_t row_bytes = n_in / ti->block_elems * ti->block_bytes, chunk = 32;
    std::atomic<bool> failed{false};
    pool_->run((rows + chunk - 1) / chunk, [&](size_t c) {
        thread_local std::vector<float> buf;
        const uint64_t r0 = c * chunk, n = std::min(chunk, rows - r0);
        buf.resize(n * n_in);
        if (!dequantize(w->type, w->data + (row0 + r0) * row_bytes, buf.data(), n * n_in)) { failed = true; return; }
        for (uint64_t r = 0; r < n; r++) y[r0 + r] = scale * dot(buf.data() + r * n_in, x, n_in);
    });
    if (failed) throw std::runtime_error("no decoder for tensor " + w->name);
}

// Attention over the depth: the block inputs so far and the current stream, each normalised, are
// combined with weights from a learned query. One result per query.
InklingRef::Vec InklingRef::mix(const std::vector<Vec> &blocks, const Vec &cur, const TensorInfo *query, int n_query) {
    const int n = d_.n_embd, S = (int)blocks.size() + 1;
    std::vector<Vec> states(S, Vec(n));
    for (int s = 0; s < S; s++) rms_norm(s + 1 < S ? blocks[s].data() : cur.data(), nullptr, states[s].data(), n, 1e-6f);
    const float *q = full(query);
    Vec out((size_t)n_query * n, 0.0f);
    std::vector<float> p(S);
    for (int k = 0; k < n_query; k++) {
        float mx = -INFINITY, sum = 0;
        for (int s = 0; s < S; s++) { p[s] = dot(states[s].data(), q + (size_t)k * n, n) / std::sqrt((float)n); mx = std::max(mx, p[s]); }
        for (int s = 0; s < S; s++) { p[s] = std::exp(p[s] - mx); sum += p[s]; }
        for (int s = 0; s < S; s++) { const float w = p[s] / sum; for (int i = 0; i < n; i++) out[(size_t)k * n + i] += w * states[s][i]; }
    }
    return out;
}

// Gate a message from the previous pass into a layer input: (1 - forget) * cur + gain * message.
InklingRef::Vec InklingRef::control(const Vec &cur, const float *message, const InklingLayer &l, int c) {
    const int n = d_.n_embd, r = d_.gate_rank;
    Vec both(2 * (size_t)n), low(r), gain(n), forget(n), out(n);
    rms_norm(cur.data(), nullptr, both.data(), n, 1e-6f);
    rms_norm(message, nullptr, both.data() + n, n, 1e-6f);
    matvec(l.ctrl_down[c], both.data(), low.data());
    matvec(l.ctrl_gain[c], low.data(), gain.data());
    matvec(l.ctrl_forget[c], low.data(), forget.data());
    for (int i = 0; i < n; i++) out[i] = cur[i] - std::tanh(forget[i]) * cur[i] + std::tanh(gain[i]) * message[i];
    return out;
}

// Causal convolution over the last n_conv values of each channel, with a residual. `taps` adds
// input-dependent taps shared by the channels of a head. The earlier values live in the slot's
// state at channel offset `offset`.
void InklingRef::convolve(float *cur, int n, const TensorInfo *weight, const float *taps, int n_heads, int slot, int offset) {
    const int nc = d_.n_conv, per_head = n / n_heads;
    const float *w = full(weight);
    float *hist = conv_[slot].data() + (size_t)offset * (nc - 1);
    const float norm = 1.0f / std::sqrt((float)((taps ? 2 : 1) * nc + 1));
    float x[64];
    for (int c = 0; c < n; c++) {
        float *h = hist + (size_t)c * (nc - 1);
        for (int i = 0; i + 1 < nc; i++) x[i] = h[i];
        x[nc - 1] = cur[c];
        float out = 0;
        for (int i = 0; i < nc; i++) out += x[i] * w[(size_t)c * nc + i];
        if (taps) for (int lag = 0; lag < nc; lag++) out += x[nc - 1 - lag] * taps[(c / per_head) * nc + lag];
        for (int i = 0; i + 1 < nc; i++) h[i] = x[i + 1];
        cur[c] = (cur[c] + out) * norm;
    }
}

InklingRef::Vec InklingRef::layer(int slot, int ip, const Vec &in, std::vector<Vec> &blocks, const Vec *messages) {
    const InklingLayer &l = d_.layers[ip];
    const int n = d_.n_embd, hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, nr = d_.n_rel, pos = n_past_;
    const int G = nh / nkv;

    // Inputs of the query, key and value projections.
    Vec inp[3] = {in, in, in};
    if (l.attn_query) {
        Vec mixed = mix(blocks, in, l.attn_query, l.window ? 1 : 3);
        for (int j = 0; j < 3; j++) inp[j].assign(mixed.begin() + (l.window ? 0 : j) * n, mixed.begin() + (l.window ? 0 : j) * n + n);
    }
    if (messages) for (int j = 0; j < 3; j++) inp[j] = control(inp[j], messages->data() + (size_t)j * n, l, j);
    if (l.window) {
        const float *w = full(l.attn_norm);
        rms_norm(inp[0].data(), w, inp[0].data(), n, d_.norm_eps);
        if (messages) { rms_norm(inp[1].data(), w, inp[1].data(), n, d_.norm_eps); rms_norm(inp[2].data(), w, inp[2].data(), n, d_.norm_eps); }
        else inp[1] = inp[2] = inp[0];
    } else {
        rms_norm(inp[0].data(), full(l.q_norm), inp[0].data(), n, d_.norm_eps);
        rms_norm(inp[1].data(), full(l.k_norm), inp[1].data(), n, d_.norm_eps);
        rms_norm(inp[2].data(), full(l.v_norm), inp[2].data(), n, d_.norm_eps);
    }

    Vec q((size_t)nh * hd), k((size_t)nkv * hd), v((size_t)nkv * hd), r((size_t)nh * nr), sinks(nh), gate((size_t)nh * hd);
    matvec(l.wq, inp[0].data(), q.data());
    matvec(l.wk, inp[1].data(), k.data());
    matvec(l.wv, inp[2].data(), v.data());
    matvec(l.rel, inp[0].data(), r.data());
    matvec(l.sink, inp[0].data(), sinks.data());
    matvec(l.wgate, inp[0].data(), gate.data());
    Vec ktaps, vtaps;
    if (!l.window) {
        ktaps.resize((size_t)d_.n_conv * nkv); vtaps.resize((size_t)d_.n_conv * nkv);
        matvec(l.k_taps, inp[1].data(), ktaps.data());
        matvec(l.v_taps, inp[2].data(), vtaps.data());
    }
    convolve(k.data(), nkv * hd, l.k_conv, l.window ? nullptr : ktaps.data(), nkv, slot, 0);
    convolve(v.data(), nkv * hd, l.v_conv, l.window ? nullptr : vtaps.data(), nkv, slot, nkv * hd);

    const float *q_clip = full(l.q_clip), *rel_clip = full(l.rel_clip);
    for (int h = 0; h < nh; h++) clip(q.data() + (size_t)h * hd, q_clip + (size_t)h * hd, hd, d_.qk_clip);
    for (int h = 0; h < nkv; h++) clip(k.data() + (size_t)h * hd, nullptr, hd, d_.qk_clip);
    for (int h = 0; h < nkv; h++) rms_norm(v.data() + (size_t)h * hd, nullptr, v.data() + (size_t)h * hd, hd, d_.norm_eps);
    for (int h = 0; h < nh; h++) clip(r.data() + (size_t)h * nr, rel_clip + (size_t)h * nr, nr, d_.rel_clip);
    if (!l.window) { for (float &x : q) x *= tau_; for (float &x : r) x *= tau_; }

    // Position bias by distance, for each head; distances past the table get none.
    const int extent = (int)l.rel_bias->dim(1);
    const float *bias = full(l.rel_bias);
    const int first = l.window ? std::max(0, pos - d_.window + 1) : 0, span = std::min(pos - first + 1, extent);
    Vec table((size_t)nh * span);
    for (int h = 0; h < nh; h++)
        for (int e = 0; e < span; e++) table[(size_t)h * span + e] = dot(bias + (size_t)e * nr, r.data() + (size_t)h * nr, nr);

    float *kc = k_cache_[slot].data(), *vc = v_cache_[slot].data();
    const size_t row = (size_t)nkv * hd;
    std::copy(k.begin(), k.end(), kc + (size_t)pos * row);
    std::copy(v.begin(), v.end(), vc + (size_t)pos * row);

    // Attention. The scaled scores get the position bias and lose the query's sink logit (neither
    // is scaled); a sink with logit zero takes part in the softmax and contributes nothing.
    Vec att((size_t)nh * hd, 0.0f);
    const float scale = 1.0f / std::sqrt((float)hd);
    static const bool debug = getenv("GFX_INK_DEBUG") != nullptr;
    std::vector<float> dbg_sink(nh), dbg_self(nh), dbg_prev(nh);
    pool_->run(nh, [&](size_t h) {
        const int kvh = (int)h / G, count = pos - first + 1;
        std::vector<float> s(count);
        float mx = 0.0f;   // the sink's logit
        for (int j = first; j <= pos; j++) {
            const int dist = pos - j;
            const float b = dist < span ? table[h * span + dist] : 0.0f;
            s[j - first] = dot(q.data() + h * hd, kc + (size_t)j * row + (size_t)kvh * hd, hd) * scale + b - sinks[h];
            mx = std::max(mx, s[j - first]);
        }
        float sum = std::exp(0.0f - mx);
        for (float &x : s) { x = std::exp(x - mx); sum += x; }
        float *o = att.data() + h * hd;
        dbg_sink[h] = std::exp(0.0f - mx) / sum; dbg_self[h] = s[count - 1] / sum; dbg_prev[h] = count > 1 ? s[count - 2] / sum : 0.0f;
        for (int j = first; j <= pos; j++) {
            const float p = s[j - first] / sum;
            const float *vj = vc + (size_t)j * row + (size_t)kvh * hd;
            for (int i = 0; i < hd; i++) o[i] += p * vj[i];
        }
    });
    for (size_t i = 0; i < att.size(); i++) att[i] *= sigmoid(gate[i]);
    Vec a(n);
    matvec(l.wo, att.data(), a.data());
    convolve(a.data(), n, l.attn_conv, nullptr, 1, slot, 2 * nkv * hd);

    // The first layer of a block starts a new residual stream; its input joins the mixed streams.
    Vec cur(n);
    if (ip % d_.block == 0) { blocks.push_back(in); cur = a; }
    else for (int i = 0; i < n; i++) cur[i] = in[i] + a[i];

    // Mixture of experts.
    Vec f = mix(blocks, cur, l.ffn_query, 1);
    if (messages) f = control(f, messages->data() + 3 * (size_t)n, l, 3);
    rms_norm(f.data(), full(l.ffn_norm), f.data(), n, d_.norm_eps);
    const int ne = d_.n_expert, used = d_.n_expert_used, nff = d_.n_ff;
    Vec probs(ne);
    matvec(l.router, f.data(), probs.data());
    const float *rb = l.router_bias ? full(l.router_bias) : nullptr;
    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    Vec pick(ne);
    for (int e = 0; e < ne; e++) { probs[e] = sigmoid(probs[e]); pick[e] = probs[e] + (rb ? rb[e] : 0.0f); }
    std::partial_sort(order.begin(), order.begin() + used, order.end(), [&](int x, int y) { return pick[x] > pick[y] || (pick[x] == pick[y] && x < y); });
    order.resize(used);
    experts_[slot] = order;
    float wsum = 0;
    for (int e : order) wsum += probs[e];
    const float *gs = l.gate_up_scale ? full(l.gate_up_scale) : nullptr, *ds = l.down_scale ? full(l.down_scale) : nullptr;
    Vec ffn(n, 0.0f), gu(2 * (size_t)nff), act(nff), down(n);
    const float beta = d_.situ_beta, lb = d_.situ_linear_beta;
    for (int e : order) {
        matvec(l.gate_up, f.data(), gu.data(), (uint64_t)e * 2 * nff, 2ull * nff, gs ? gs[e] : 1.0f);
        for (int i = 0; i < nff; i++) {
            const float g = gu[i], u = gu[nff + i];
            act[i] = beta * std::tanh(g / beta) * sigmoid(g) * (lb > 0 ? lb * std::tanh(u / lb) : u);
        }
        matvec(l.down, act.data(), down.data(), (uint64_t)e * n, n, ds ? ds[e] : 1.0f);
        const float w = probs[e] / (wsum + 1e-20f) * d_.expert_scale;
        for (int i = 0; i < n; i++) ffn[i] += w * down[i];
    }
    convolve(ffn.data(), n, l.ffn_conv, nullptr, 1, slot, 2 * nkv * hd + n);
    if (debug) {
        auto mean = [&](const std::vector<float> &x) { float t = 0; for (float y : x) t += y; return t / x.size(); };
        auto len = [&](const Vec &x) { return std::sqrt(dot(x.data(), x.data(), x.size())); };
        fprintf(stderr, "  pos %d slot %2d layer %2d %s: |in| %8.2f |q| %7.2f |k| %7.2f sinklogit %7.2f  attn: sink %.3f self %.3f prev %.3f  |attn| %8.2f |ffn_in| %8.2f |ffn| %8.2f  wsum %.3f experts %d %d %d\n", pos, slot, ip,
                l.window ? "win " : "full", len(in), len(q) / std::sqrt((float)nh), len(k) / std::sqrt((float)nkv), mean(sinks), mean(dbg_sink), mean(dbg_self), mean(dbg_prev), len(a), len(cur), len(ffn), wsum, order[0], order[1], order[2]);
    }
    for (int i = 0; i < n; i++) cur[i] += ffn[i];
    return cur;
}

void InklingRef::eval_one(int32_t token) {
    if (n_past_ >= n_ctx_) throw std::runtime_error("context is full");
    if (token < 0 || token >= d_.n_vocab) throw std::runtime_error("token id out of range");
    const int n = d_.n_embd, start = d_.loop_start, end = d_.loop_end;
    // Positions scale the queries of the full-attention layers, logarithmically.
    tau_ = 1.0f + d_.log_alpha * std::log(std::max(1.0f, (float)(n_past_ + 1) / (float)d_.log_floor));

    Vec cur(n);
    {
        const TypeInfo *ti = type_info(d_.tok_embd->type);
        const uint64_t row_bytes = (uint64_t)n / ti->block_elems * ti->block_bytes;
        if (!dequantize(d_.tok_embd->type, d_.tok_embd->data + (uint64_t)token * row_bytes, cur.data(), n)) throw std::runtime_error("no decoder for the embedding table");
    }
    rms_norm(cur.data(), full(d_.tok_norm), cur.data(), n, d_.norm_eps);

    std::vector<Vec> blocks;
    if (d_.loop_count <= 1) {
        for (int il = 0; il < d_.n_layer; il++) cur = layer(il, il, cur, blocks, nullptr);
    } else {
        for (int il = 0; il < start; il++) cur = layer(il, il, cur, blocks, nullptr);
        // Every pass starts from the output of the layers before the loop.
        const Vec prelude = cur;
        const std::vector<Vec> prelude_blocks = blocks;
        for (int pass = 0; pass < d_.loop_count; pass++) {
            // What the previous pass hands on: its block inputs after the first, and its output.
            std::vector<Vec> bank; Vec bank_out;
            if (pass > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur; cur = prelude; blocks = prelude_blocks;
            }
            for (int ip = start; ip < end; ip++) {
                if (pass == 0) { cur = layer(ip, ip, cur, blocks, nullptr); continue; }
                const int slot = d_.n_layer + (pass - 1) * (end - start) + (ip - start);
                Vec messages = mix(bank, bank_out, d_.layers[ip].msg_query, 4);
                cur = layer(slot, ip, cur, blocks, &messages);
            }
        }
        for (int il = end; il < d_.n_layer; il++) cur = layer(il, il, cur, blocks, nullptr);
    }

    Vec out = mix(blocks, cur, d_.output_query, 1);
    rms_norm(out.data(), full(d_.output_norm), out.data(), n, d_.norm_eps);
    Vec all(d_.n_vocab_padded);
    matvec(d_.output, out.data(), all.data());
    for (int i = 0; i < d_.n_vocab; i++) logits_[i] = d_.softcap != 0.0f ? d_.softcap * std::tanh(all[i] / d_.softcap) : all[i];
    n_past_++;
}

const std::vector<float> &InklingRef::eval(const int32_t *tokens, int n) {
    if (n <= 0) throw std::runtime_error("eval needs at least one token");
    for (int i = 0; i < n; i++) eval_one(tokens[i]);
    return logits_;
}

}  // namespace gfx
