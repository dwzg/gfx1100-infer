#include "ref.h"

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "quant.h"
#include "threads.h"

namespace gfx {

static inline float dot(const float *a, const float *b, size_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; k++) acc[k] += a[i + k] * b[i + k];
    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

RefModel::RefModel(const GGUF &g, const ModelDesc &desc, int n_ctx, int n_threads) : g_(g), d_(desc), n_ctx_(n_ctx) {
    pool_ = std::make_unique<ThreadPool>(n_threads < 1 ? 1 : n_threads);
    int n_attn = 0, n_rec = 0;
    for (const auto &l : d_.layers) slot_.push_back(l.recurrent ? n_rec++ : n_attn++);
    size_t kv = (size_t)n_ctx_ * d_.n_head_kv * d_.head_dim;
    k_cache_.assign(n_attn, std::vector<float>(kv)); v_cache_.assign(n_attn, std::vector<float>(kv));
    dn_state_.assign(n_rec, std::vector<float>((size_t)d_.dn_v_heads * d_.dn_head_dim * d_.dn_head_dim));
    dn_conv_.assign(n_rec, std::vector<float>((size_t)(d_.dn_conv > 0 ? d_.dn_conv - 1 : 0) * d_.dn_conv_dim()));
    x_.resize(d_.n_embd); logits_.resize(d_.n_vocab);
}
RefModel::~RefModel() = default;

void RefModel::reset() {
    n_past_ = 0;
    for (auto &s : dn_state_) std::fill(s.begin(), s.end(), 0.0f);
    for (auto &s : dn_conv_) std::fill(s.begin(), s.end(), 0.0f);
}

const float *RefModel::weights(const TensorInfo *t) {
    if (t->type == T_F32) return (const float *)t->data;
    auto it = decoded_.find(t);
    if (it == decoded_.end()) {
        std::vector<float> v(t->nelems);
        if (!dequantize(t->type, t->data, v.data(), t->nelems)) throw std::runtime_error("no decoder for tensor " + t->name);
        it = decoded_.emplace(t, std::move(v)).first;
    }
    return it->second.data();
}

// y = W x (+ bias). A GGUF matrix has dims [n_in, n_out] and stores one output row after another.
void RefModel::matvec(const TensorInfo *w, const float *x, float *y, const TensorInfo *bias) {
    const float *W = weights(w), *b = bias ? weights(bias) : nullptr;
    size_t n_in = w->dim(0), n_out = w->dim(1), chunk = 64;
    pool_->run((n_out + chunk - 1) / chunk, [&](size_t c) {
        size_t end = std::min(n_out, (c + 1) * chunk);
        for (size_t r = c * chunk; r < end; r++) y[r] = dot(W + r * n_in, x, n_in) + (b ? b[r] : 0.0f);
    });
}

// RMS normalisation over `groups` consecutive spans of n elements, with a weight of n elements.
void RefModel::rms_norm(const float *x, const TensorInfo *w, float *y, int n, int groups) {
    const float *g = w ? weights(w) : nullptr;
    for (int k = 0; k < groups; k++, x += n, y += n) {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        float scale = 1.0f / std::sqrt((float)(ss / n) + d_.norm_eps);
        for (int i = 0; i < n; i++) y[i] = x[i] * scale * (g ? g[i] : 1.0f);
    }
}

void RefModel::rope(float *x, int n_heads, int pos) const {
    int nd = d_.rope_dim, half = nd / 2;
    for (int h = 0; h < n_heads; h++, x += d_.head_dim)
        for (int i = 0; i < half; i++) {
            float theta = (float)pos * std::pow(d_.rope_base, -2.0f * (float)i / (float)nd);
            float c = std::cos(theta), s = std::sin(theta);
            int a = d_.rope_half_split ? i : 2 * i, b = d_.rope_half_split ? i + half : 2 * i + 1;
            float x0 = x[a], x1 = x[b];
            x[a] = x0 * c - x1 * s; x[b] = x0 * s + x1 * c;
        }
}

void RefModel::attention(int il, int slot, const float *h, float *out) {
    const LayerDesc &l = d_.layers[il];
    int hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, qw = d_.q_has_gate ? 2 * hd : hd, pos = n_past_;
    std::vector<float> qfull((size_t)nh * qw), q((size_t)nh * hd), gate, k((size_t)nkv * hd), v((size_t)nkv * hd);
    matvec(l.wq, h, qfull.data(), l.bq); matvec(l.wk, h, k.data(), l.bk); matvec(l.wv, h, v.data(), l.bv);
    if (d_.q_has_gate) {
        gate.resize((size_t)nh * hd);
        for (int i = 0; i < nh; i++) { memcpy(&q[(size_t)i * hd], &qfull[(size_t)i * qw], hd * 4); memcpy(&gate[(size_t)i * hd], &qfull[(size_t)i * qw + hd], hd * 4); }
    } else q = qfull;
    if (l.q_norm) rms_norm(q.data(), l.q_norm, q.data(), hd, nh);
    if (l.k_norm) rms_norm(k.data(), l.k_norm, k.data(), hd, nkv);
    rope(q.data(), nh, pos); rope(k.data(), nkv, pos);
    if (probe) { probe(il, "q", q.data(), q.size()); probe(il, "k", k.data(), k.size()); probe(il, "v", v.data(), v.size()); }

    size_t row = (size_t)nkv * hd;
    float *K = k_cache_[slot].data(), *V = v_cache_[slot].data();
    memcpy(K + pos * row, k.data(), row * 4); memcpy(V + pos * row, v.data(), row * 4);

    std::vector<float> mix((size_t)nh * hd);
    float scale = 1.0f / std::sqrt((float)hd);
    pool_->run(nh, [&](size_t i) {
        int kvh = (int)i / (nh / nkv);
        std::vector<float> p(pos + 1);
        float mx = -INFINITY;
        for (int t = 0; t <= pos; t++) { p[t] = dot(&q[i * hd], K + t * row + (size_t)kvh * hd, hd) * scale; mx = std::max(mx, p[t]); }
        float sum = 0;
        for (int t = 0; t <= pos; t++) { p[t] = std::exp(p[t] - mx); sum += p[t]; }
        float *o = &mix[i * hd];
        std::fill(o, o + hd, 0.0f);
        for (int t = 0; t <= pos; t++) { float w = p[t] / sum; const float *vt = V + t * row + (size_t)kvh * hd; for (int d = 0; d < hd; d++) o[d] += w * vt[d]; }
        if (d_.q_has_gate) for (int d = 0; d < hd; d++) o[d] *= sigmoid(gate[i * hd + d]);
    });
    if (probe) probe(il, "attn_mix", mix.data(), mix.size());
    matvec(l.wo, mix.data(), out, l.bo);
}

// Gated delta-net layer. Each value head keeps a head_dim x head_dim state matrix that is decayed,
// corrected towards the new key/value pair, and then read with the query.
void RefModel::linear_attention(int il, int slot, const float *h, float *out) {
    const LayerDesc &l = d_.layers[il];
    int S = d_.dn_head_dim, HK = d_.dn_k_heads, HV = d_.dn_v_heads, C = d_.dn_conv_dim(), KW = d_.dn_conv, kdim = d_.dn_key_dim();
    std::vector<float> mixed(C), z((size_t)HV * S), beta(HV), alpha(HV);
    matvec(l.wqkv, h, mixed.data()); matvec(l.wz, h, z.data()); matvec(l.wbeta, h, beta.data()); matvec(l.walpha, h, alpha.data());
    const float *dt = weights(l.dt_bias), *a = weights(l.ssm_a), *cw = weights(l.conv);

    // Causal depthwise convolution over the last KW projections, then SiLU.
    float *hist = dn_conv_[slot].data();  // [C][KW-1], oldest first
    std::vector<float> conv(C);
    for (int c = 0; c < C; c++) {
        float s = mixed[c] * cw[(size_t)c * KW + KW - 1];
        for (int i = 0; i < KW - 1; i++) s += hist[(size_t)c * (KW - 1) + i] * cw[(size_t)c * KW + i];
        conv[c] = silu(s);
        for (int i = 0; i + 1 < KW - 1; i++) hist[(size_t)c * (KW - 1) + i] = hist[(size_t)c * (KW - 1) + i + 1];
        hist[(size_t)c * (KW - 1) + KW - 2] = mixed[c];
    }
    float *q = conv.data(), *k = q + kdim, *v = k + kdim;
    for (float *p : {q, k})  // L2-normalise each query and key head
        for (int g = 0; g < HK; g++) {
            double ss = 0;
            for (int i = 0; i < S; i++) ss += (double)p[g * S + i] * p[g * S + i];
            float sc = 1.0f / std::sqrt((float)ss + d_.norm_eps);
            for (int i = 0; i < S; i++) p[g * S + i] *= sc;
        }
    if (probe) { probe(il, "dn_q", q, kdim); probe(il, "dn_k", k, kdim); probe(il, "dn_v", v, (size_t)HV * S); }

    std::vector<float> o((size_t)HV * S);
    float *state = dn_state_[slot].data();
    float oscale = 1.0f / std::sqrt((float)S);
    pool_->run(HV, [&](size_t hv) {
        const float *qh = q + (hv % HK) * S, *kh = k + (hv % HK) * S, *vh = v + hv * S;
        float *M = state + hv * S * S;  // M[j][i] holds state element (i, j): row j pairs with value dim j
        float decay = std::exp(softplus(alpha[hv] + dt[hv]) * a[hv]), b = sigmoid(beta[hv]);
        for (int j = 0; j < S; j++) {
            float *row = M + (size_t)j * S;
            for (int i = 0; i < S; i++) row[i] *= decay;
            float delta = (vh[j] - dot(row, kh, S)) * b;
            for (int i = 0; i < S; i++) row[i] += delta * kh[i];
            o[hv * S + j] = dot(row, qh, S) * oscale;
        }
    });
    if (probe) probe(il, "dn_out", o.data(), o.size());
    rms_norm(o.data(), l.ssm_norm, o.data(), S, HV);
    for (size_t i = 0; i < o.size(); i++) o[i] *= silu(z[i]);
    matvec(l.ssm_out, o.data(), out);
}

void RefModel::eval_one(int32_t token) {
    if (token < 0 || token >= d_.n_vocab) throw std::runtime_error("token id out of range");
    if (n_past_ >= n_ctx_) throw std::runtime_error("context is full");
    int n = d_.n_embd;
    {   // embedding row
        const TypeInfo *ti = type_info(d_.tok_embd->type);
        if ((size_t)n % ti->block_elems) throw std::runtime_error("embedding rows are not block aligned");
        if (!dequantize(d_.tok_embd->type, d_.tok_embd->data + (size_t)token * (n / ti->block_elems) * ti->block_bytes, x_.data(), n))
            throw std::runtime_error("no decoder for the embedding table");
    }
    std::vector<float> h(n), a(n), g(d_.n_ff), u(d_.n_ff);
    for (int il = 0; il < d_.n_layer; il++) {
        const LayerDesc &l = d_.layers[il];
        rms_norm(x_.data(), l.attn_norm, h.data(), n);
        if (l.recurrent) linear_attention(il, slot_[il], h.data(), a.data());
        else attention(il, slot_[il], h.data(), a.data());
        if (probe) probe(il, "attn_out", a.data(), n);
        for (int i = 0; i < n; i++) x_[i] += a[i];
        rms_norm(x_.data(), l.ffn_norm, h.data(), n);
        matvec(l.ffn_gate, h.data(), g.data()); matvec(l.ffn_up, h.data(), u.data());
        for (int i = 0; i < d_.n_ff; i++) g[i] = silu(g[i]) * u[i];
        matvec(l.ffn_down, g.data(), a.data());
        for (int i = 0; i < n; i++) x_[i] += a[i];
        if (probe) probe(il, "layer_out", x_.data(), n);
    }
    n_past_++;
}

const std::vector<float> &RefModel::eval(const int32_t *tokens, int n) {
    if (n <= 0) throw std::runtime_error("eval needs at least one token");
    for (int i = 0; i < n; i++) eval_one(tokens[i]);
    std::vector<float> h(d_.n_embd);
    rms_norm(x_.data(), d_.output_norm, h.data(), d_.n_embd);
    matvec(d_.output, h.data(), logits_.data());
    return logits_;
}

}  // namespace gfx
