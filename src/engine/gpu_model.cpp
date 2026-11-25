#include "gpu_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "core/quant.h"
#include "pack.h"

namespace gfx {

namespace {
struct PcMatvec { uint32_t n_in, n_rows, row0, flags; };
struct PcMatvecQ { uint32_t n_sb, stride, n_rows, row0, flags; };
struct PcNorm { uint32_t n, flags; float eps; };
struct PcAttnPrep { uint32_t hd, nh, nkv, qw, rope_dim, flags; float eps; uint32_t stride; };
struct PcMatmatQ { uint32_t n_sb, stride, n_rows, row0, flags, n_tokens, x_stride, xs_stride, y_stride; };
struct PcMatmat { uint32_t n_in, n_rows, row0, flags, n_tokens, y_stride; };
struct PcScore { uint32_t hd, stride, n_ctx; float scale; uint32_t nh, tq0; };
struct PcSoft { uint32_t n_ctx, nh, tq0; };
struct PcVsum { uint32_t hd, nh, nkv, n_ctx, seg, part_stride, tq0; };
struct PcCombine { uint32_t hd, nh, seg, flags, part_stride, tq0; };
struct PcDnUpdate { uint32_t hk, s, n_tokens, hv_total, c_total; };
struct PcDnHist { uint32_t c_total, hw, n_tokens; };
constexpr uint32_t k_tile = 4;                       // tokens per workgroup in the batched matrix kernels (T in the shaders)
constexpr uint64_t k_scores_budget = 64ull << 20;    // bytes of score scratch (bounds how many queries are scored together)
constexpr uint32_t k_seg = 1024;        // cached positions per value-sum segment
struct PcDnPrep { uint32_t n_norm, kw; float eps; uint32_t c_total; };
constexpr uint32_t k_ctl_header = 16;  // bytes before the rotation table
}  // namespace

uint64_t GpuModel::bytes_per_token(const ModelDesc &d, bool kv8) {
    uint64_t words = kv8 ? d.head_dim / 4 + d.head_dim / 64 : d.head_dim / 2;
    // key + value cache in every attention layer
    return (uint64_t)d.n_attn_layers() * 2 * 4 * d.n_head_kv * words;
}

GpuModel::GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16, bool kv8, uint64_t reserve_bytes, int batch) : dev_(dev), g_(g), d_(desc), n_ctx_(n_ctx), force_fp16_(force_fp16), kv8_(kv8) {
    // Batching addresses per-token vectors inside shared buffers, which needs aligned vector sizes.
    if (d_.n_embd % 16 || d_.n_ff % 16 || d_.head_dim % 16 || (d_.dn_head_dim % 16)) throw std::runtime_error("vector sizes that are not multiples of 16 are not supported");
    bool can_batch = d_.n_embd % 64 == 0 && d_.n_ff % 16 == 0 && (d_.n_head * d_.head_dim) % 16 == 0 && (d_.dn_v_heads * d_.dn_head_dim) % 16 == 0;
    batch_ = can_batch ? std::max(1, std::min(batch, 256)) : 1;
    batch_cap_ = (batch_ + (int)k_tile - 1) / (int)k_tile * (int)k_tile;
    if (kv8_ && d_.head_dim % 64) throw std::runtime_error("the 8-bit cache needs a head size that is a multiple of 64");
    if (d_.head_dim % 2 || d_.n_head % d_.n_head_kv) throw std::runtime_error("unsupported attention head layout");
    kv_words_ = kv8_ ? d_.head_dim / 4 + d_.head_dim / 64 : d_.head_dim / 2;
    ind_ = dev_.alloc_host(32ull * batch_cap_);
    if (d_.head_dim > 256) throw std::runtime_error("attention heads wider than 256 are not supported");
    dummy_ = dev_.alloc(256); dev_.fill_zero(dummy_);
    ctl_ = dev_.alloc_host(k_ctl_header + 4ull * std::max(d_.rope_dim, 2) * batch_cap_);
    embd_ = dev_.alloc_host(4ull * d_.n_embd * batch_cap_);
    logits_host_ = dev_.alloc_host(4ull * d_.n_vocab);
    logits_.resize(d_.n_vocab);

    layers_.resize(d_.n_layer);
    for (int il = 0; il < d_.n_layer; il++) {
        const LayerDesc &ld = d_.layers[il];
        Layer &l = layers_[il];
        l.recurrent = ld.recurrent;
        l.attn_norm = load_vector(ld.attn_norm); l.ffn_norm = load_vector(ld.ffn_norm);
        l.ffn_gate = load_matrix(ld.ffn_gate); l.ffn_up = load_matrix(ld.ffn_up); l.ffn_down = load_matrix(ld.ffn_down);
        if (ld.recurrent) {
            l.wqkv = load_matrix(ld.wqkv); l.wz = load_matrix(ld.wz); l.wbeta = load_matrix(ld.wbeta); l.walpha = load_matrix(ld.walpha);
            l.ssm_out = load_matrix(ld.ssm_out);
            l.conv_w = load_vector(ld.conv); l.dt = load_vector(ld.dt_bias); l.a = load_vector(ld.ssm_a); l.ssm_norm = load_vector(ld.ssm_norm);
            l.state = dev_.alloc(4ull * d_.dn_v_heads * d_.dn_head_dim * d_.dn_head_dim);
            l.hist = dev_.alloc(4ull * d_.dn_conv_dim() * (d_.dn_conv - 1));
            state_bytes_ += l.state.size + l.hist.size;
            zero_on_reset_.push_back(l.state); zero_on_reset_.push_back(l.hist);
        } else {
            l.wq = load_matrix(ld.wq, ld.bq); l.wk = load_matrix(ld.wk, ld.bk); l.wv = load_matrix(ld.wv, ld.bv); l.wo = load_matrix(ld.wo, ld.bo);
            l.has_qk_norm = ld.q_norm && ld.k_norm;
            if (l.has_qk_norm) {
                std::vector<float> both(2 * (size_t)d_.head_dim);
                dequantize(ld.q_norm->type, ld.q_norm->data, both.data(), d_.head_dim);
                dequantize(ld.k_norm->type, ld.k_norm->data, both.data() + d_.head_dim, d_.head_dim);
                l.qk_norm = dev_.alloc(both.size() * 4); dev_.upload(l.qk_norm, both.data(), both.size() * 4);
            }
        }
    }
    output_norm_ = load_vector(d_.output_norm);
    output_ = load_matrix(d_.output);

    // Context size. With n_ctx <= 0, take what the free GPU memory allows.
    const uint64_t per_pos_layer = 4ull * d_.n_head_kv * kv_words_;
    const uint64_t limit_ctx = dev_.caps().max_storage_range / std::max<uint64_t>(per_pos_layer, 4ull * d_.n_head) / 64 * 64;
    if (n_ctx_ <= 0) {
        const uint64_t fixed = 4ull * batch_cap_ * (4 * d_.n_embd + 4 * d_.n_ff + 8 * d_.n_head * d_.head_dim + 4 * d_.dn_conv_dim()) + 4ull * d_.n_vocab + 2 * k_scores_budget + (16ull << 20);
        uint64_t freeb = dev_.vram_free();
        if (freeb <= reserve_bytes + fixed) throw std::runtime_error("no GPU memory left for context after loading the weights");
        uint64_t fit = (freeb - reserve_bytes - fixed) / bytes_per_token(d_, kv8_);
        if (d_.n_ctx_train > 0) fit = std::min<uint64_t>(fit, d_.n_ctx_train);
        n_ctx_ = (int)(std::min(fit, limit_ctx) / 64 * 64);
        if (n_ctx_ < 64) throw std::runtime_error("not enough GPU memory left for any context");
    }
    if ((uint64_t)n_ctx_ > limit_ctx) throw std::runtime_error("context too large for this device's buffer limit (maximum " + std::to_string(limit_ctx) + ")");
    stride_ = ((uint32_t)n_ctx_ + 63) / 64 * 64;
    for (Layer &l : layers_) {
        if (l.recurrent) continue;
        uint64_t kbytes = per_pos_layer * stride_, vbytes = per_pos_layer * (uint64_t)n_ctx_;
        l.kc = dev_.alloc(kbytes); l.vc = dev_.alloc(vbytes);
        state_bytes_ += kbytes + vbytes;
    }
    alloc_activations();
    reset();
    program(1);
}
GpuModel::~GpuModel() = default;

void GpuModel::reset() {
    n_past_ = 0;
    for (const Buffer &b : zero_on_reset_) dev_.fill_zero(b);
}

void GpuModel::skip_to(int n) {
    reset();
    for (Layer &l : layers_) if (!l.recurrent) { dev_.fill_zero(l.kc); dev_.fill_zero(l.vc); }
    n_past_ = n;
}

Buffer GpuModel::load_vector(const TensorInfo *t) {
    std::vector<float> v(t->nelems);
    if (!dequantize(t->type, t->data, v.data(), t->nelems)) throw std::runtime_error("no decoder for tensor " + t->name);
    Buffer b = dev_.alloc(v.size() * 4);
    dev_.upload(b, v.data(), v.size() * 4);
    weight_bytes_ += b.size;
    return b;
}

// Upload a matrix. Encodings with a native layout keep their exact bits (see pack.h); anything else
// is stored as half-precision rows. Either way the matrix is split into row chunks that fit the
// device's storage-binding limit.
GpuModel::Matrix GpuModel::load_matrix(const TensorInfo *w, const TensorInfo *bias) {
    Matrix m; m.n_in = (uint32_t)w->dim(0); m.n_out = (uint32_t)w->dim(1);
    const TypeInfo *ti = type_info(w->type);
    if (m.n_in % ti->block_elems) throw std::runtime_error("tensor " + w->name + ": rows are not block aligned");
    const uint64_t limit = std::min<uint64_t>(dev_.caps().max_storage_range, 1ull << 30);
    m.fmt = force_fp16_ ? nullptr : pack_format(w->type);
    if (m.fmt && m.n_in % m.fmt->sb_cols) m.fmt = nullptr;
    if (m.fmt && m.fmt->needs_table) {
        bool have = false;
        for (auto &t : tables_) have |= t.first == m.fmt->type;
        if (!have) {
            uint32_t count; const uint32_t *data = pack_table(*m.fmt, &count);
            Buffer b = dev_.alloc(count * 4ull); dev_.upload(b, data, count * 4ull);
            tables_.emplace_back(m.fmt->type, b);
        }
    }
    if (m.fmt) {
        const uint64_t words_per_row = (uint64_t)(m.n_in / m.fmt->sb_cols) * m.fmt->sb_words;
        uint64_t max_rows = limit / (words_per_row * 4) / 64 * 64;
        if (max_rows == 0) throw std::runtime_error("tensor " + w->name + " is too wide for this device's buffer limit");
        std::vector<uint32_t> buf;
        for (uint32_t r0 = 0; r0 < m.n_out;) {
            uint32_t rows = (uint32_t)std::min<uint64_t>(max_rows, m.n_out - r0), stride = (rows + 63) / 64 * 64;
            buf.resize(words_per_row * stride);
            pack_rows(*m.fmt, w->data, m.n_in, r0, rows, stride, buf.data());
            Buffer b = dev_.alloc(buf.size() * 4);
            dev_.upload(b, buf.data(), buf.size() * 4);
            m.chunks.push_back(b); m.chunk_rows.push_back(rows);
            weight_bytes_ += b.size; r0 += rows;
        }
    } else {
        uint64_t row_bytes = 2ull * m.n_in;
        uint32_t rows_per_chunk = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(m.n_out, limit / row_bytes));
        uint64_t src_row_bytes = (uint64_t)m.n_in / ti->block_elems * ti->block_bytes;
        std::vector<float> f(m.n_in);
        std::vector<uint16_t> hbuf;
        for (uint32_t r0 = 0; r0 < m.n_out; r0 += rows_per_chunk) {
            uint32_t rows = std::min(rows_per_chunk, m.n_out - r0);
            hbuf.resize((size_t)rows * m.n_in);
            for (uint32_t r = 0; r < rows; r++) {
                if (!dequantize(w->type, w->data + (uint64_t)(r0 + r) * src_row_bytes, f.data(), m.n_in)) throw std::runtime_error("no decoder for tensor " + w->name);
                uint16_t *o = &hbuf[(size_t)r * m.n_in];
                for (uint32_t i = 0; i < m.n_in; i++) o[i] = fp32_to_fp16(f[i]);
            }
            Buffer b = dev_.alloc(hbuf.size() * 2);
            dev_.upload(b, hbuf.data(), hbuf.size() * 2);
            m.chunks.push_back(b); m.chunk_rows.push_back(rows);
            weight_bytes_ += b.size;
        }
    }
    if (bias) m.bias = load_vector(bias);
    return m;
}

Buffer GpuModel::scratch(uint64_t n_floats) { Buffer b = dev_.alloc(n_floats * 4); dev_.fill_zero(b); return b; }

GpuModel::Vec GpuModel::scratch_vec(uint64_t n) { return Vec{scratch(n), scratch((n + 15) / 16), 0}; }

void GpuModel::alloc_activations() {
    const uint64_t B = batch_cap_, n = d_.n_embd, hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, qw = d_.q_has_gate ? 2 * hd : hd;
    const uint64_t S = d_.dn_head_dim, HV = d_.dn_v_heads, C = d_.dn_conv_dim(), n_seg_max = ((uint64_t)n_ctx_ + k_seg - 1) / k_seg;
    Act &a = act_;
    a.x = scratch(B * n); a.a = scratch(B * n); a.g = scratch(B * d_.n_ff); a.u = scratch(B * d_.n_ff);
    a.h = scratch_vec(B * n); a.act = scratch_vec(B * d_.n_ff); a.mix = scratch_vec(B * nh * hd); a.h_last = scratch_vec(n);
    a.qfull = scratch(B * nh * qw); a.kraw = scratch(B * nkv * hd); a.vraw = scratch(B * nkv * hd);
    a.q = scratch(B * nh * hd); a.gate = scratch(B * nh * hd); a.zsum = scratch(B * nh);
    // Scores need one row of n_ctx per query head and query token, so queries are scored in
    // sub-batches small enough for the scratch budget.
    uint64_t budget = k_scores_budget;
    if (const char *e = getenv("GFX_SCORES_BUDGET")) budget = strtoull(e, nullptr, 10);   // testing aid
    const uint64_t per_query = 4ull * nh * stride_, limit = std::min<uint64_t>(budget, dev_.caps().max_storage_range);
    q_batch_ = (int)std::max<uint64_t>(1, std::min<uint64_t>(batch_, limit / per_query));
    part_stride_ = (uint32_t)(n_seg_max * nh * hd);
    a.scores = dev_.alloc(per_query * q_batch_);
    a.part = dev_.alloc(4ull * part_stride_ * q_batch_);
    if (C > 0) {
        a.mixed = scratch(B * C); a.z = scratch(B * HV * S); a.alpha = scratch(B * HV); a.beta = scratch(B * HV);
        a.qkv = scratch(B * C); a.dn_o = scratch(B * HV * S); a.dn_on = scratch_vec(B * HV * S);
    }
    a.logits = scratch(d_.n_vocab);
}

void GpuModel::matmul(Program &p, const Matrix &m, const Vec &x, const Buffer &y, uint32_t n_tokens) {
    const Buffer &bias = m.bias.valid() ? m.bias : dummy_;
    const bool batched = n_tokens > 1;
    const uint32_t tiles = (n_tokens + k_tile - 1) / k_tile;
    uint32_t row0 = 0, flags = m.bias.valid() ? 1u : 0u;
    for (size_t c = 0; c < m.chunks.size(); c++) {
        uint32_t rows = m.chunk_rows[c];
        if (m.fmt) {
            uint32_t stride = (rows + 63) / 64 * 64;
            std::vector<Binding> b{R(m.chunks[c]), R(x.v), R(x.s), W(y), R(bias)};
            if (m.fmt->needs_table) for (auto &t : tables_) if (t.first == m.fmt->type) b.push_back(R(t.second));
            if (batched) {
                std::string kernel = std::string("matmat") + (m.fmt->kernel + 6);   // "matvec_x" -> "matmat_x"
                PcMatmatQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, flags, n_tokens, m.n_in, m.n_in / 16, m.n_out};
                p.dispatch(dev_.pipeline(kernel), b, &pc, sizeof pc, stride / 64, tiles);
            } else {
                PcMatvecQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, flags};
                p.dispatch(dev_.pipeline(m.fmt->kernel), b, &pc, sizeof pc, stride / 64);
            }
        } else if (batched) {
            PcMatmat pc{m.n_in, rows, row0, flags, n_tokens, m.n_out};
            p.dispatch(dev_.pipeline("matmat_f16"), {R(m.chunks[c]), R(x.v), W(y), R(bias)}, &pc, sizeof pc, (rows + 63) / 64, tiles);
        } else {
            PcMatvec pc{m.n_in, rows, row0, flags};
            p.dispatch(dev_.pipeline("matvec_f16"), {R(m.chunks[c]), R(x.v), W(y), R(bias)}, &pc, sizeof pc, (rows + 63) / 64);
        }
        row0 += rows;
    }
}

// The forward pass for N tokens at consecutive positions, recorded once per N and resubmitted.
// Everything position dependent (first position, rotation tables, dispatch sizes of the
// position-parallel attention kernels) is read from host buffers at submission time.
Program &GpuModel::program(int n_tokens) {
    auto &slot = programs_[n_tokens];
    if (slot) return *slot;
    const uint32_t N = (uint32_t)n_tokens;
    const int n = d_.n_embd, hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, qw = d_.q_has_gate ? 2 * hd : hd;
    const int S = d_.dn_head_dim, HK = d_.dn_k_heads, HV = d_.dn_v_heads, C = d_.dn_conv_dim();
    const uint32_t G = (uint32_t)(nh / nkv);
    Act &a = act_;

    const Pipeline *p_norm = dev_.pipeline("norm"), *p_swiglu = dev_.pipeline("swiglu"), *p_soft = dev_.pipeline("attn_soft");
    const Pipeline *p_score = dev_.pipeline(kv8_ ? "attn_score_kv8" : "attn_score", {G});
    const Pipeline *p_vsum = dev_.pipeline(kv8_ ? "attn_vsum_kv8" : "attn_vsum", {(uint32_t)(kv8_ ? hd / 4 : hd / 2), G});
    const Pipeline *p_combine = dev_.pipeline("attn_combine", {(uint32_t)hd});
    const Pipeline *p_prep = dev_.pipeline(kv8_ ? "attn_prep_kv8" : "attn_prep", {(uint32_t)hd});
    const Pipeline *p_dn_prep = C > 0 ? dev_.pipeline("dn_prep", {(uint32_t)S}) : nullptr;
    const Pipeline *p_dn_update = C > 0 ? dev_.pipeline("dn_update", {(uint32_t)S}) : nullptr;
    const Pipeline *p_dn_hist = C > 0 ? dev_.pipeline("dn_hist") : nullptr;

    slot = std::make_unique<Program>(dev_);
    Program &p = *slot;
    p.copy(embd_.sub(0, 4ull * N * n), a.x.sub(0, 4ull * N * n));
    // Normalise `groups` spans of `len` elements into y, with its 16-element block sums.
    auto norm = [&](const Buffer &xin, const Buffer &add, const Buffer &w, const Buffer &gt, const Vec &y, uint32_t len, uint32_t groups) {
        PcNorm pc{len, (add.valid() ? 1u : 0u) | (gt.valid() ? 2u : 0u) | 4u, d_.norm_eps};
        p.dispatch(p_norm, {W(xin), R(add.valid() ? add : dummy_), R(w), R(gt.valid() ? gt : dummy_), W(y.v), W(y.s)}, &pc, sizeof pc, groups);
    };
    for (int il = 0; il < d_.n_layer; il++) {
        Layer &l = layers_[il];
        norm(a.x, il > 0 ? a.a : Buffer{}, l.attn_norm, Buffer{}, a.h, n, N);
        if (l.recurrent) {
            matmul(p, l.wqkv, a.h, a.mixed, N); matmul(p, l.wz, a.h, a.z, N); matmul(p, l.wbeta, a.h, a.beta, N); matmul(p, l.walpha, a.h, a.alpha, N);
            PcDnPrep pp{(uint32_t)(2 * HK), (uint32_t)d_.dn_conv, d_.norm_eps, (uint32_t)C};
            p.dispatch(p_dn_prep, {R(a.mixed), R(l.hist), R(l.conv_w), W(a.qkv)}, &pp, sizeof pp, C / S, N);
            PcDnHist ph{(uint32_t)C, (uint32_t)(d_.dn_conv - 1), N};
            p.dispatch(p_dn_hist, {R(a.mixed), W(l.hist)}, &ph, sizeof ph, (C + 63) / 64);
            PcDnUpdate pu{(uint32_t)HK, (uint32_t)S, N, (uint32_t)HV, (uint32_t)C};
            p.dispatch(p_dn_update, {R(a.qkv), R(a.alpha), R(a.beta), R(l.dt), R(l.a), W(l.state), W(a.dn_o)}, &pu, sizeof pu, HV);
            norm(a.dn_o, Buffer{}, l.ssm_norm, a.z, a.dn_on, S, N * HV);
            matmul(p, l.ssm_out, a.dn_on, a.a, N);
        } else {
            matmul(p, l.wq, a.h, a.qfull, N); matmul(p, l.wk, a.h, a.kraw, N); matmul(p, l.wv, a.h, a.vraw, N);
            PcAttnPrep pp{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, (uint32_t)qw, (uint32_t)d_.rope_dim,
                          (l.has_qk_norm ? 1u : 0u) | (d_.q_has_gate ? 2u : 0u) | (d_.rope_half_split ? 4u : 0u), d_.norm_eps, stride_};
            p.dispatch(p_prep, {R(a.qfull), R(a.kraw), R(a.vraw), R(l.has_qk_norm ? l.qk_norm : dummy_), R(ctl_), W(a.q), W(a.gate), W(l.kc), W(l.vc)}, &pp, sizeof pp, std::max(nh, nkv), N);
            // All keys and values of the batch are in the cache now; each query attends up to its own position.
            for (uint32_t q0 = 0, sub = 0; q0 < N; q0 += q_batch_, sub++) {
                uint32_t nb = std::min<uint32_t>(q_batch_, N - q0);
                PcScore ps{(uint32_t)hd, stride_, stride_, 1.0f / std::sqrt((float)hd), (uint32_t)nh, q0};
                p.dispatch_indirect(p_score, {R(a.q), R(l.kc), R(ctl_), W(a.scores)}, &ps, sizeof ps, ind_.sub(32ull * sub, 12));
                PcSoft pf{stride_, (uint32_t)nh, q0};
                p.dispatch(p_soft, {R(ctl_), W(a.scores), W(a.zsum)}, &pf, sizeof pf, nh, nb);
                PcVsum pv{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, stride_, k_seg, part_stride_, q0};
                p.dispatch_indirect(p_vsum, {R(l.vc), R(a.scores), R(ctl_), W(a.part)}, &pv, sizeof pv, ind_.sub(32ull * sub + 16, 12));
                PcCombine pcb{(uint32_t)hd, (uint32_t)nh, k_seg, d_.q_has_gate ? 1u : 0u, part_stride_, q0};
                p.dispatch(p_combine, {R(a.part), R(a.zsum), R(a.gate), R(ctl_), W(a.mix.v), W(a.mix.s)}, &pcb, sizeof pcb, nh, nb);
            }
            matmul(p, l.wo, a.mix, a.a, N);
        }
        norm(a.x, a.a, l.ffn_norm, Buffer{}, a.h, n, N);
        matmul(p, l.ffn_gate, a.h, a.g, N); matmul(p, l.ffn_up, a.h, a.u, N);
        uint32_t nff = N * d_.n_ff;
        p.dispatch(p_swiglu, {R(a.g), R(a.u), W(a.act.v), W(a.act.s)}, &nff, sizeof nff, (nff / 16 + 63) / 64);
        matmul(p, l.ffn_down, a.act, a.a, N);
    }
    // Only the last token's logits are needed.
    const uint64_t last = 4ull * (N - 1) * n;
    norm(a.x.sub(last, 4ull * n), a.a.sub(last, 4ull * n), output_norm_, Buffer{}, a.h_last, n, 1);
    matmul(p, output_, a.h_last, a.logits, 1);
    p.copy(a.logits, logits_host_);
    p.finish();
    return p;
}

size_t GpuModel::dispatches_per_token() const { return programs_.at(1)->n_dispatches(); }
size_t GpuModel::barriers_per_token() const { return programs_.at(1)->n_barriers(); }

void GpuModel::eval_batch(const int32_t *tokens, int n) {
    if (n_past_ + n > n_ctx_) throw std::runtime_error("context is full");
    // Embedding rows are decoded on the CPU; the table never occupies GPU memory unless it is
    // also the output projection.
    const TypeInfo *ti = type_info(d_.tok_embd->type);
    const uint64_t row_bytes = (uint64_t)(d_.n_embd / ti->block_elems) * ti->block_bytes;
    const uint32_t pos0 = (uint32_t)n_past_;
    memcpy(ctl_.mapped, &pos0, 4);
    float *rot = (float *)((char *)ctl_.mapped + k_ctl_header);
    for (int t = 0; t < n; t++) {
        if (tokens[t] < 0 || tokens[t] >= d_.n_vocab) throw std::runtime_error("token id out of range");
        if (!dequantize(d_.tok_embd->type, d_.tok_embd->data + (uint64_t)tokens[t] * row_bytes, (float *)embd_.mapped + (size_t)t * d_.n_embd, d_.n_embd))
            throw std::runtime_error("no decoder for the embedding table");
        // Rotation table for this token's position, computed in double precision.
        for (int i = 0; i < d_.rope_dim / 2; i++) {
            double theta = (double)(pos0 + t) * std::pow((double)d_.rope_base, -2.0 * i / d_.rope_dim);
            rot[(size_t)t * d_.rope_dim + 2 * i] = (float)std::cos(theta); rot[(size_t)t * d_.rope_dim + 2 * i + 1] = (float)std::sin(theta);
        }
    }
    // Group counts of the position-parallel attention kernels, per query sub-batch: enough
    // positions for the last query of the sub-batch.
    uint32_t *ind = (uint32_t *)ind_.mapped;
    for (uint32_t q0 = 0, sub = 0; q0 < (uint32_t)n; q0 += q_batch_, sub++) {
        uint32_t nb = std::min<uint32_t>(q_batch_, n - q0), last_pos = pos0 + q0 + nb - 1;
        uint32_t v[8] = {last_pos / 64 + 1, (uint32_t)d_.n_head_kv, nb, 0, last_pos / k_seg + 1, (uint32_t)d_.n_head_kv, nb, 0};
        memcpy(ind + 8 * sub, v, sizeof v);
    }
    program(n).run();
    n_past_ += n;
}

const std::vector<float> &GpuModel::eval(const int32_t *tokens, int n) {
    if (n <= 0) throw std::runtime_error("eval needs at least one token");
    for (int done = 0; done < n;) {
        int take = std::min(batch_, n - done);
        eval_batch(tokens + done, take);
        done += take;
    }
    memcpy(logits_.data(), logits_host_.mapped, logits_.size() * 4);
    return logits_;
}

}  // namespace gfx

namespace gfx {

namespace {
struct StateHeader {
    char magic[8];
    uint32_t version, kv8, n_layer, n_attn, head_dim, n_head_kv, kv_words, n_embd, n_vocab, n_past, n_tokens, pad;
    uint64_t model_bytes;
};
const char k_state_magic[8] = {'G', 'F', 'X', 'C', 'T', 'X', '1', 0};

struct File {
    FILE *f; std::string path;
    File(const std::string &p, const char *mode) : f(fopen(p.c_str(), mode)), path(p) { if (!f) throw std::runtime_error("cannot open " + p); }
    ~File() { if (f) fclose(f); }
    void write(const void *d, size_t n) { if (n && fwrite(d, 1, n, f) != n) throw std::runtime_error("write failed: " + path); }
    void read(void *d, size_t n) { if (n && fread(d, 1, n, f) != n) throw std::runtime_error("state file is truncated: " + path); }
};
}  // namespace

void GpuModel::save_state(const std::string &path, const std::vector<int32_t> &tokens) {
    File f(path, "wb");
    StateHeader h{};
    memcpy(h.magic, k_state_magic, 8);
    h.version = 1; h.kv8 = kv8_; h.n_layer = d_.n_layer; h.n_attn = d_.n_attn_layers(); h.head_dim = d_.head_dim; h.n_head_kv = d_.n_head_kv;
    h.kv_words = kv_words_; h.n_embd = d_.n_embd; h.n_vocab = d_.n_vocab; h.n_past = n_past_; h.n_tokens = (uint32_t)tokens.size(); h.model_bytes = g_.file_size;
    f.write(&h, sizeof h);
    f.write(tokens.data(), tokens.size() * 4);
    f.write(logits_.data(), logits_.size() * 4);
    std::vector<uint8_t> buf;
    auto dump = [&](const Buffer &b, uint64_t off, uint64_t bytes) { buf.resize(bytes); dev_.download(b.sub(off, bytes), buf.data(), bytes); f.write(buf.data(), bytes); };
    const uint64_t used = 4ull * n_past_, planes = (uint64_t)d_.n_head_kv * kv_words_;
    for (const Layer &l : layers_) {
        if (l.recurrent) { dump(l.state, 0, l.state.size); dump(l.hist, 0, l.hist.size); continue; }
        if (n_past_ == 0) continue;
        for (uint64_t pl = 0; pl < planes; pl++) dump(l.kc, pl * stride_ * 4, used);   // each key plane: the filled positions only
        dump(l.vc, 0, used * planes);
    }
}

std::vector<int32_t> GpuModel::load_state(const std::string &path) {
    File f(path, "rb");
    StateHeader h;
    f.read(&h, sizeof h);
    if (memcmp(h.magic, k_state_magic, 8) || h.version != 1) throw std::runtime_error(path + " is not a saved context");
    if (h.model_bytes != g_.file_size || h.n_layer != (uint32_t)d_.n_layer || h.head_dim != (uint32_t)d_.head_dim || h.n_head_kv != (uint32_t)d_.n_head_kv ||
        h.n_embd != (uint32_t)d_.n_embd || h.n_vocab != (uint32_t)d_.n_vocab)
        throw std::runtime_error("the saved context was made with a different model");
    if (h.kv8 != (uint32_t)kv8_ || h.kv_words != kv_words_) throw std::runtime_error(std::string("the saved context uses the ") + (h.kv8 ? "8-bit" : "half-precision") + " cache; run with the same cache type");
    if ((int)h.n_past >= n_ctx_) throw std::runtime_error("the saved context (" + std::to_string(h.n_past) + " tokens) does not fit the current context size");
    std::vector<int32_t> tokens(h.n_tokens);
    f.read(tokens.data(), tokens.size() * 4);
    f.read(logits_.data(), logits_.size() * 4);
    std::vector<uint8_t> buf;
    auto restore = [&](const Buffer &b, uint64_t off, uint64_t bytes) { buf.resize(bytes); f.read(buf.data(), bytes); dev_.upload(b.sub(off, bytes), buf.data(), bytes); };
    const uint64_t used = 4ull * h.n_past, planes = (uint64_t)d_.n_head_kv * kv_words_;
    for (const Layer &l : layers_) {
        if (l.recurrent) { restore(l.state, 0, l.state.size); restore(l.hist, 0, l.hist.size); continue; }
        if (h.n_past == 0) continue;
        for (uint64_t pl = 0; pl < planes; pl++) restore(l.kc, pl * stride_ * 4, used);
        restore(l.vc, 0, used * planes);
    }
    n_past_ = (int)h.n_past;
    return tokens;
}

}  // namespace gfx
