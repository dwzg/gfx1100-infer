#include "gpu_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/quant.h"
#include "pack.h"

namespace gfx {

namespace {
struct PcMatvec { uint32_t n_in, n_rows, row0, flags; };
struct PcMatvecQ { uint32_t n_sb, stride, n_rows, row0, flags; };
struct PcNorm { uint32_t n, flags; float eps; };
struct PcAttnPrep { uint32_t hd, nh, nkv, qw, rope_dim, flags; float eps; };
struct PcAttn { uint32_t hd, nh, nkv, n_ctx, flags; float scale; };
struct PcDnPrep { uint32_t n_norm, kw; float eps; };
constexpr uint32_t k_ctl_header = 16;  // bytes before the rotation table
}  // namespace

GpuModel::GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16, bool kv8) : dev_(dev), g_(g), d_(desc), n_ctx_(n_ctx), force_fp16_(force_fp16), kv8_(kv8) {
    if (kv8_ && d_.head_dim % 32) throw std::runtime_error("the 8-bit cache needs a head size that is a multiple of 32");
    if (d_.head_dim > 256) throw std::runtime_error("attention heads wider than 256 are not supported");
    dummy_ = dev_.alloc(256); dev_.fill_zero(dummy_);
    ctl_ = dev_.alloc_host(k_ctl_header + 4ull * std::max(d_.rope_dim, 2));
    embd_ = dev_.alloc_host(4ull * d_.n_embd);
    logits_host_ = dev_.alloc_host(4ull * d_.n_vocab);
    logits_.resize(d_.n_vocab);

    uint64_t kv_row = (uint64_t)d_.n_head_kv * d_.head_dim;
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
            uint64_t bytes = (kv8_ ? 1ull : 2ull) * n_ctx_ * kv_row;
            if (bytes > dev_.caps().max_storage_range) throw std::runtime_error("context too large for this device's buffer limit");
            l.kc = dev_.alloc(bytes); l.vc = dev_.alloc(bytes);
            state_bytes_ += 2 * bytes;
            if (kv8_) { uint64_t sb = 2ull * n_ctx_ * kv_row / 32; l.ks = dev_.alloc(sb); l.vs = dev_.alloc(sb); state_bytes_ += 2 * sb; }
        }
    }
    output_norm_ = load_vector(d_.output_norm);
    output_ = load_matrix(d_.output);
    reset();
    build_program();
}
GpuModel::~GpuModel() = default;

void GpuModel::reset() {
    n_past_ = 0;
    for (const Buffer &b : zero_on_reset_) dev_.fill_zero(b);
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

GpuModel::Vec GpuModel::scratch_vec(uint32_t n) { return Vec{scratch(n), scratch((n + 15) / 16), n}; }

void GpuModel::blocksum(Program &p, const Vec &x) {
    uint32_t nb = (x.n + 15) / 16;
    p.dispatch(dev_.pipeline("blocksum"), {R(x.v), W(x.s)}, &nb, sizeof nb, (nb + 63) / 64);
}

void GpuModel::matvec(Program &p, const Matrix &m, const Vec &x, const Buffer &y) {
    const Buffer &bias = m.bias.valid() ? m.bias : dummy_;
    uint32_t row0 = 0, flags = m.bias.valid() ? 1u : 0u;
    for (size_t c = 0; c < m.chunks.size(); c++) {
        uint32_t rows = m.chunk_rows[c];
        if (m.fmt) {
            uint32_t stride = (rows + 63) / 64 * 64;
            PcMatvecQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, flags};
            p.dispatch(dev_.pipeline(m.fmt->kernel), {R(m.chunks[c]), R(x.v), R(x.s), W(y), R(bias)}, &pc, sizeof pc, stride / 64);
        } else {
            PcMatvec pc{m.n_in, rows, row0, flags};
            p.dispatch(dev_.pipeline("matvec_f16"), {R(m.chunks[c]), R(x.v), W(y), R(bias)}, &pc, sizeof pc, (rows + 63) / 64);
        }
        row0 += rows;
    }
}

void GpuModel::build_program() {
    const int n = d_.n_embd, hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, qw = d_.q_has_gate ? 2 * hd : hd;
    const int S = d_.dn_head_dim, HK = d_.dn_k_heads, HV = d_.dn_v_heads, C = d_.dn_conv_dim();
    Buffer x = scratch(n), a = scratch(n), g = scratch(d_.n_ff), u = scratch(d_.n_ff);
    Vec h = scratch_vec(n), act = scratch_vec(d_.n_ff), mix = scratch_vec(nh * hd), dn_on;
    Buffer qfull = scratch((uint64_t)nh * qw), kraw = scratch((uint64_t)nkv * hd), vraw = scratch((uint64_t)nkv * hd);
    Buffer q = scratch((uint64_t)nh * hd), gate = scratch((uint64_t)nh * hd), scores = scratch((uint64_t)nh * n_ctx_);
    Buffer mixed, z, alpha, beta, qkv, dn_o;
    if (C > 0) { mixed = scratch(C); z = scratch((uint64_t)HV * S); alpha = scratch(HV); beta = scratch(HV); qkv = scratch(C); dn_o = scratch((uint64_t)HV * S); dn_on = scratch_vec(HV * S); }
    Buffer logits = scratch(d_.n_vocab);

    const Pipeline *p_norm = dev_.pipeline("norm"), *p_swiglu = dev_.pipeline("swiglu"), *p_attn = dev_.pipeline(kv8_ ? "attn_kv8" : "attn");
    const Pipeline *p_prep = dev_.pipeline(kv8_ ? "attn_prep_kv8" : "attn_prep", {(uint32_t)hd});
    const Pipeline *p_dn_prep = C > 0 ? dev_.pipeline("dn_prep", {(uint32_t)S}) : nullptr;
    const Pipeline *p_dn_update = C > 0 ? dev_.pipeline("dn_update", {(uint32_t)S}) : nullptr;

    program_ = std::make_unique<Program>(dev_);
    Program &p = *program_;
    p.copy(embd_, x);
    auto norm = [&](const Buffer &xin, const Buffer &add, const Buffer &w, const Buffer &gt, const Buffer &y, uint32_t len, uint32_t groups) {
        PcNorm pc{len, (add.valid() ? 1u : 0u) | (gt.valid() ? 2u : 0u), d_.norm_eps};
        p.dispatch(p_norm, {W(xin), R(add.valid() ? add : dummy_), R(w), R(gt.valid() ? gt : dummy_), W(y)}, &pc, sizeof pc, groups);
    };
    for (int il = 0; il < d_.n_layer; il++) {
        Layer &l = layers_[il];
        norm(x, il > 0 ? a : Buffer{}, l.attn_norm, Buffer{}, h.v, n, 1); blocksum(p, h);
        if (l.recurrent) {
            matvec(p, l.wqkv, h, mixed); matvec(p, l.wz, h, z); matvec(p, l.wbeta, h, beta); matvec(p, l.walpha, h, alpha);
            PcDnPrep pp{(uint32_t)(2 * HK), (uint32_t)d_.dn_conv, d_.norm_eps};
            p.dispatch(p_dn_prep, {R(mixed), W(l.hist), R(l.conv_w), W(qkv)}, &pp, sizeof pp, C / S);
            uint32_t pu[2] = {(uint32_t)HK, (uint32_t)S};
            p.dispatch(p_dn_update, {R(qkv), R(alpha), R(beta), R(l.dt), R(l.a), W(l.state), W(dn_o)}, pu, sizeof pu, HV);
            norm(dn_o, Buffer{}, l.ssm_norm, z, dn_on.v, S, HV); blocksum(p, dn_on);
            matvec(p, l.ssm_out, dn_on, a);
        } else {
            matvec(p, l.wq, h, qfull); matvec(p, l.wk, h, kraw); matvec(p, l.wv, h, vraw);
            PcAttnPrep pp{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, (uint32_t)qw, (uint32_t)d_.rope_dim,
                          (l.has_qk_norm ? 1u : 0u) | (d_.q_has_gate ? 2u : 0u) | (d_.rope_half_split ? 4u : 0u), d_.norm_eps};
            std::vector<Binding> pb{R(qfull), R(kraw), R(vraw), R(l.has_qk_norm ? l.qk_norm : dummy_), R(ctl_), W(q), W(gate), W(l.kc), W(l.vc)};
            if (kv8_) { pb.push_back(W(l.ks)); pb.push_back(W(l.vs)); }
            p.dispatch(p_prep, pb, &pp, sizeof pp, std::max(nh, nkv));
            PcAttn pa{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, (uint32_t)n_ctx_, d_.q_has_gate ? 1u : 0u, 1.0f / std::sqrt((float)hd)};
            std::vector<Binding> ab{R(q), R(gate), R(l.kc), R(l.vc), R(ctl_), W(scores), W(mix.v)};
            if (kv8_) { ab.push_back(R(l.ks)); ab.push_back(R(l.vs)); }
            p.dispatch(p_attn, ab, &pa, sizeof pa, nh);
            blocksum(p, mix);
            matvec(p, l.wo, mix, a);
        }
        norm(x, a, l.ffn_norm, Buffer{}, h.v, n, 1); blocksum(p, h);
        matvec(p, l.ffn_gate, h, g); matvec(p, l.ffn_up, h, u);
        uint32_t nff = d_.n_ff;
        p.dispatch(p_swiglu, {R(g), R(u), W(act.v)}, &nff, sizeof nff, (nff + 63) / 64);
        blocksum(p, act);
        matvec(p, l.ffn_down, act, a);
    }
    norm(x, a, output_norm_, Buffer{}, h.v, n, 1); blocksum(p, h);
    matvec(p, output_, h, logits);
    p.copy(logits, logits_host_);
    p.finish();
}

size_t GpuModel::dispatches_per_token() const { return program_->n_dispatches(); }
size_t GpuModel::barriers_per_token() const { return program_->n_barriers(); }

void GpuModel::eval_one(int32_t token) {
    if (token < 0 || token >= d_.n_vocab) throw std::runtime_error("token id out of range");
    if (n_past_ >= n_ctx_) throw std::runtime_error("context is full");
    // The embedding row is decoded on the CPU; the table never occupies GPU memory unless it is
    // also the output projection.
    const TypeInfo *ti = type_info(d_.tok_embd->type);
    if (!dequantize(d_.tok_embd->type, d_.tok_embd->data + (uint64_t)token * (d_.n_embd / ti->block_elems) * ti->block_bytes, (float *)embd_.mapped, d_.n_embd))
        throw std::runtime_error("no decoder for the embedding table");
    // Position and rotation table, computed in double precision.
    uint32_t pos = (uint32_t)n_past_;
    memcpy(ctl_.mapped, &pos, 4);
    float *rot = (float *)((char *)ctl_.mapped + k_ctl_header);
    for (int i = 0; i < d_.rope_dim / 2; i++) {
        double theta = (double)pos * std::pow((double)d_.rope_base, -2.0 * i / d_.rope_dim);
        rot[2 * i] = (float)std::cos(theta); rot[2 * i + 1] = (float)std::sin(theta);
    }
    program_->run();
    n_past_++;
}

const std::vector<float> &GpuModel::eval(const int32_t *tokens, int n) {
    if (n <= 0) throw std::runtime_error("eval needs at least one token");
    for (int i = 0; i < n; i++) eval_one(tokens[i]);
    memcpy(logits_.data(), logits_host_.mapped, logits_.size() * 4);
    return logits_;
}

}  // namespace gfx
