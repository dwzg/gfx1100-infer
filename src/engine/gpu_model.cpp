#include "gpu_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
struct PcScore { uint32_t hd, stride, n_ctx; float scale; };
struct PcVsum { uint32_t hd, nh, nkv, n_ctx, seg; };
struct PcCombine { uint32_t hd, nh, seg, flags; };
constexpr uint32_t k_seg = 1024;        // cached positions per value-sum segment
struct PcDnPrep { uint32_t n_norm, kw; float eps; };
constexpr uint32_t k_ctl_header = 16;  // bytes before the rotation table
}  // namespace

uint64_t GpuModel::bytes_per_token(const ModelDesc &d, bool kv8) {
    uint64_t words = kv8 ? d.head_dim / 4 + d.head_dim / 64 : d.head_dim / 2;
    // key + value cache in every attention layer, one row of scores per query head, and the
    // per-segment value sums (shared by all layers)
    return (uint64_t)d.n_attn_layers() * 2 * 4 * d.n_head_kv * words + 4ull * d.n_head + (4ull * d.n_head * d.head_dim + k_seg - 1) / k_seg;
}

GpuModel::GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16, bool kv8, uint64_t reserve_bytes) : dev_(dev), g_(g), d_(desc), n_ctx_(n_ctx), force_fp16_(force_fp16), kv8_(kv8) {
    if (kv8_ && d_.head_dim % 64) throw std::runtime_error("the 8-bit cache needs a head size that is a multiple of 64");
    if (d_.head_dim % 2 || d_.n_head % d_.n_head_kv) throw std::runtime_error("unsupported attention head layout");
    kv_words_ = kv8_ ? d_.head_dim / 4 + d_.head_dim / 64 : d_.head_dim / 2;
    ind_ = dev_.alloc_host(32);
    if (d_.head_dim > 256) throw std::runtime_error("attention heads wider than 256 are not supported");
    dummy_ = dev_.alloc(256); dev_.fill_zero(dummy_);
    ctl_ = dev_.alloc_host(k_ctl_header + 4ull * std::max(d_.rope_dim, 2));
    embd_ = dev_.alloc_host(4ull * d_.n_embd);
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
        const uint64_t fixed = 4ull * (4 * d_.n_embd + 4 * d_.n_ff + d_.n_vocab + 8 * d_.n_head * d_.head_dim + 4 * d_.dn_conv_dim()) + (16ull << 20);
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
    reset();
    build_program();
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
            std::vector<Binding> b{R(m.chunks[c]), R(x.v), R(x.s), W(y), R(bias)};
            if (m.fmt->needs_table) for (auto &t : tables_) if (t.first == m.fmt->type) b.push_back(R(t.second));
            p.dispatch(dev_.pipeline(m.fmt->kernel), b, &pc, sizeof pc, stride / 64);
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
    const uint32_t G = (uint32_t)(nh / nkv), n_seg_max = ((uint32_t)n_ctx_ + k_seg - 1) / k_seg;
    Buffer q = scratch((uint64_t)nh * hd), gate = scratch((uint64_t)nh * hd), scores = dev_.alloc(4ull * nh * stride_), zsum = scratch(nh);
    Buffer part = dev_.alloc(4ull * n_seg_max * nh * hd);
    Buffer mixed, z, alpha, beta, qkv, dn_o;
    if (C > 0) { mixed = scratch(C); z = scratch((uint64_t)HV * S); alpha = scratch(HV); beta = scratch(HV); qkv = scratch(C); dn_o = scratch((uint64_t)HV * S); dn_on = scratch_vec(HV * S); }
    Buffer logits = scratch(d_.n_vocab);

    const Pipeline *p_norm = dev_.pipeline("norm"), *p_swiglu = dev_.pipeline("swiglu"), *p_soft = dev_.pipeline("attn_soft");
    const Pipeline *p_score = dev_.pipeline(kv8_ ? "attn_score_kv8" : "attn_score", {G});
    const Pipeline *p_vsum = dev_.pipeline(kv8_ ? "attn_vsum_kv8" : "attn_vsum", {(uint32_t)(kv8_ ? hd / 4 : hd / 2), G});
    const Pipeline *p_combine = dev_.pipeline("attn_combine", {(uint32_t)hd});
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
                          (l.has_qk_norm ? 1u : 0u) | (d_.q_has_gate ? 2u : 0u) | (d_.rope_half_split ? 4u : 0u), d_.norm_eps, stride_};
            p.dispatch(p_prep, {R(qfull), R(kraw), R(vraw), R(l.has_qk_norm ? l.qk_norm : dummy_), R(ctl_), W(q), W(gate), W(l.kc), W(l.vc)}, &pp, sizeof pp, std::max(nh, nkv));
            // Group counts for the two position-parallel kernels come from ind_, set per token.
            PcScore ps{(uint32_t)hd, stride_, stride_, 1.0f / std::sqrt((float)hd)};
            p.dispatch_indirect(p_score, {R(q), R(l.kc), R(ctl_), W(scores)}, &ps, sizeof ps, ind_.sub(0, 12));
            uint32_t nctx = stride_;
            p.dispatch(p_soft, {R(ctl_), W(scores), W(zsum)}, &nctx, sizeof nctx, nh);
            PcVsum pv{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, stride_, k_seg};
            p.dispatch_indirect(p_vsum, {R(l.vc), R(scores), R(ctl_), W(part)}, &pv, sizeof pv, ind_.sub(16, 12));
            PcCombine pcb{(uint32_t)hd, (uint32_t)nh, k_seg, d_.q_has_gate ? 1u : 0u};
            p.dispatch(p_combine, {R(part), R(zsum), R(gate), R(ctl_), W(mix.v)}, &pcb, sizeof pcb, nh);
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
    uint32_t ind[8] = {pos / 64 + 1, (uint32_t)d_.n_head_kv, 1, 0, pos / k_seg + 1, (uint32_t)d_.n_head_kv, 1, 0};
    memcpy(ind_.mapped, ind, sizeof ind);
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
