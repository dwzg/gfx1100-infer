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
struct PcAttnPrep { uint32_t hd, nh, nkv, qw, rope_dim, flags; float eps; uint32_t stride, page_shift; };
struct PcMatmatQ { uint32_t n_sb, stride, n_rows, row0, flags, n_tokens, x_stride, xs_stride, y_stride; };
struct PcMatmat { uint32_t n_in, n_rows, row0, flags, n_tokens, y_stride; };
struct PcScore { uint32_t hd, stride, n_ctx; float scale; uint32_t nh, tq0, page_shift; };
struct PcSoft { uint32_t n_ctx, nh, tq0; };
struct PcVsum { uint32_t hd, nh, nkv, n_ctx, seg, part_stride, tq0; };
struct PcCombine { uint32_t hd, nh, seg, flags, part_stride, tq0; };
struct PcDnUpdate { uint32_t hk, s, n_tokens, hv_total, c_total; };
struct PcDnHist { uint32_t c_total, hw, n_tokens; };
constexpr uint32_t k_tile = 4;                       // tokens per workgroup in the batched matrix kernels (T in the shaders)
constexpr uint32_t k_wmma_tile = 16;                 // tokens per workgroup in the matrix-instruction kernels
constexpr uint32_t k_wmma_min_tokens = 12;           // below this the portable batched kernels are used
constexpr uint64_t k_scores_budget = 64ull << 20;    // bytes of score scratch (bounds how many queries are scored together)
constexpr uint32_t k_page = 1024;       // cached positions per page (also the segment of the value sum)
struct PcDnPrep { uint32_t n_norm, kw; float eps; uint32_t c_total; };
constexpr uint32_t k_ctl_header = 16;  // bytes before the rotation table
}  // namespace

uint64_t GpuModel::bytes_per_token(const ModelDesc &d, bool kv8) {
    uint64_t words = kv8 ? d.head_dim / 4 + d.head_dim / 64 : d.head_dim / 2;
    // key + value cache in every attention layer
    return (uint64_t)d.n_attn_layers() * 2 * 4 * d.n_head_kv * words;
}

GpuModel::GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16, bool kv8, uint64_t reserve_bytes, int batch, bool allow_wmma) : dev_(dev), g_(g), d_(desc), n_ctx_(n_ctx), force_fp16_(force_fp16), kv8_(kv8) {
    // Batching addresses per-token vectors inside shared buffers, which needs aligned vector sizes.
    if (d_.n_embd % 16 || d_.n_ff % 16 || d_.head_dim % 16 || (d_.dn_head_dim % 16)) throw std::runtime_error("vector sizes that are not multiples of 16 are not supported");
    bool can_batch = d_.n_embd % 64 == 0 && d_.n_ff % 16 == 0 && (d_.n_head * d_.head_dim) % 16 == 0 && (d_.dn_v_heads * d_.dn_head_dim) % 16 == 0;
    batch_ = can_batch ? std::max(1, std::min(batch, 256)) : 1;
    wmma_emu_ = getenv("GFX_WMMA_EMU") != nullptr;   // testing aid, see the kernel source
    wmma_ = allow_wmma && (dev_.caps().coop_matrix || wmma_emu_) && batch_ >= (int)k_wmma_min_tokens && !getenv("GFX_NO_WMMA");
    const int cap_unit = wmma_ ? (int)k_wmma_tile : (int)k_tile;   // whole tiles of tokens
    batch_cap_ = (batch_ + cap_unit - 1) / cap_unit * cap_unit;
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
        if (n_ctx_ >= (int)k_page) n_ctx_ = n_ctx_ / k_page * k_page;
    }
    // Pages: 1024 positions, or less for contexts smaller than that. A requested size is rounded
    // up to whole pages.
    page_ = k_page;
    if (const char *e = getenv("GFX_PAGE")) page_ = (uint32_t)strtoul(e, nullptr, 10);   // testing aid
    while (page_ > 64 && page_ > (uint32_t)n_ctx_) page_ /= 2;
    for (page_shift_ = 6; (1u << page_shift_) < page_; page_shift_++) {}
    if ((1u << page_shift_) != page_) throw std::runtime_error("the page size must be a power of two, at least 64");
    n_ctx_ = (int)(((uint64_t)n_ctx_ + page_ - 1) / page_ * page_);
    if ((uint64_t)n_ctx_ > limit_ctx) throw std::runtime_error("context too large for this device's buffer limit (maximum " + std::to_string(limit_ctx / page_ * page_) + ")");
    stride_ = (uint32_t)n_ctx_;
    const uint32_t n_pages = (uint32_t)n_ctx_ / page_;
    page_ref_.assign(n_pages, 0);
    const bool backwards = getenv("GFX_PAGE_REVERSE") != nullptr;   // testing aid: hand pages out from the top
    for (uint32_t i = 0; i < n_pages; i++) free_pages_.push_back(backwards ? i : n_pages - 1 - i);
    pt_ = dev_.alloc_host(4ull * n_pages);
    seqs_.resize(1); seqs_[0].live = true;
    write_page_table();
    for (Layer &l : layers_) {
        if (l.recurrent) continue;
        uint64_t kbytes = per_pos_layer * stride_, vbytes = per_pos_layer * (uint64_t)n_ctx_;
        l.kc = dev_.alloc(kbytes); l.vc = dev_.alloc(vbytes);
        state_bytes_ += kbytes + vbytes;
    }
    alloc_activations();
    if (wmma_) verify_wmma();
    // Host copy of the linear-attention state, filled and read back with one submission each.
    for (const Layer &l : layers_) if (l.recurrent) recurrent_bytes_ += l.state.size + l.hist.size;
    if (recurrent_bytes_) {
        snap_ = dev_.alloc_host(recurrent_bytes_);
        snap_save_ = std::make_unique<Program>(dev_); snap_load_ = std::make_unique<Program>(dev_);
        uint64_t off = 0;
        for (const Layer &l : layers_) {
            if (!l.recurrent) continue;
            for (const Buffer *b : {&l.state, &l.hist}) {
                snap_save_->copy(*b, snap_.sub(off, b->size)); snap_load_->copy(snap_.sub(off, b->size), *b);
                off += b->size;
            }
        }
        snap_save_->finish(); snap_load_->finish();
    }
    reset();
    program(1);
}
GpuModel::~GpuModel() = default;

void GpuModel::reset() {
    seq_truncate(0);
    state_clear();
}

void GpuModel::state_clear() {
    for (const Buffer &b : zero_on_reset_) dev_.fill_zero(b);
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

void GpuModel::skip_to(int n) {
    reset();
    for (Layer &l : layers_) if (!l.recurrent) { dev_.fill_zero(l.kc); dev_.fill_zero(l.vc); }
    if (n > 0) take_pages(0, (uint32_t)n);
    seqs_[active_].len = n;
}

// ---- Sequences and cache pages ----

void GpuModel::write_page_table() {
    uint32_t *pt = (uint32_t *)pt_.mapped;
    const Seq &s = seqs_[active_];
    for (size_t i = 0; i < page_ref_.size(); i++) pt[i] = i < s.pages.size() ? s.pages[i] : 0;
}

int GpuModel::seq_create() {
    for (size_t i = 0; i < seqs_.size(); i++) if (!seqs_[i].live) { seqs_[i] = Seq{true, 0, {}}; return (int)i; }
    seqs_.push_back(Seq{true, 0, {}});
    return (int)seqs_.size() - 1;
}

void GpuModel::release_pages(Seq &s, size_t keep) {
    while (s.pages.size() > keep) {
        uint32_t pg = s.pages.back(); s.pages.pop_back();
        if (--page_ref_[pg] == 0) free_pages_.push_back(pg);
    }
}

void GpuModel::seq_destroy(int id) {
    if (id == active_) throw std::runtime_error("internal: the selected sequence cannot be destroyed");
    Seq &s = seqs_.at(id);
    if (!s.live) return;
    release_pages(s, 0);
    s = Seq{};
}

int GpuModel::seq_fork(int src, int n) {
    if (n < 0 || n > seqs_.at(src).len) throw std::runtime_error("internal: fork position outside the sequence");
    int id = seq_create();
    Seq &s = seqs_[id];
    const Seq &from = seqs_[src];
    s.pages.assign(from.pages.begin(), from.pages.begin() + ((size_t)n + page_ - 1) / page_);
    for (uint32_t pg : s.pages) page_ref_[pg]++;
    s.len = n;
    return id;
}

void GpuModel::seq_select(int id) {
    if (!seqs_.at(id).live) throw std::runtime_error("internal: selecting a sequence that does not exist");
    active_ = id;
    write_page_table();
}

void GpuModel::seq_truncate(int n) {
    Seq &s = seqs_[active_];
    if (n < 0 || n > s.len) throw std::runtime_error("internal: truncating beyond the sequence");
    release_pages(s, ((size_t)n + page_ - 1) / page_);
    s.len = n;
}

int GpuModel::pages_needed(int id, int upto) const {
    const Seq &s = seqs_.at(id);
    if (upto <= s.len) return 0;
    int need = 0;
    for (size_t pi = (size_t)s.len / page_; pi <= ((size_t)upto - 1) / page_; pi++)
        if (pi >= s.pages.size() || page_ref_[s.pages[pi]] > 1) need++;
    return need;
}

int GpuModel::pages_owned(int id) const {
    int n = 0;
    for (uint32_t pg : seqs_.at(id).pages) n += page_ref_[pg] == 1;
    return n;
}

// Make positions [pos0, pos0 + n) of the selected sequence writable: new pages where it has
// none, private copies of pages it shares.
void GpuModel::take_pages(uint32_t pos0, uint32_t n) {
    Seq &s = seqs_[active_];
    uint32_t *pt = (uint32_t *)pt_.mapped;
    for (size_t pi = pos0 / page_; pi <= (pos0 + n - 1) / page_; pi++) {
        if (pi < s.pages.size() && page_ref_[s.pages[pi]] == 1) continue;
        if (free_pages_.empty()) throw std::runtime_error("context is full");
        uint32_t pg = free_pages_.back(); free_pages_.pop_back();
        page_ref_[pg] = 1;
        if (pi < s.pages.size()) { copy_page(s.pages[pi], pg); page_ref_[s.pages[pi]]--; s.pages[pi] = pg; }
        else s.pages.push_back(pg);
        pt[pi] = pg;
    }
}

void GpuModel::copy_page(uint32_t from, uint32_t to) {
    Program p(dev_);
    const uint64_t planes = (uint64_t)d_.n_head_kv * kv_words_, pb = 4ull * page_;
    std::vector<VkBufferCopy> rk(planes);
    for (uint64_t pl = 0; pl < planes; pl++) rk[pl] = {(pl * stride_ + (uint64_t)from * page_) * 4, (pl * stride_ + (uint64_t)to * page_) * 4, pb};
    for (const Layer &l : layers_) {
        if (l.recurrent) continue;
        p.copy_many(l.kc, l.kc, rk);
        p.copy_many(l.vc, l.vc, {{from * pb * planes, to * pb * planes, pb * planes}});
    }
    p.run();
}

// The parts of a key / value cache that hold the first n positions of a sequence, in order.
GpuModel::Regions GpuModel::key_regions(const Seq &s, int n) const {
    Regions r;
    const uint64_t planes = (uint64_t)d_.n_head_kv * kv_words_;
    for (uint64_t pl = 0; pl < planes; pl++)
        for (uint64_t pi = 0; pi * page_ < (uint64_t)n; pi++)
            r.emplace_back((pl * stride_ + (uint64_t)s.pages[pi] * page_) * 4, 4 * std::min<uint64_t>(page_, n - pi * page_));
    return r;
}
GpuModel::Regions GpuModel::value_regions(const Seq &s, int n) const {
    Regions r;
    const uint64_t row = 4ull * d_.n_head_kv * kv_words_;
    for (uint64_t pi = 0; pi * page_ < (uint64_t)n; pi++) r.emplace_back((uint64_t)s.pages[pi] * page_ * row, row * std::min<uint64_t>(page_, n - pi * page_));
    return r;
}

// Gather / scatter buffer ranges through a host buffer, many ranges per submission.
void GpuModel::read_regions(const Buffer &b, const Regions &regs, std::vector<uint8_t> &out) {
    if (!xfer_.valid()) xfer_ = dev_.alloc_host(32ull << 20);
    uint64_t total = 0;
    for (auto &r : regs) total += r.second;
    out.resize(total);
    uint64_t done = 0;
    std::vector<VkBufferCopy> batch;
    uint64_t fill = 0;
    auto flush = [&] {
        if (batch.empty()) return;
        Program p(dev_); p.copy_many(b, xfer_, batch); p.run();
        memcpy(out.data() + done, xfer_.mapped, fill);
        done += fill; fill = 0; batch.clear();
    };
    for (auto [off, bytes] : regs)
        while (bytes) {
            uint64_t n = std::min(bytes, xfer_.size - fill);
            batch.push_back({off, fill, n});
            fill += n; off += n; bytes -= n;
            if (fill == xfer_.size || batch.size() >= 4096) flush();
        }
    flush();
}
void GpuModel::write_regions(const Buffer &b, const Regions &regs, const uint8_t *data) {
    if (!xfer_.valid()) xfer_ = dev_.alloc_host(32ull << 20);
    uint64_t done = 0, fill = 0;
    std::vector<VkBufferCopy> batch;
    auto flush = [&] {
        if (batch.empty()) return;
        memcpy(xfer_.mapped, data + done, fill);
        Program p(dev_); p.copy_many(xfer_, b, batch); p.run();
        done += fill; fill = 0; batch.clear();
    };
    for (auto [off, bytes] : regs)
        while (bytes) {
            uint64_t n = std::min(bytes, xfer_.size - fill);
            batch.push_back({fill, off, n});
            fill += n; off += n; bytes -= n;
            if (fill == xfer_.size || batch.size() >= 4096) flush();
        }
    flush();
}

void GpuModel::state_save(std::vector<uint8_t> &out) {
    out.resize(state_size());
    if (recurrent_bytes_) { snap_save_->run(); memcpy(out.data(), snap_.mapped, recurrent_bytes_); }
    memcpy(out.data() + recurrent_bytes_, logits_.data(), 4 * logits_.size());
}
void GpuModel::state_load(const std::vector<uint8_t> &in) {
    if (in.size() != state_size()) throw std::runtime_error("internal: state of the wrong size");
    if (recurrent_bytes_) { memcpy(snap_.mapped, in.data(), recurrent_bytes_); snap_load_->run(); }
    memcpy(logits_.data(), in.data() + recurrent_bytes_, 4 * logits_.size());
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

GpuModel::Vec GpuModel::scratch_vec(uint64_t n) {
    Vec v{scratch(n), scratch((n + 15) / 16), Buffer{}, 0};
    if (wmma_) { v.h = dev_.alloc(2 * n + 4); dev_.fill_zero(v.h); }
    return v;
}

void GpuModel::alloc_activations() {
    const uint64_t B = batch_cap_, n = d_.n_embd, hd = d_.head_dim, nh = d_.n_head, nkv = d_.n_head_kv, qw = d_.q_has_gate ? 2 * hd : hd;
    const uint64_t S = d_.dn_head_dim, HV = d_.dn_v_heads, C = d_.dn_conv_dim(), n_seg_max = (uint64_t)n_ctx_ / page_;
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
            uint32_t stride = m.stride(c), groups = (rows + 63) / 64;
            std::vector<Binding> b{R(m.chunks[c]), R(x.v), R(x.s), W(y), R(bias)};
            if (m.fmt->needs_table) for (auto &t : tables_) if (t.first == m.fmt->type) b.push_back(R(t.second));
            if (use_wmma(m, n_tokens)) {
                dispatch_wmma(p, m, c, row0, x.h, y, n_tokens);
            } else if (batched) {
                std::string kernel = std::string("matmat") + (m.fmt->kernel + 6);   // "matvec_x" -> "matmat_x"
                PcMatmatQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, flags, n_tokens, m.n_in, m.n_in / 16, m.n_out};
                p.dispatch(dev_.pipeline(kernel), b, &pc, sizeof pc, groups, tiles);
            } else {
                PcMatvecQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, flags};
                p.dispatch(dev_.pipeline(m.fmt->kernel), b, &pc, sizeof pc, groups);
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

// The matrix-instruction kernels work on 256-column steps and 16-row tiles, and have no bias term.
bool GpuModel::use_wmma(const Matrix &m, uint32_t n_tokens) const {
    if (!wmma_ || n_tokens < k_wmma_min_tokens || !m.fmt || m.bias.valid() || m.n_in % 256 || m.n_out % 16) return false;
    return std::find(wmma_ok_.begin(), wmma_ok_.end(), m.fmt->type) != wmma_ok_.end();
}

void GpuModel::dispatch_wmma(Program &p, const Matrix &m, size_t c, uint32_t row0, const Buffer &xh, const Buffer &y, uint32_t n_tokens) {
    uint32_t rows = m.chunk_rows[c], stride = m.stride(c);
    std::string kernel = std::string(wmma_emu_ ? "wmmaemu" : "wmma") + (m.fmt->kernel + 6);   // "matvec_x" -> "wmma_x"
    PcMatmatQ pc{m.n_in / m.fmt->sb_cols, stride, rows, row0, 0, n_tokens, m.n_in, m.n_in / 16, m.n_out};
    std::vector<Binding> b{R(m.chunks[c]), R(xh), R(dummy_), W(y), R(dummy_)};
    if (m.fmt->needs_table) for (auto &t : tables_) if (t.first == m.fmt->type) b.push_back(R(t.second));
    p.dispatch(dev_.pipeline(kernel), b, &pc, sizeof pc, (rows + 63) / 64, (n_tokens + k_wmma_tile - 1) / k_wmma_tile);
}

// Half-precision copy of the first n elements of x (rounded up to whole token tiles by the caller).
void GpuModel::to_half(Program &p, const Vec &x, uint64_t n) {
    uint32_t nn = (uint32_t)n;
    p.dispatch(dev_.pipeline("to_f16"), {R(x.v), W(x.h)}, &nn, sizeof nn, (nn / 2 + 63) / 64);
}

// The matrix-instruction kernels cannot be tested without a device that has them, so they are
// checked here, on the device, before use: for one matrix of each weight format, the product with
// 16 random input vectors is computed by the portable batched kernel and by the matrix-instruction
// kernel, and the format is only enabled if the two agree.
void GpuModel::verify_wmma() {
    std::vector<const Matrix *> all;
    for (const Layer &l : layers_) for (const Matrix *m : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wz, &l.wbeta, &l.walpha, &l.ssm_out, &l.ffn_gate, &l.ffn_up, &l.ffn_down}) all.push_back(m);
    all.push_back(&output_);
    std::vector<uint32_t> seen;
    std::string ok_names, bad_names;
    const uint32_t N = k_wmma_tile;
    for (const Matrix *m : all) {
        if (!m->fmt || m->chunks.empty() || m->bias.valid() || m->n_in % 256 || m->n_out % 16) continue;
        if (std::find(seen.begin(), seen.end(), m->fmt->type) != seen.end()) continue;
        seen.push_back(m->fmt->type);
        const uint32_t rows = std::min<uint32_t>(m->chunk_rows[0], 256), n_in = m->n_in;   // the first rows are enough
        Matrix probe = *m; probe.chunks.resize(1); probe.chunk_stride = {m->stride(0)}; probe.chunk_rows = {rows}; probe.n_out = rows;
        std::vector<float> x((size_t)N * n_in), xs((size_t)N * n_in / 16), y1((size_t)N * rows), y2((size_t)N * rows);
        uint32_t seed = 4242;
        for (auto &v : x) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        for (size_t b = 0; b < xs.size(); b++) { float t = 0; for (int i = 0; i < 16; i++) t += x[16 * b + i]; xs[b] = t; }
        Vec tx{dev_.alloc(x.size() * 4), dev_.alloc(xs.size() * 4), dev_.alloc(x.size() * 2 + 4), 0};
        Buffer ta = dev_.alloc(y1.size() * 4), tb = dev_.alloc(y2.size() * 4);
        dev_.upload(tx.v, x.data(), x.size() * 4); dev_.upload(tx.s, xs.data(), xs.size() * 4);
        dev_.fill_zero(ta); dev_.fill_zero(tb);
        bool ok = false;
        try {
            Program p(dev_);
            to_half(p, tx, x.size());
            wmma_ = false; matmul(p, probe, tx, ta, N); wmma_ = true;   // portable batched kernel
            dispatch_wmma(p, probe, 0, 0, tx.h, tb, N);
            p.finish(); p.run();
            dev_.download(ta, y1.data(), y1.size() * 4); dev_.download(tb, y2.data(), y2.size() * 4);
            double num = 0, den = 0;
            for (size_t i = 0; i < y1.size(); i++) { double d = (double)y2[i] - y1[i]; num += d * d; den += (double)y1[i] * y1[i]; }
            ok = den > 0 && std::isfinite(num) && std::sqrt(num / den) < 5e-3;
        } catch (const std::exception &) { wmma_ = true; }
        const char *name = type_info(m->fmt->type)->name;
        if (ok) { wmma_ok_.push_back(m->fmt->type); ok_names += std::string(ok_names.empty() ? "" : ", ") + name; }
        else bad_names += std::string(bad_names.empty() ? "" : ", ") + name;
    }
    wmma_report_ = ok_names.empty() ? "none" : ok_names;
    if (!bad_names.empty()) wmma_report_ += " (failed the check, using the portable kernels: " + bad_names + ")";
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
    // Inputs of the matrix-instruction kernels: half-precision copies, for whole tiles of tokens.
    const bool halves = wmma_ && !wmma_ok_.empty() && N >= k_wmma_min_tokens;
    const uint64_t n16 = (N + k_wmma_tile - 1) / k_wmma_tile * k_wmma_tile;
    p.copy(embd_.sub(0, 4ull * N * n), a.x.sub(0, 4ull * N * n));
    // Normalise `groups` spans of `len` elements into y, with its 16-element block sums.
    auto norm = [&](const Buffer &xin, const Buffer &add, const Buffer &w, const Buffer &gt, const Vec &y, uint32_t len, uint32_t groups) {
        PcNorm pc{len, (add.valid() ? 1u : 0u) | (gt.valid() ? 2u : 0u) | 4u, d_.norm_eps};
        p.dispatch(p_norm, {W(xin), R(add.valid() ? add : dummy_), R(w), R(gt.valid() ? gt : dummy_), W(y.v), W(y.s)}, &pc, sizeof pc, groups);
    };
    for (int il = 0; il < d_.n_layer; il++) {
        Layer &l = layers_[il];
        norm(a.x, il > 0 ? a.a : Buffer{}, l.attn_norm, Buffer{}, a.h, n, N);
        if (halves) to_half(p, a.h, n16 * n);
        if (l.recurrent) {
            matmul(p, l.wqkv, a.h, a.mixed, N); matmul(p, l.wz, a.h, a.z, N); matmul(p, l.wbeta, a.h, a.beta, N); matmul(p, l.walpha, a.h, a.alpha, N);
            PcDnPrep pp{(uint32_t)(2 * HK), (uint32_t)d_.dn_conv, d_.norm_eps, (uint32_t)C};
            p.dispatch(p_dn_prep, {R(a.mixed), R(l.hist), R(l.conv_w), W(a.qkv)}, &pp, sizeof pp, C / S, N);
            PcDnHist ph{(uint32_t)C, (uint32_t)(d_.dn_conv - 1), N};
            p.dispatch(p_dn_hist, {R(a.mixed), W(l.hist)}, &ph, sizeof ph, (C + 63) / 64);
            PcDnUpdate pu{(uint32_t)HK, (uint32_t)S, N, (uint32_t)HV, (uint32_t)C};
            p.dispatch(p_dn_update, {R(a.qkv), R(a.alpha), R(a.beta), R(l.dt), R(l.a), W(l.state), W(a.dn_o)}, &pu, sizeof pu, HV);
            norm(a.dn_o, Buffer{}, l.ssm_norm, a.z, a.dn_on, S, N * HV);
            if (halves) to_half(p, a.dn_on, n16 * HV * S);
            matmul(p, l.ssm_out, a.dn_on, a.a, N);
        } else {
            matmul(p, l.wq, a.h, a.qfull, N); matmul(p, l.wk, a.h, a.kraw, N); matmul(p, l.wv, a.h, a.vraw, N);
            PcAttnPrep pp{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, (uint32_t)qw, (uint32_t)d_.rope_dim,
                          (l.has_qk_norm ? 1u : 0u) | (d_.q_has_gate ? 2u : 0u) | (d_.rope_half_split ? 4u : 0u), d_.norm_eps, stride_, page_shift_};
            p.dispatch(p_prep, {R(a.qfull), R(a.kraw), R(a.vraw), R(l.has_qk_norm ? l.qk_norm : dummy_), R(ctl_), W(a.q), W(a.gate), W(l.kc), W(l.vc), R(pt_)}, &pp, sizeof pp, std::max(nh, nkv), N);
            // All keys and values of the batch are in the cache now; each query attends up to its own position.
            for (uint32_t q0 = 0, sub = 0; q0 < N; q0 += q_batch_, sub++) {
                uint32_t nb = std::min<uint32_t>(q_batch_, N - q0);
                PcScore ps{(uint32_t)hd, stride_, stride_, 1.0f / std::sqrt((float)hd), (uint32_t)nh, q0, page_shift_};
                p.dispatch_indirect(p_score, {R(a.q), R(l.kc), R(ctl_), W(a.scores), R(pt_)}, &ps, sizeof ps, ind_.sub(32ull * sub, 12));
                PcSoft pf{stride_, (uint32_t)nh, q0};
                p.dispatch(p_soft, {R(ctl_), W(a.scores), W(a.zsum)}, &pf, sizeof pf, nh, nb);
                PcVsum pv{(uint32_t)hd, (uint32_t)nh, (uint32_t)nkv, stride_, page_, part_stride_, q0};
                p.dispatch_indirect(p_vsum, {R(l.vc), R(a.scores), R(ctl_), W(a.part), R(pt_)}, &pv, sizeof pv, ind_.sub(32ull * sub + 16, 12));
                PcCombine pcb{(uint32_t)hd, (uint32_t)nh, page_, d_.q_has_gate ? 1u : 0u, part_stride_, q0};
                p.dispatch(p_combine, {R(a.part), R(a.zsum), R(a.gate), R(ctl_), W(a.mix.v), W(a.mix.s)}, &pcb, sizeof pcb, nh, nb);
            }
            if (halves) to_half(p, a.mix, n16 * nh * hd);
            matmul(p, l.wo, a.mix, a.a, N);
        }
        norm(a.x, a.a, l.ffn_norm, Buffer{}, a.h, n, N);
        if (halves) to_half(p, a.h, n16 * n);
        matmul(p, l.ffn_gate, a.h, a.g, N); matmul(p, l.ffn_up, a.h, a.u, N);
        uint32_t nff = N * d_.n_ff;
        p.dispatch(p_swiglu, {R(a.g), R(a.u), W(a.act.v), W(a.act.s)}, &nff, sizeof nff, (nff / 16 + 63) / 64);
        if (halves) to_half(p, a.act, n16 * d_.n_ff);
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
    const int n_past = seqs_[active_].len;
    if (n_past + n > n_ctx_) throw std::runtime_error("context is full");
    take_pages((uint32_t)n_past, (uint32_t)n);
    // Embedding rows are decoded on the CPU; the table never occupies GPU memory unless it is
    // also the output projection.
    const TypeInfo *ti = type_info(d_.tok_embd->type);
    const uint64_t row_bytes = (uint64_t)(d_.n_embd / ti->block_elems) * ti->block_bytes;
    const uint32_t pos0 = (uint32_t)n_past;
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
        uint32_t v[8] = {last_pos / 64 + 1, (uint32_t)d_.n_head_kv, nb, 0, last_pos / page_ + 1, (uint32_t)d_.n_head_kv, nb, 0};
        memcpy(ind + 8 * sub, v, sizeof v);
    }
    program(n).run();
    seqs_[active_].len += n;
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
    const Seq &sq = seqs_[active_];
    StateHeader h{};
    memcpy(h.magic, k_state_magic, 8);
    h.version = 1; h.kv8 = kv8_; h.n_layer = d_.n_layer; h.n_attn = d_.n_attn_layers(); h.head_dim = d_.head_dim; h.n_head_kv = d_.n_head_kv;
    h.kv_words = kv_words_; h.n_embd = d_.n_embd; h.n_vocab = d_.n_vocab; h.n_past = sq.len; h.n_tokens = (uint32_t)tokens.size(); h.model_bytes = g_.file_size;
    f.write(&h, sizeof h);
    f.write(tokens.data(), tokens.size() * 4);
    f.write(logits_.data(), logits_.size() * 4);
    std::vector<uint8_t> buf;
    const Regions kr = key_regions(sq, sq.len), vr = value_regions(sq, sq.len);
    for (const Layer &l : layers_) {
        if (l.recurrent) {
            read_regions(l.state, {{0, l.state.size}}, buf); f.write(buf.data(), buf.size());
            read_regions(l.hist, {{0, l.hist.size}}, buf); f.write(buf.data(), buf.size());
            continue;
        }
        // each key plane's filled positions, then the value rows, in sequence order
        read_regions(l.kc, kr, buf); f.write(buf.data(), buf.size());
        read_regions(l.vc, vr, buf); f.write(buf.data(), buf.size());
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
    seq_truncate(0);
    if (h.n_past > 0) take_pages(0, h.n_past);
    Seq &sq = seqs_[active_];
    std::vector<uint8_t> buf;
    auto restore = [&](const Buffer &b, const Regions &r) {
        uint64_t total = 0;
        for (auto &x : r) total += x.second;
        buf.resize(total); f.read(buf.data(), total); write_regions(b, r, buf.data());
    };
    const Regions kr = key_regions(sq, (int)h.n_past), vr = value_regions(sq, (int)h.n_past);
    for (const Layer &l : layers_) {
        if (l.recurrent) { restore(l.state, {{0, l.state.size}}); restore(l.hist, {{0, l.hist.size}}); continue; }
        restore(l.kc, kr); restore(l.vc, vr);
    }
    sq.len = (int)h.n_past;
    return tokens;
}

}  // namespace gfx
