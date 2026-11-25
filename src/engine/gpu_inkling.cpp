// The GPU engine for Inkling 2 models. See core/inkling.h for the architecture; this file loads
// its weights and records its forward pass. Sequences, paging and state copies are shared with
// the other architectures (gpu_model.cpp).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/quant.h"
#include "gpu_model.h"
#include "pack.h"

namespace gfx {

namespace {
struct PcNorm { uint32_t n, flags; float eps; };
struct PcMv32 { uint32_t n_in, n_rows, x_stride, y_stride; };
struct PcMix { uint32_t n, n_states, out_stride; };
struct PcConv { uint32_t n_ch, nc, n_tokens, x_stride, flags, per_head, taps_stride; };
struct PcPrep { uint32_t hd, nh, nkv, nr, flags; float qk_clip, rel_clip, eps; uint32_t stride, page_shift, ring; };
struct PcTable { uint32_t nr, extent, nh; };
struct PcScore { uint32_t hd, stride, n_ctx; float scale; uint32_t nh, tq0, page_shift, extent, ring, window, n_batch; };
struct PcSoft { uint32_t n_ctx, nh, tq0; };
struct PcVsum { uint32_t hd, nh, nkv, n_ctx, seg, part_stride, tq0; };
struct PcCombine { uint32_t hd, nh, seg, flags, part_stride, tq0, n_seg; };
struct PcRoute { uint32_t n_expert, used; float scale; uint32_t flags; };
struct PcSitu { uint32_t nff; float beta, lb; };
struct PcEsum { uint32_t n, used, flags; };
struct PcExpert { uint32_t n_sb, stride, n_rows, row0, flags, used, x_stride, xs_stride; };
constexpr uint32_t k_ring_seg = 256;   // places of a sliding-window cache summed per workgroup
}  // namespace

struct GpuModel::Ink {
    InklingDesc d;
    struct Layer {
        bool window = false;
        Matrix wq, wk, wv, wo, wgate, rel;
        Buffer sink, k_taps, v_taps, q_clip, rel_clip, rel_bias, k_conv, v_conv, attn_conv, ffn_conv;
        Buffer attn_norm, q_norm, k_norm, v_norm, attn_query, ffn_query, ffn_norm, router, router_bias;
        Buffer gate_up, down, gate_up_scale, down_scale;   // all experts of the layer, one after another
        const PackFormat *gu_fmt = nullptr, *dn_fmt = nullptr;
        Buffer msg_query, ctrl_down[4], ctrl_gain[4], ctrl_forget[4];
        uint32_t extent = 0;
    };
    std::vector<Layer> layers;
    // A cache slot: a paged cache (layers that attend to everything) or a ring that keeps the
    // sliding window, and the convolution history.
    struct Slot { Buffer kc, vc, hist; };
    std::vector<Slot> slots;
    Buffer tok_norm, output_norm, output_query;
    Matrix output;
    uint32_t ring = 0, ring_segs = 0, table_stride = 0;
    // activations (B = batch capacity)
    Buffer xraw, x, cur, blk[5], pass_out[2], m, msg, ci, cl, cg, cf, in[3], qraw, kraw, vraw, rraw, sink, gate, ktaps, vtaps, q, r, table, scores, zsum, part;
    Buffer a, f, fc, rl, ids, wts, gu, act, acts, dn, ff, logits;
    Vec h[3], hf, mix, h_last;
};

GpuModel::~GpuModel() = default;

uint64_t GpuModel::bytes_per_token(const InklingDesc &d, bool kv8) {
    uint64_t words = kv8 ? d.head_dim / 4 + d.head_dim / 64 : d.head_dim / 2;
    return (uint64_t)d.n_full_slots() * 2 * 4 * d.n_head_kv * words;
}

GpuModel::GpuModel(Device &dev, const GGUF &g, const InklingDesc &desc, int n_ctx, bool kv8, uint64_t reserve_bytes, int batch, bool allow_wmma, const ExpertFilter &want_expert)
    : dev_(dev), g_(g), n_ctx_(n_ctx), kv8_(kv8) {
    ink_ = std::make_shared<Ink>();
    Ink &k = *ink_;
    k.d = desc;
    const InklingDesc &D = k.d;
    // The parts of the description the shared code reads.
    d_.name = D.name; d_.arch_name = g.architecture; d_.n_vocab = D.n_vocab; d_.n_embd = D.n_embd; d_.n_layer = D.n_layer; d_.n_ff = D.n_ff; d_.n_ctx_train = D.n_ctx_train;
    d_.n_head = D.n_head; d_.n_head_kv = D.n_head_kv; d_.head_dim = D.head_dim; d_.norm_eps = D.norm_eps; d_.tok_embd = D.tok_embd;
    ink_tau_alpha_ = D.log_alpha; ink_tau_floor_ = (float)D.log_floor; ink_softcap_ = D.softcap;
    if (D.n_conv > 8 || D.n_expert > 256 || D.n_expert_used > 32 || D.n_rel > D.head_dim || D.n_ff % 16) throw std::runtime_error("Inkling dimensions outside what the kernels were written for");
    init_batching(batch, allow_wmma);

    const uint32_t n = D.n_embd, ne = D.n_expert, nff = D.n_ff;
    auto ones = [&](size_t count) { std::vector<float> v(count, 1.0f); Buffer b = dev_.alloc(count * 4); dev_.upload(b, v.data(), count * 4); return b; };
    // All experts of a layer in one buffer, each in the plane-major layout of its format.
    auto load_experts = [&](const TensorInfo *w, int layer, const PackFormat *&fmt) {
        const uint32_t n_in = (uint32_t)w->dim(0), n_out = (uint32_t)w->dim(1);
        fmt = pack_format(w->type);
        if (!fmt || n_in % fmt->sb_cols || fmt->needs_table) throw std::runtime_error("tensor " + w->name + ": this weight format is not supported for expert matrices yet");
        const uint32_t stride = (n_out + 63) / 64 * 64;
        const uint64_t words = (uint64_t)(n_in / fmt->sb_cols) * fmt->sb_words * stride;
        if (words * 4 * ne > dev_.caps().max_storage_range) throw std::runtime_error("tensor " + w->name + " is too large for one buffer on this device");
        Buffer b = dev_.alloc(words * 4 * ne);
        std::vector<uint32_t> buf(words);
        for (uint32_t e = 0; e < ne; e++) {
            if (want_expert && !want_expert(layer, (int)e)) continue;
            pack_rows(*fmt, w->data, n_in, e * n_out, n_out, stride, buf.data());
            dev_.upload(b.sub(e * words * 4, words * 4), buf.data(), words * 4);
        }
        weight_bytes_ += b.size;
        return b;
    };
    k.layers.resize(D.n_layer);
    for (int i = 0; i < D.n_layer; i++) {
        const InklingLayer &ld = D.layers[i];
        Ink::Layer &l = k.layers[i];
        l.window = ld.window; l.extent = (uint32_t)ld.rel_bias->dim(1);
        l.wq = load_matrix(ld.wq); l.wk = load_matrix(ld.wk); l.wv = load_matrix(ld.wv); l.wo = load_matrix(ld.wo); l.wgate = load_matrix(ld.wgate); l.rel = load_matrix(ld.rel);
        l.sink = load_vector(ld.sink); l.q_clip = load_vector(ld.q_clip); l.rel_clip = load_vector(ld.rel_clip); l.rel_bias = load_vector(ld.rel_bias);
        l.k_conv = load_vector(ld.k_conv); l.v_conv = load_vector(ld.v_conv); l.attn_conv = load_vector(ld.attn_conv); l.ffn_conv = load_vector(ld.ffn_conv);
        if (ld.window) l.attn_norm = load_vector(ld.attn_norm);
        else {
            l.q_norm = load_vector(ld.q_norm); l.k_norm = load_vector(ld.k_norm); l.v_norm = load_vector(ld.v_norm);
            l.k_taps = load_vector(ld.k_taps); l.v_taps = load_vector(ld.v_taps);
        }
        if (ld.attn_query) l.attn_query = load_vector(ld.attn_query);
        l.ffn_query = load_vector(ld.ffn_query); l.ffn_norm = load_vector(ld.ffn_norm); l.router = load_vector(ld.router);
        if (ld.router_bias) l.router_bias = load_vector(ld.router_bias);
        l.gate_up = load_experts(ld.gate_up, i, l.gu_fmt); l.down = load_experts(ld.down, i, l.dn_fmt);
        l.gate_up_scale = ld.gate_up_scale ? load_vector(ld.gate_up_scale) : ones(ne);
        l.down_scale = ld.down_scale ? load_vector(ld.down_scale) : ones(ne);
        if (ld.msg_query) {
            l.msg_query = load_vector(ld.msg_query);
            for (int c = 0; c < 4; c++) { l.ctrl_down[c] = load_vector(ld.ctrl_down[c]); l.ctrl_gain[c] = load_vector(ld.ctrl_gain[c]); l.ctrl_forget[c] = load_vector(ld.ctrl_forget[c]); }
        }
    }
    k.tok_norm = load_vector(D.tok_norm); k.output_norm = load_vector(D.output_norm); k.output_query = load_vector(D.output_query);
    k.output = load_matrix(D.output);

    // Caches: paged for the slots that attend to everything, rings for the sliding windows.
    const uint64_t B = batch_cap_, HQ = (uint64_t)D.n_head * D.head_dim, HK = (uint64_t)D.n_head_kv * D.head_dim, RN = (uint64_t)D.n_head * D.n_rel;
    const uint64_t per_pos = 4ull * D.n_head_kv * kv_words_;
    k.ring = ((uint32_t)D.window + (uint32_t)B + 63) / 64 * 64;
    k.ring_segs = (k.ring + k_ring_seg - 1) / k_ring_seg;
    const uint64_t n_window_slots = D.n_slots() - D.n_full_slots();
    const uint64_t fixed = n_window_slots * 2 * per_pos * k.ring + B * 4 * (30 * n + 6 * HQ + 4ull * D.n_expert_used * 2 * n) + 4ull * D.n_vocab_padded + 2 * k_scores_budget + (32ull << 20);
    init_context(reserve_bytes, fixed, D.n_full_slots());
    k.slots.resize(D.n_slots());
    size_t ci = 0;
    for (int s = 0; s < D.n_slots(); s++) {
        Ink::Slot &sl = k.slots[s];
        sl.hist = dev_.alloc(4ull * D.conv_channels() * (D.n_conv - 1));
        state_bufs_.push_back(sl.hist); state_bytes_ += sl.hist.size;
        if (D.layers[D.slot_layer(s)].window) {
            sl.kc = dev_.alloc(per_pos * k.ring); sl.vc = dev_.alloc(per_pos * k.ring);
            state_bufs_.push_back(sl.kc); state_bufs_.push_back(sl.vc); state_bytes_ += sl.kc.size + sl.vc.size;
        } else { sl.kc = caches_[ci].first; sl.vc = caches_[ci].second; ci++; }
    }

    // Activations.
    k.table_stride = (uint32_t)std::max(D.window, D.rel_extent) + 1;
    k.xraw = scratch(B * n); k.x = scratch(B * n); k.cur = scratch(B * n);
    for (Buffer &b : k.blk) b = scratch(B * n);
    for (Buffer &b : k.pass_out) b = scratch(B * n);
    k.m = scratch(3 * B * n); k.msg = scratch(4 * B * n); k.ci = scratch(B * 2 * n); k.cl = scratch(B * D.gate_rank + 64); k.cg = scratch(B * n); k.cf = scratch(B * n);
    for (Buffer &b : k.in) b = scratch(B * n);
    for (Vec &v : k.h) v = scratch_vec(B * n);
    k.hf = scratch_vec(B * n); k.mix = scratch_vec(B * HQ); k.h_last = scratch_vec(n);
    k.qraw = scratch(B * HQ); k.kraw = scratch(B * HK); k.vraw = scratch(B * HK); k.rraw = scratch(B * RN); k.sink = scratch(B * D.n_head); k.gate = scratch(B * HQ);
    k.ktaps = scratch(B * D.n_conv * D.n_head_kv); k.vtaps = scratch(B * D.n_conv * D.n_head_kv);
    k.q = scratch(B * HQ); k.r = scratch(B * RN); k.table = scratch(B * D.n_head * k.table_stride); k.zsum = scratch(B * D.n_head);
    uint64_t budget = k_scores_budget;
    if (const char *e = getenv("GFX_SCORES_BUDGET")) budget = strtoull(e, nullptr, 10);   // testing aid
    const uint64_t per_query = 4ull * D.n_head * std::max<uint64_t>(stride_, k.ring), limit = std::min<uint64_t>(budget, dev_.caps().max_storage_range);
    q_batch_ = (int)std::max<uint64_t>(1, std::min<uint64_t>(batch_, limit / per_query));
    part_stride_ = (uint32_t)(std::max<uint64_t>((uint64_t)n_ctx_ / page_, k.ring_segs) * HQ);
    k.scores = dev_.alloc(per_query * q_batch_);
    k.part = dev_.alloc(4ull * part_stride_ * q_batch_);
    const uint64_t U = D.n_expert_used;
    k.a = scratch(B * n); k.f = scratch(B * n); k.fc = scratch(B * n); k.rl = scratch(B * ne); k.ids = scratch(B * U); k.wts = scratch(B * U);
    k.gu = scratch(B * U * 2 * nff); k.act = scratch(B * U * nff); k.acts = scratch(B * U * nff / 16); k.dn = scratch(B * U * n); k.ff = scratch(B * n);
    k.logits = scratch(D.n_vocab_padded);
    if (wmma_) verify_wmma();
    finish_init();
}

std::vector<std::vector<int>> GpuModel::last_experts() { return {}; }

// The forward pass for N tokens at consecutive positions.
void GpuModel::record_inkling(Program &p, uint32_t N) {
    Ink &k = *ink_;
    const InklingDesc &D = k.d;
    const uint32_t n = D.n_embd, hd = D.head_dim, nh = D.n_head, nkv = D.n_head_kv, nr = D.n_rel, nc = D.n_conv, B = batch_cap_;
    const uint32_t HQ = nh * hd, HK = nkv * hd, G = nh / nkv, ne = D.n_expert, U = D.n_expert_used, nff = D.n_ff, rank = D.gate_rank;
    const uint64_t vec = 4ull * B * n;   // bytes of one activation of every token
    const bool halves = wmma_ && !wmma_ok_.empty() && N >= k_wmma_min_tokens;
    const uint64_t n16 = (N + k_wmma_tile - 1) / k_wmma_tile * k_wmma_tile;

    const Pipeline *p_norm = dev_.pipeline("norm"), *p_mv32 = dev_.pipeline("matvec_f32"), *p_mix = dev_.pipeline("ink_mix"), *p_ctl_in = dev_.pipeline("ink_ctl_in"),
                   *p_ctl_out = dev_.pipeline("ink_ctl_out"), *p_conv = dev_.pipeline("ink_conv"), *p_table = dev_.pipeline("ink_table"), *p_route = dev_.pipeline("ink_route"),
                   *p_situ = dev_.pipeline("ink_situ"), *p_esum = dev_.pipeline("ink_esum");
    const Pipeline *p_prep = dev_.pipeline(kv8_ ? "ink_prep_kv8" : "ink_prep", {hd});
    const Pipeline *p_score = dev_.pipeline(kv8_ ? "ink_score_kv8" : "ink_score", {G}), *p_wscore = dev_.pipeline(kv8_ ? "ink_wscore_kv8" : "ink_wscore", {G});
    const Pipeline *p_soft = dev_.pipeline("ink_soft"), *p_wsoft = dev_.pipeline("ink_wsoft");
    const uint32_t vwords = kv8_ ? hd / 4 : hd / 2;
    const Pipeline *p_vsum = dev_.pipeline(kv8_ ? "attn_vsum_kv8" : "attn_vsum", {vwords, G}), *p_wvsum = dev_.pipeline(kv8_ ? "ink_wvsum_kv8" : "ink_wvsum", {vwords, G});
    const Pipeline *p_combine = dev_.pipeline("attn_combine", {hd});

    // y = rms_norm(x) * w for N vectors of n elements, with block sums when a matrix product follows.
    auto norm = [&](const Buffer &x, const Buffer &w, const Vec &y, uint32_t count) {
        PcNorm pc{n, 4u, D.norm_eps};
        p.dispatch(p_norm, {R(x), R(dummy_), R(w), R(dummy_), W(y.v), W(y.s)}, &pc, sizeof pc, count);
        if (halves && count > 1) to_half(p, y, n16 * n);
    };
    auto mv32 = [&](const Buffer &w, uint32_t n_in, uint32_t rows, const Buffer &x, uint32_t xs, const Buffer &y, uint32_t ys) {
        PcMv32 pc{n_in, rows, xs, ys};
        p.dispatch(p_mv32, {R(w), R(x), W(y)}, &pc, sizeof pc, (rows + 63) / 64, N);
    };
    // Depth mixing: out[q][token] for nq queries (out holds B vectors per query).
    auto mix = [&](const std::vector<Buffer> &states, const Buffer &query, uint32_t nq, const Buffer &out) {
        if (states.size() > 5) throw std::runtime_error("internal: too many residual streams to mix");
        std::vector<Binding> b;
        for (size_t i = 0; i < 5; i++) b.push_back(R(i < states.size() ? states[i] : dummy_));
        b.push_back(R(query)); b.push_back(W(out));
        PcMix pc{n, (uint32_t)states.size(), B * n};
        p.dispatch(p_mix, b, &pc, sizeof pc, nq, N);
    };
    // Message controller c of a looped layer: out = (1 - forget) * cur + gain * message.
    auto control = [&](const Buffer &cur, const Buffer &message, const Ink::Layer &l, int c, const Buffer &out) {
        uint32_t pn = n;
        p.dispatch(p_ctl_in, {R(cur), R(message), W(k.ci)}, &pn, sizeof pn, N);
        mv32(l.ctrl_down[c], 2 * n, rank, k.ci, 2 * n, k.cl, rank);
        mv32(l.ctrl_gain[c], rank, n, k.cl, rank, k.cg, n);
        mv32(l.ctrl_forget[c], rank, n, k.cl, rank, k.cf, n);
        uint32_t count = N * n;
        p.dispatch(p_ctl_out, {R(cur), R(message), R(k.cg), R(k.cf), W(out)}, &count, sizeof count, (count + 63) / 64);
    };
    auto conv = [&](const Buffer &x, uint32_t channels, const Buffer &w, const Ink::Slot &sl, uint32_t offset, const Buffer *taps, uint32_t heads) {
        PcConv pc{channels, nc, N, channels, taps ? 1u : 0u, channels / heads, nc * heads};
        p.dispatch(p_conv, {W(x), R(w), W(sl.hist.sub(4ull * offset * (nc - 1), 4ull * channels * (nc - 1))), R(taps ? *taps : dummy_)}, &pc, sizeof pc, (channels + 63) / 64);
    };
    auto add = [&](const Buffer &a, const Buffer &b, const Buffer &y) {
        PcEsum pc{N * n, 0, 1u};
        p.dispatch(p_esum, {R(a), R(b), W(y)}, &pc, sizeof pc, (N * n + 63) / 64);
    };

    // One layer in cache slot `slot`. x: the stream (updated in place). blocks: the inputs of the
    // blocks so far. messages: of a later loop pass, four vectors per token.
    auto layer = [&](int slot, int ip, std::vector<Buffer> &blocks, const Buffer &block_store, const Buffer *messages) {
        const Ink::Layer &l = k.layers[ip];
        const Ink::Slot &sl = k.slots[slot];
        std::vector<Buffer> states = blocks;
        states.push_back(k.x);

        // Inputs of the query, key and value projections.
        const uint32_t nq = l.window ? 1 : 3;
        Buffer inp[3] = {k.x, k.x, k.x};
        if (l.attn_query.valid()) {
            mix(states, l.attn_query, nq, k.m);
            for (int j = 0; j < 3; j++) inp[j] = k.m.sub((l.window ? 0 : j) * vec, vec);
        }
        if (messages) for (int j = 0; j < 3; j++) { control(inp[j], messages->sub(j * vec, vec), l, j, k.in[j]); inp[j] = k.in[j]; }
        const Vec *hq = &k.h[0], *hk = &k.h[0], *hv = &k.h[0];
        if (l.window && !messages) norm(inp[0], l.attn_norm, k.h[0], N);
        else {
            norm(inp[0], l.window ? l.attn_norm : l.q_norm, k.h[0], N);
            norm(inp[1], l.window ? l.attn_norm : l.k_norm, k.h[1], N);
            norm(inp[2], l.window ? l.attn_norm : l.v_norm, k.h[2], N);
            hk = &k.h[1]; hv = &k.h[2];
        }
        matmul(p, l.wq, *hq, k.qraw, N); matmul(p, l.wk, *hk, k.kraw, N); matmul(p, l.wv, *hv, k.vraw, N);
        matmul(p, l.rel, *hq, k.rraw, N); matmul(p, l.wgate, *hq, k.gate, N);
        mv32(l.sink, n, nh, hq->v, n, k.sink, nh);
        if (!l.window) { mv32(l.k_taps, n, nc * nkv, hk->v, n, k.ktaps, nc * nkv); mv32(l.v_taps, n, nc * nkv, hv->v, n, k.vtaps, nc * nkv); }
        conv(k.kraw, HK, l.k_conv, sl, 0, l.window ? nullptr : &k.ktaps, nkv);
        conv(k.vraw, HK, l.v_conv, sl, HK, l.window ? nullptr : &k.vtaps, nkv);

        PcPrep pp{hd, nh, nkv, nr, (l.window ? 2u : 1u), D.qk_clip, D.rel_clip, D.norm_eps, l.window ? k.ring : stride_, page_shift_, k.ring};
        p.dispatch(p_prep, {R(k.qraw), R(k.kraw), R(k.vraw), R(k.rraw), R(l.q_clip), R(l.rel_clip), R(ctl_), W(k.q), W(k.r), W(sl.kc), W(sl.vc), R(pt_)}, &pp, sizeof pp, std::max(nh, nkv), N);
        PcTable pt{nr, l.extent, nh};
        p.dispatch(p_table, {R(l.rel_bias), R(k.r), R(k.sink), W(k.table)}, &pt, sizeof pt, (l.extent + 1 + 63) / 64, nh, N);

        // All keys and values of the batch are cached now; each query attends up to its own position.
        const float scale = 1.0f / std::sqrt((float)hd);
        for (uint32_t q0 = 0, sub = 0; q0 < N; q0 += q_batch_, sub++) {
            const uint32_t nb = std::min<uint32_t>(q_batch_, N - q0);
            if (l.window) {
                PcScore ps{hd, k.ring, k.ring, scale, nh, q0, 0, l.extent, k.ring, (uint32_t)D.window, N};
                p.dispatch(p_wscore, {R(k.q), R(sl.kc), R(ctl_), W(k.scores), R(dummy_), R(k.table)}, &ps, sizeof ps, k.ring / 64, nkv, nb);
                PcSoft pf{k.ring, nh, q0};
                p.dispatch(p_wsoft, {R(ctl_), W(k.scores), W(k.zsum)}, &pf, sizeof pf, nh, nb);
                PcVsum pv{hd, nh, nkv, k.ring, k_ring_seg, part_stride_, q0};
                p.dispatch(p_wvsum, {R(sl.vc), R(k.scores), R(ctl_), W(k.part)}, &pv, sizeof pv, k.ring_segs, nkv, nb);
                PcCombine pcb{hd, nh, k_ring_seg, 1u, part_stride_, q0, k.ring_segs};
                p.dispatch(p_combine, {R(k.part), R(k.zsum), R(k.gate), R(ctl_), W(k.mix.v), W(k.mix.s)}, &pcb, sizeof pcb, nh, nb);
            } else {
                PcScore ps{hd, stride_, stride_, scale, nh, q0, page_shift_, l.extent, 0, 0, N};
                p.dispatch_indirect(p_score, {R(k.q), R(sl.kc), R(ctl_), W(k.scores), R(pt_), R(k.table)}, &ps, sizeof ps, ind_.sub(32ull * sub, 12));
                PcSoft pf{stride_, nh, q0};
                p.dispatch(p_soft, {R(ctl_), W(k.scores), W(k.zsum)}, &pf, sizeof pf, nh, nb);
                PcVsum pv{hd, nh, nkv, stride_, page_, part_stride_, q0};
                p.dispatch_indirect(p_vsum, {R(sl.vc), R(k.scores), R(ctl_), W(k.part), R(pt_)}, &pv, sizeof pv, ind_.sub(32ull * sub + 16, 12));
                PcCombine pcb{hd, nh, page_, 1u, part_stride_, q0, 0};
                p.dispatch(p_combine, {R(k.part), R(k.zsum), R(k.gate), R(ctl_), W(k.mix.v), W(k.mix.s)}, &pcb, sizeof pcb, nh, nb);
            }
        }
        if (halves) to_half(p, k.mix, n16 * HQ);
        matmul(p, l.wo, k.mix, k.a, N);
        conv(k.a, n, l.attn_conv, sl, 2 * HK, nullptr, 1);

        // The first layer of a block starts a new residual stream; its input joins the mixed streams.
        if (ip % D.block == 0) {
            p.copy(k.x.sub(0, 4ull * N * n), block_store.sub(0, 4ull * N * n));
            blocks.push_back(block_store);
            p.copy(k.a.sub(0, 4ull * N * n), k.cur.sub(0, 4ull * N * n));
        } else add(k.x, k.a, k.cur);

        // Mixture of experts.
        states = blocks;
        states.push_back(k.cur);
        mix(states, l.ffn_query, 1, k.f);
        const Buffer *fin = &k.f;
        if (messages) { control(k.f, messages->sub(3 * vec, vec), l, 3, k.fc); fin = &k.fc; }
        norm(*fin, l.ffn_norm, k.hf, N);
        mv32(l.router, n, ne, k.hf.v, n, k.rl, ne);
        PcRoute pr{ne, U, D.expert_scale, l.router_bias.valid() ? 1u : 0u};
        p.dispatch(p_route, {R(k.rl), R(l.router_bias.valid() ? l.router_bias : dummy_), W(k.ids), W(k.wts)}, &pr, sizeof pr, N);
        {
            const uint32_t rows = 2 * nff, stride = (rows + 63) / 64 * 64;
            PcExpert pe{n / l.gu_fmt->sb_cols, stride, rows, 0, 0, U, n, n / 16};
            std::string kernel = std::string("expert") + (l.gu_fmt->kernel + 6);
            p.dispatch(dev_.pipeline(kernel), {R(l.gate_up), R(k.hf.v), R(k.hf.s), W(k.gu), R(l.gate_up_scale), R(dummy_), R(k.ids)}, &pe, sizeof pe, stride / 64, U, N);
        }
        PcSitu psitu{nff, D.situ_beta, D.situ_linear_beta};
        p.dispatch(p_situ, {R(k.gu), W(k.act), W(k.acts)}, &psitu, sizeof psitu, (nff + 63) / 64, N * U);
        {
            const uint32_t stride = (n + 63) / 64 * 64;
            PcExpert pe{nff / l.dn_fmt->sb_cols, stride, n, 0, 2u, U, nff, nff / 16};
            std::string kernel = std::string("expert") + (l.dn_fmt->kernel + 6);
            p.dispatch(dev_.pipeline(kernel), {R(l.down), R(k.act), R(k.acts), W(k.dn), R(l.down_scale), R(dummy_), R(k.ids)}, &pe, sizeof pe, stride / 64, U, N);
        }
        PcEsum pes{n, U, 0};
        p.dispatch(p_esum, {R(k.dn), R(k.wts), W(k.ff)}, &pes, sizeof pes, (n + 63) / 64, N);
        conv(k.ff, n, l.ffn_conv, sl, 2 * HK + n, nullptr, 1);
        add(k.cur, k.ff, k.x);
    };

    p.copy(embd_.sub(0, 4ull * N * n), k.xraw.sub(0, 4ull * N * n));
    {
        PcNorm pc{n, 0u, D.norm_eps};
        p.dispatch(p_norm, {R(k.xraw), R(dummy_), R(k.tok_norm), R(dummy_), W(k.x), W(dummy_)}, &pc, sizeof pc, N);
    }
    std::vector<Buffer> blocks;
    const int start = D.loop_start, end = D.loop_end;
    // Where the input of each block is kept: block b of the layers, with two copies for the
    // blocks inside the loop, which the next pass still reads while this one writes its own.
    auto store = [&](int ip, int pass) -> const Buffer & {
        const int b = ip / D.block, first_loop_block = start / D.block;
        if (D.loop_count > 1 && ip >= start && ip < end && b > first_loop_block) return pass % 2 ? k.blk[4] : k.blk[2];
        return k.blk[std::min(b, 3)];
    };
    if (D.loop_count <= 1) {
        if (D.n_layer / D.block > 4) throw std::runtime_error("internal: more residual blocks than this engine keeps");
        for (int il = 0; il < D.n_layer; il++) layer(il, il, blocks, k.blk[il / D.block], nullptr);
    } else {
        if (start != D.block || end != 3 * D.block || D.n_layer != 4 * D.block) throw std::runtime_error("this loop layout is not supported on the GPU path yet");
        for (int il = 0; il < start; il++) layer(il, il, blocks, k.blk[0], nullptr);
        // Every pass starts from the output of the layers before the loop, which is also the
        // input of the loop's first block (kept in blk[1]).
        std::vector<Buffer> before = blocks;
        for (int pass = 0; pass < D.loop_count; pass++) {
            std::vector<Buffer> bank;
            if (pass > 0) {
                bank.push_back(store(start + D.block, pass - 1));   // the previous pass's blocks after its first
                bank.push_back(k.pass_out[(pass - 1) % 2]);         // and its output
                p.copy(k.blk[1].sub(0, 4ull * N * n), k.x.sub(0, 4ull * N * n));
                blocks = before;
            }
            for (int ip = start; ip < end; ip++) {
                if (pass == 0) { layer(ip, ip, blocks, store(ip, pass), nullptr); continue; }
                const int slot = D.n_layer + (pass - 1) * (end - start) + (ip - start);
                mix(bank, k.layers[ip].msg_query, 4, k.msg);
                layer(slot, ip, blocks, store(ip, pass), &k.msg);
            }
            if (pass + 1 < D.loop_count) p.copy(k.x.sub(0, 4ull * N * n), k.pass_out[pass % 2].sub(0, 4ull * N * n));
        }
        for (int il = end; il < D.n_layer; il++) layer(il, il, blocks, k.blk[3], nullptr);
    }
    // Output: mix the streams once more; only the last token's logits are needed.
    std::vector<Buffer> states = blocks;
    states.push_back(k.x);
    mix(states, k.output_query, 1, k.f);
    norm(k.f.sub(4ull * (N - 1) * n, 4ull * n), k.output_norm, k.h_last, 1);
    matmul(p, k.output, k.h_last, k.logits, 1);
    p.copy(k.logits, logits_host_);
}

}  // namespace gfx
