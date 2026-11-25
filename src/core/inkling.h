// Inkling 2 (architectures "inkling2" and "inkling2_loop"): model description and a plain CPU
// reference of its forward pass.
//
// The architecture, in short. Every layer is attention followed by a mixture-of-experts
// feed-forward, but with several parts no other supported model has:
//   - No rotary positions. Position enters through a learned bias on the attention scores, looked
//     up by distance, and each query can also send weight to a "sink" instead of any position.
//   - Most layers attend to a sliding window of the last 1,024 positions; every sixth attends to
//     everything. Only those keep a cache that grows with the context.
//   - Keys, values, the attention output and the feed-forward output each pass through a short
//     causal convolution over the last four tokens, which needs a little per-layer state.
//   - Layers are grouped in blocks of six. A block's first layer starts a new residual stream,
//     and layer inputs are attention-weighted mixes of the streams so far ("depth mixing").
//   - The feed-forward picks 8 of 128 experts per token.
//   - In the looped variant a run of middle layers is executed several times; later passes get
//     messages from the previous pass, gated into each layer's inputs. Each pass has its own
//     caches, so the model has more cache slots than layers.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace gfx {

class ThreadPool;

struct InklingLayer {
    bool window = false;     // attends to a sliding window (otherwise to everything)
    const TensorInfo *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *wgate = nullptr;
    const TensorInfo *q_clip = nullptr, *sink = nullptr, *rel = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    const TensorInfo *k_conv = nullptr, *v_conv = nullptr, *attn_conv = nullptr, *ffn_conv = nullptr;
    const TensorInfo *attn_norm = nullptr;                                      // windowed layers
    const TensorInfo *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;   // full layers
    const TensorInfo *k_taps = nullptr, *v_taps = nullptr;                      // full layers: input-dependent convolution taps
    const TensorInfo *attn_query = nullptr, *ffn_query = nullptr;               // depth-mixing queries
    const TensorInfo *ffn_norm = nullptr, *router = nullptr, *router_bias = nullptr;
    const TensorInfo *gate_up = nullptr, *gate_up_scale = nullptr, *down = nullptr, *down_scale = nullptr;
    // looped layers: one message query and four controllers (for the q, k, v and feed-forward inputs)
    const TensorInfo *msg_query = nullptr, *ctrl_down[4] = {}, *ctrl_gain[4] = {}, *ctrl_forget[4] = {};
};

struct InklingDesc {
    std::string name;
    int n_vocab = 0, n_vocab_padded = 0, n_embd = 0, n_layer = 0, n_ctx_train = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    float norm_eps = 1e-6f, qk_clip = 0, rel_clip = 0;
    int n_rel = 0, rel_extent = 0, window = 0, n_conv = 0, block = 0;
    int log_floor = 0; float log_alpha = 0;
    int n_expert = 0, n_expert_used = 0, n_ff = 0;
    float expert_scale = 1, situ_beta = 0, situ_linear_beta = 0, softcap = 0;
    int loop_start = 0, loop_end = 0, loop_count = 1, gate_rank = 0;
    const TensorInfo *tok_embd = nullptr, *tok_norm = nullptr, *output = nullptr, *output_norm = nullptr, *output_query = nullptr;
    std::vector<InklingLayer> layers;

    // Cache slots: one per layer, plus one per looped layer for every pass after the first.
    int n_slots() const { return n_layer + (loop_count - 1) * (loop_end - loop_start); }
    int slot_layer(int slot) const { return slot < n_layer ? slot : loop_start + (slot - n_layer) % (loop_end - loop_start); }
    int conv_channels() const { return 2 * n_head_kv * head_dim + 2 * n_embd; }
    int n_full_slots() const { int n = 0; for (int s = 0; s < n_slots(); s++) n += !layers[slot_layer(s)].window; return n; }
};

bool is_inkling(const GGUF &g);
// Throws std::runtime_error naming what is missing or unexpected.
InklingDesc describe_inkling(const GGUF &g);

// One token at a time, scalar, single precision. Weights are decoded from the file as they are
// used (the experts alone would not fit in memory decoded).
class InklingRef {
  public:
    InklingRef(const GGUF &g, const InklingDesc &desc, int n_ctx, int n_threads);
    ~InklingRef();
    void reset();
    int n_past() const { return n_past_; }
    int n_ctx() const { return n_ctx_; }
    // Test aid: continue from position n as if n all-zero cache entries preceded it.
    void skip_to(int n);
    const std::vector<float> &eval(const int32_t *tokens, int n);
    // The experts the last token used, per cache slot (for comparing against the GPU path).
    const std::vector<std::vector<int>> &last_experts() const { return experts_; }

  private:
    using Vec = std::vector<float>;
    const float *full(const TensorInfo *t);
    void matvec(const TensorInfo *w, const float *x, float *y, uint64_t row0 = 0, uint64_t rows = 0, float scale = 1.0f);
    Vec mix(const std::vector<Vec> &blocks, const Vec &cur, const TensorInfo *query, int n_query);
    Vec control(const Vec &cur, const float *message, const InklingLayer &l, int ctrl);
    void convolve(float *cur, int n, const TensorInfo *weight, const float *taps, int n_heads, int slot, int offset);
    Vec layer(int slot, int ip, const Vec &in, std::vector<Vec> &blocks, const Vec *messages);
    void eval_one(int32_t token);

    const GGUF &g_;
    InklingDesc d_;
    int n_ctx_, n_past_ = 0;
    std::unique_ptr<ThreadPool> pool_;
    std::unordered_map<const TensorInfo *, std::vector<float>> decoded_;   // the small tensors
    std::vector<Vec> k_cache_, v_cache_;    // per slot: [position][kv head][head_dim]
    std::vector<Vec> conv_;                 // per slot: [channel][n_conv - 1], oldest first
    std::vector<std::vector<int>> experts_;
    Vec logits_;
    float tau_ = 1.0f;
};

}  // namespace gfx
