// Reference implementation of the forward pass: plain scalar fp32 on the CPU, one token at a
// time. It favours being obviously correct over speed; the GPU engine is validated against it.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "model.h"

namespace gfx {

class ThreadPool;

class RefModel {
  public:
    RefModel(const GGUF &g, const ModelDesc &desc, int n_ctx, int n_threads);
    ~RefModel();

    void reset();
    int n_past() const { return n_past_; }
    int n_ctx() const { return n_ctx_; }
    // Test aid: continue from position n as if n all-zero cache entries preceded it.
    void skip_to(int n) { reset(); for (auto &c : k_cache_) std::fill(c.begin(), c.end(), 0.0f); for (auto &c : v_cache_) std::fill(c.begin(), c.end(), 0.0f); n_past_ = n; }
    // Process tokens at the next positions. Returns the logits for the last one.
    const std::vector<float> &eval(const int32_t *tokens, int n);

    // Optional probe: called with (layer, name, data, count) for intermediate results of the last
    // evaluated token. Used to compare against the GPU engine stage by stage.
    std::function<void(int, const char *, const float *, size_t)> probe;

  private:
    const float *weights(const TensorInfo *t);
    void matvec(const TensorInfo *w, const float *x, float *y, const TensorInfo *bias = nullptr);
    void rms_norm(const float *x, const TensorInfo *w, float *y, int n, int groups = 1);
    void rope(float *x, int n_heads, int pos) const;
    void attention(int il, int slot, const float *h, float *out);
    void linear_attention(int il, int slot, const float *h, float *out);
    void eval_one(int32_t token);

    const GGUF &g_;
    ModelDesc d_;
    int n_ctx_, n_past_ = 0;
    std::unique_ptr<ThreadPool> pool_;
    std::unordered_map<const TensorInfo *, std::vector<float>> decoded_;
    std::vector<int> slot_;                        // layer -> index among attention / recurrent layers
    std::vector<std::vector<float>> k_cache_, v_cache_;   // per attention layer: [n_ctx][n_head_kv * head_dim]
    std::vector<std::vector<float>> dn_state_, dn_conv_;  // per recurrent layer
    std::vector<float> x_, logits_;
};

}  // namespace gfx
