// Token sampling: temperature, top-k and nucleus (top-p) filtering.
#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace gfx {

struct SamplerParams {
    float temperature = 0.7f;  // <= 0 selects the most likely token
    int top_k = 20;            // <= 0 disables
    float top_p = 0.8f;        // >= 1 disables
    uint64_t seed = 0;
};

class Sampler {
  public:
    explicit Sampler(const SamplerParams &p) : p_(p), rng_(p.seed) {}
    int32_t sample(const float *logits, int n_vocab);
  private:
    SamplerParams p_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int32_t>> cand_;
};

}  // namespace gfx
