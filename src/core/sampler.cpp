#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace gfx {

int32_t Sampler::sample(const float *logits, int n) {
    if (p_.temperature <= 0.0f) return (int32_t)(std::max_element(logits, logits + n) - logits);
    cand_.resize(n);
    for (int i = 0; i < n; i++) cand_[i] = {logits[i], i};
    size_t keep = (p_.top_k > 0 && p_.top_k < n) ? (size_t)p_.top_k : (size_t)n;
    auto greater = [](const auto &a, const auto &b) { return a.first > b.first; };
    if (keep < cand_.size()) { std::partial_sort(cand_.begin(), cand_.begin() + keep, cand_.end(), greater); cand_.resize(keep); }
    else std::sort(cand_.begin(), cand_.end(), greater);
    float mx = cand_[0].first; double sum = 0;
    for (auto &c : cand_) { c.first = std::exp((c.first - mx) / p_.temperature); sum += c.first; }
    if (p_.top_p < 1.0f) {
        double acc = 0; size_t cut = cand_.size();
        for (size_t i = 0; i < cand_.size(); i++) { acc += cand_[i].first / sum; if (acc >= p_.top_p) { cut = i + 1; break; } }
        cand_.resize(cut);
        sum = 0; for (auto &c : cand_) sum += c.first;
    }
    double r = std::uniform_real_distribution<double>(0.0, sum)(rng_), acc = 0;
    for (auto &c : cand_) { acc += c.first; if (r < acc) return c.second; }
    return cand_.back().second;
}

}  // namespace gfx
