// A small backtracking regular-expression matcher over Unicode code points. It implements the
// subset used by tokenizer word-splitting rules: literals, ., classes with ranges and \p{..},
// \s \S \d, groups, alternation, greedy quantifiers, and lookahead.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

class Regex {
  public:
    explicit Regex(std::string_view pattern);  // throws std::runtime_error on unsupported syntax
    ~Regex();
    Regex(Regex &&) noexcept;
    // Length of the leftmost-alternative, greedy match starting exactly at `pos`, or -1.
    long match_at(const std::vector<uint32_t> &text, size_t pos) const;

    struct Impl;
  private:
    std::unique_ptr<Impl> impl_;
};

// Split text into pieces the way a tokenizer "Split" pre-tokenizer with isolated behaviour does:
// every regex match becomes a piece and so does every gap between matches. Returns piece
// boundaries as [begin, end) code point indices.
std::vector<std::pair<uint32_t, uint32_t>> regex_split(const Regex &re, const std::vector<uint32_t> &text, uint32_t begin, uint32_t end);

}  // namespace gfx
