// Byte-pair-encoding tokenizer driven by the vocabulary stored in a GGUF file.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "regex.h"

namespace gfx {

class GGUF;

class Tokenizer {
  public:
    // Throws std::runtime_error if the vocabulary type or pre-tokenizer is not supported.
    explicit Tokenizer(const GGUF &g);

    // parse_special: recognise control tokens such as <|im_start|> written out in the text.
    // add_special: prepend BOS when the model's metadata asks for it.
    // literal: sorted, non-overlapping byte ranges [begin, end) of the text that are quoted
    // material (a user's message, a file's contents): no control or other special token is
    // recognised where it would overlap one, so quoted text can never act as markup.
    using Ranges = std::vector<std::pair<size_t, size_t>>;
    std::vector<int32_t> encode(std::string_view text, bool add_special, bool parse_special, const Ranges *literal = nullptr) const;

    // Raw bytes of one token (may be an incomplete UTF-8 sequence).
    std::string piece(int32_t token) const;
    std::string decode(const std::vector<int32_t> &tokens, bool skip_special = true) const;

    int32_t n_vocab() const { return (int32_t)tokens_.size(); }
    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    bool add_bos() const { return add_bos_; }
    bool is_control(int32_t token) const;
    // End-of-generation tokens: EOS, EOT and the usual end-of-turn markers.
    bool is_eog(int32_t token) const;
    int32_t find(std::string_view text) const;  // exact vocabulary lookup, -1 if absent
    const std::string &pre_type() const { return pre_; }

  private:
    void bpe_word(std::string_view word, std::vector<int32_t> &out) const;
    void encode_plain(std::string_view text, std::vector<int32_t> &out) const;

    std::vector<std::string> tokens_;   // vocabulary text as stored (byte-level alphabet)
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> index_;
    std::unordered_map<std::string, int32_t> ranks_;  // "left right" -> merge priority
    std::vector<std::vector<int32_t>> special_by_first_;  // special tokens bucketed by first byte, longest first
    std::vector<Regex> splitters_;
    std::vector<int32_t> eog_;
    std::string pre_;
    int32_t bos_ = -1, eos_ = -1, eot_ = -1;
    bool add_bos_ = false, ignore_merges_ = false;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

}  // namespace gfx
