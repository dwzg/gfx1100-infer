#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "gguf.h"
#include "unicode.h"

namespace gfx {

// GPT-2 style byte-level alphabet: every byte is written as one printable code point.
static uint32_t byte_to_cp(uint8_t b) {
    static uint32_t table[256]; static bool init = false;
    if (!init) {
        uint32_t next = 256;
        for (int i = 0; i < 256; i++) {
            bool printable = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174);
            table[i] = printable ? (uint32_t)i : next++;
        }
        init = true;
    }
    return table[b];
}
static int cp_to_byte(uint32_t cp) {
    static std::unordered_map<uint32_t, int> rev;
    if (rev.empty()) for (int i = 0; i < 256; i++) rev[byte_to_cp((uint8_t)i)] = i;
    auto it = rev.find(cp);
    return it == rev.end() ? -1 : it->second;
}

struct PreRule { const char *name; bool ignore_merges; std::vector<const char *> regexes; };
static const std::vector<PreRule> &pre_rules() {
    static const char *contractions = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
    static std::string qwen35 = std::string(contractions) + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static std::string qwen2 = std::string(contractions) + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static std::string llama3 = std::string(contractions) + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static std::string gpt4o = std::string("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+") + contractions + "?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*" + contractions + "?|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const char *gpt2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    static const std::vector<PreRule> rules = {
        {"qwen35", false, {qwen35.c_str()}},
        {"qwen2", false, {qwen2.c_str()}},
        {"deepseek-r1-qwen", false, {qwen2.c_str()}},
        {"llama3", true, {llama3.c_str()}},
        {"llama-bpe", true, {llama3.c_str()}},
        {"llama-v3", true, {llama3.c_str()}},
        {"inkling2", true, {gpt4o.c_str()}},
        {"gpt-4o", false, {gpt4o.c_str()}},
        {"smollm", false, {"\\p{N}", gpt2}},
        {"gpt-2", false, {gpt2}},
    };
    return rules;
}

Tokenizer::Tokenizer(const GGUF &g) {
    std::string model = g.get_str("tokenizer.ggml.model");
    if (model != "gpt2") throw std::runtime_error("tokenizer type '" + model + "' is not supported (only byte-pair 'gpt2' vocabularies)");
    pre_ = g.get_str("tokenizer.ggml.pre", "gpt-2");
    const PreRule *rule = nullptr;
    for (const auto &r : pre_rules()) if (pre_ == r.name) rule = &r;
    if (!rule) throw std::runtime_error("pre-tokenizer '" + pre_ + "' is not supported");
    ignore_merges_ = rule->ignore_merges;
    for (const char *re : rule->regexes) splitters_.emplace_back(re);

    const MetaValue *toks = g.find("tokenizer.ggml.tokens");
    if (!toks || toks->type != M_ARR || toks->elem_type != M_STR) throw std::runtime_error("model has no token list");
    tokens_.reserve(toks->count); index_.reserve(toks->count * 2);
    for (size_t i = 0; i < toks->strs.size(); i++) { tokens_.emplace_back(toks->strs[i]); index_.emplace(tokens_.back(), (int32_t)i); }
    types_.assign(tokens_.size(), 1);
    if (const MetaValue *ty = g.find("tokenizer.ggml.token_type"); ty && ty->type == M_ARR && ty->elem_type != M_STR)
        for (uint64_t i = 0; i < ty->count && i < types_.size(); i++) types_[i] = (int32_t)ty->arr_int(i);
    if (const MetaValue *mg = g.find("tokenizer.ggml.merges"); mg && mg->type == M_ARR && mg->elem_type == M_STR) {
        ranks_.reserve(mg->strs.size() * 2);
        for (size_t i = 0; i < mg->strs.size(); i++) ranks_.emplace(std::string(mg->strs[i]), (int32_t)i);
    }
    bos_ = (int32_t)g.get_int("tokenizer.ggml.bos_token_id", -1);
    eos_ = (int32_t)g.get_int("tokenizer.ggml.eos_token_id", -1);
    eot_ = (int32_t)g.get_int("tokenizer.ggml.eot_token_id", -1);
    add_bos_ = g.get_int("tokenizer.ggml.add_bos_token", 0) != 0 && bos_ >= 0;

    special_by_first_.resize(256);
    for (int32_t i = 0; i < (int32_t)tokens_.size(); i++)
        if ((types_[i] == 3 || types_[i] == 4) && !tokens_[i].empty()) special_by_first_[(uint8_t)tokens_[i][0]].push_back(i);
    for (auto &b : special_by_first_)
        std::sort(b.begin(), b.end(), [&](int32_t a, int32_t c) { return tokens_[a].size() > tokens_[c].size(); });

    if (eos_ >= 0) eog_.push_back(eos_);
    if (eot_ >= 0) eog_.push_back(eot_);
    for (const char *name : {"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<end_of_turn>", "<|eom_id|>"}) {
        int32_t id = find(name);
        if (id >= 0 && is_control(id)) eog_.push_back(id);
    }
}

int32_t Tokenizer::find(std::string_view text) const {
    auto it = index_.find(std::string(text));
    return it == index_.end() ? -1 : it->second;
}
bool Tokenizer::is_control(int32_t t) const { return t >= 0 && t < n_vocab() && types_[t] == 3; }
bool Tokenizer::is_eog(int32_t t) const { return std::find(eog_.begin(), eog_.end(), t) != eog_.end(); }

void Tokenizer::bpe_word(std::string_view word, std::vector<int32_t> &out) const {
    auto hit = cache_.find(std::string(word));
    if (hit != cache_.end()) { out.insert(out.end(), hit->second.begin(), hit->second.end()); return; }
    std::vector<int32_t> ids;

    // Rewrite the word in the byte-level alphabet; each byte starts as its own symbol.
    std::vector<std::string> sym; sym.reserve(word.size());
    std::string whole;
    for (unsigned char b : word) { std::string s; utf8_append(s, byte_to_cp(b)); whole += s; sym.push_back(std::move(s)); }

    auto whole_it = ignore_merges_ ? index_.find(whole) : index_.end();
    if (whole_it != index_.end()) {
        ids.push_back(whole_it->second);
    } else {
        std::string key;
        while (sym.size() > 1) {
            int best = INT_MAX; size_t at = 0;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                key.assign(sym[i]); key += ' '; key += sym[i + 1];
                auto r = ranks_.find(key);
                if (r != ranks_.end() && r->second < best) { best = r->second; at = i; }
            }
            if (best == INT_MAX) break;
            // Merge every occurrence of the winning pair, left to right.
            std::string left = sym[at], right = sym[at + 1];
            std::vector<std::string> next; next.reserve(sym.size());
            for (size_t i = 0; i < sym.size();) {
                if (i + 1 < sym.size() && sym[i] == left && sym[i + 1] == right) { next.push_back(left + right); i += 2; }
                else next.push_back(std::move(sym[i++]));
            }
            sym.swap(next);
        }
        for (const std::string &s : sym) {
            auto it = index_.find(s);
            if (it != index_.end()) { ids.push_back(it->second); continue; }
            // Not in the vocabulary as a unit: fall back to its individual byte symbols.
            for (uint32_t cp : utf8_decode(s)) {
                std::string one; utf8_append(one, cp);
                auto b = index_.find(one);
                if (b != index_.end()) ids.push_back(b->second);
            }
        }
    }
    if (cache_.size() < (1u << 20)) cache_.emplace(std::string(word), ids);
    out.insert(out.end(), ids.begin(), ids.end());
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t> &out) const {
    if (text.empty()) return;
    std::vector<uint32_t> off;
    std::vector<uint32_t> cps = utf8_decode(text, off);
    std::vector<std::pair<uint32_t, uint32_t>> pieces{{0, (uint32_t)cps.size()}};
    for (const Regex &re : splitters_) {
        std::vector<std::pair<uint32_t, uint32_t>> next;
        for (auto [b, e] : pieces) { auto sub = regex_split(re, cps, b, e); next.insert(next.end(), sub.begin(), sub.end()); }
        pieces.swap(next);
    }
    for (auto [b, e] : pieces) bpe_word(text.substr(off[b], off[e] - off[b]), out);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_special, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_special && add_bos_) out.push_back(bos_);
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        int32_t found = -1;
        for (int32_t id : special_by_first_[(uint8_t)text[i]]) {
            // User-defined tokens always match; control tokens only when asked to parse them.
            if (types_[id] == 3 && !parse_special) continue;
            const std::string &t = tokens_[id];
            if (text.compare(i, t.size(), t) == 0) { found = id; break; }
        }
        if (found < 0) { i++; continue; }
        encode_plain(text.substr(start, i - start), out);
        out.push_back(found);
        i += tokens_[found].size(); start = i;
    }
    encode_plain(text.substr(start), out);
    return out;
}

std::string Tokenizer::piece(int32_t token) const {
    if (token < 0 || token >= n_vocab()) return {};
    const std::string &t = tokens_[token];
    if (types_[token] == 3 || types_[token] == 4) return t;  // special tokens are stored as plain text
    std::string out;
    for (uint32_t cp : utf8_decode(t)) {
        int b = cp_to_byte(cp);
        if (b >= 0) out += (char)b; else utf8_append(out, cp);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> &tokens, bool skip_special) const {
    std::string out;
    for (int32_t t : tokens) {
        if (skip_special && is_control(t)) continue;
        out += piece(t);
    }
    return out;
}

}  // namespace gfx
