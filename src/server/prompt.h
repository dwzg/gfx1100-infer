// Turns a Messages API request into the prompt the model was trained on, and a reply back into
// the canonical text of an assistant message.
//
// Models ship their chat format as a Jinja template. Rather than interpreting it, the formats in
// use are recognised from the template (ChatML with tool calls written either as JSON or as
// <function=...><parameter=...> blocks) and the fixed text around the tool list is read out of it.
#pragma once
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/json.h"

namespace gfx {

class GGUF;

struct ChatTemplate {
    bool supported = false;
    std::string problem;                    // why not, if not supported
    enum class ToolStyle { Json, Xml } tool_style = ToolStyle::Json;
    bool tools_first = false;               // the tool list comes before the system text
    bool trim = false;                      // message text is trimmed
    bool reasoning = false;                 // the model writes <think>...</think> before answering
    bool think_open = false;                // the prompt itself opens <think> when reasoning is on
    std::string tools_head, tools_tail;     // text before / after the tool list
    std::string effort_high, effort_low;    // reasoning-effort instructions, if the template has any
    std::string default_system;             // the system text the template supplies when a request has none
    std::string describe() const;

    static ChatTemplate from_text(const std::string &jinja);
    static ChatTemplate detect(const GGUF &g);
};

// Text assembled from markup and quoted material. The quoted ranges are tokenised as plain text,
// so a file that contains "<|im_end|>" cannot end a message.
struct MarkedText {
    std::string s;
    std::vector<std::pair<size_t, size_t>> literal;
    MarkedText &markup(std::string_view t) { s += t; return *this; }
    MarkedText &quote(std::string_view t) {
        if (t.empty()) return *this;
        if (!literal.empty() && literal.back().second == s.size()) literal.back().second += t.size(); else literal.emplace_back(s.size(), s.size() + t.size());
        s += t;
        return *this;
    }
    MarkedText &append(const MarkedText &o) {
        for (auto [b, e] : o.literal) literal.emplace_back(b + s.size(), e + s.size());
        s += o.s;
        return *this;
    }
};

struct RenderOptions {
    bool thinking = false;          // let the model reason before answering
    std::string effort;             // "low", "medium" or "high" (empty: the template's default)
    // Keep the reasoning of every earlier assistant message in the prompt. This keeps the prompt
    // of a growing conversation append-only, which is what lets processed context be reused.
    bool keep_thinking = true;
};

struct Prompt {
    // The prompt in order. A `body` piece is the text of an earlier assistant message between
    // "<|im_start|>assistant\n" and "<|im_end|>": the server substitutes the exact tokens the
    // model generated for it when it still has them.
    struct Piece { MarkedText text; bool body = false; };
    std::vector<Piece> pieces;
    bool has_system = false;        // pieces[0] is the system message
    size_t tools_bytes = 0;         // bytes of pieces[0] up to the end of the tool list (0: not applicable)
    MarkedText reply_prefix;        // how the reply's body starts (already part of the prompt)
    std::string fed_to_parser;      // the part of reply_prefix the reply parser must see (a forced tool call)
    bool starts_reasoning = false;  // the reply starts inside <think>
    std::string text() const;
};

// Throws std::runtime_error on malformed requests.
Prompt render_prompt(const ChatTemplate &t, const Json &request, const RenderOptions &o);
// Canonical body of an assistant message with these content blocks (thinking, text, tool_use).
MarkedText render_assistant_body(const ChatTemplate &t, const Json &content, bool with_reasoning);

}  // namespace gfx
