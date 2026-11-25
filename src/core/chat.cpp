#include "chat.h"

#include "gguf.h"

namespace gfx {

ChatFormat detect_chat_format(const GGUF &g) {
    ChatFormat f;
    std::string t = g.get_str("tokenizer.chat_template");
    if (t.find("<|im_start|>") != std::string::npos) {
        f.supported = true; f.name = "chatml";
        f.think_tag = t.find("<think>\\n") != std::string::npos && t.find("enable_thinking") != std::string::npos;
    } else {
        f.name = t.empty() ? "none" : "unrecognised";
    }
    return f;
}

std::string ChatFormat::message(const std::string &role, const std::string &content) const {
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}
std::string ChatFormat::reply_start(bool thinking) const {
    std::string s = "<|im_start|>assistant\n";
    if (think_tag) s += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}
std::string ChatFormat::reply_end() const { return "<|im_end|>\n"; }

}  // namespace gfx
