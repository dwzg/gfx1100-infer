// Chat prompt formatting. Models ship a Jinja template; rather than interpreting it, the known
// formats are recognised from it and produced directly.
#pragma once
#include <string>

namespace gfx {

class GGUF;

struct ChatFormat {
    bool supported = false;   // only ChatML-style templates (<|im_start|>role ... <|im_end|>) so far
    bool think_tag = false;   // the template opens the reply with <think> (reasoning models)
    std::string name;

    // One complete message.
    std::string message(const std::string &role, const std::string &content) const;
    // What follows the last message to make the model answer. With thinking off, reasoning models
    // get an empty reasoning block so they answer directly.
    std::string reply_start(bool thinking) const;
    // What closes a reply before the next message.
    std::string reply_end() const;
};

ChatFormat detect_chat_format(const GGUF &g);

}  // namespace gfx
