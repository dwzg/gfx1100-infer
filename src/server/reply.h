// Splits the text a model generates into the content blocks of a Messages API reply: reasoning
// (<think>...</think>), text, and tool calls (<tool_call>...</tool_call> in either notation).
// Works incrementally, so each block can be streamed while it is being generated.
#pragma once
#include <string>
#include <string_view>
#include <vector>

#include "core/json.h"
#include "prompt.h"

namespace gfx {

struct ReplyEvent {
    enum Kind { ThinkingStart, ThinkingDelta, TextStart, TextDelta, ToolStart, ToolDelta, BlockStop } kind;
    std::string text;   // delta text; for ToolStart the tool name; for ToolDelta a piece of the input JSON
    std::string id;     // ToolStart: the call's id
};

class ReplyParser {
  public:
    // tools: the request's tool definitions (their schemas decide whether an argument written by
    // the model is a string or JSON). in_reasoning: the reply starts inside <think>.
    ReplyParser(const ChatTemplate &t, const Json &tools, bool in_reasoning, std::vector<std::string> stop_sequences = {});
    void feed(std::string_view text, std::vector<ReplyEvent> &out);
    // End of generation. complete: the model ended its turn itself (as opposed to being cut off).
    void finish(bool complete, std::vector<ReplyEvent> &out);

    bool stopped() const { return stopped_; }                       // a stop sequence was reached
    const std::string &stop_sequence() const { return stop_hit_; }
    bool has_tool_call() const { return n_tools_ > 0; }
    // The reply as content blocks (thinking, text, tool_use), as far as it has been parsed.
    const Json &blocks() const { return blocks_; }

  private:
    enum class State { Reasoning, Content, Tool };
    enum class Sub { Name, Args, Value, Tail };
    void step(std::vector<ReplyEvent> &out, bool final);
    bool step_tool(std::vector<ReplyEvent> &out, bool final);
    void emit_text(State st, std::string_view text, std::vector<ReplyEvent> &out);
    void close_block(std::vector<ReplyEvent> &out);
    void start_tool(const std::string &name, std::vector<ReplyEvent> &out);
    void tool_delta(const std::string &json, std::vector<ReplyEvent> &out);
    void end_tool(bool closed, std::vector<ReplyEvent> &out);
    Json typed_value(const std::string &name, std::string_view raw) const;

    bool xml_;
    const Json tools_;
    std::vector<std::string> stops_;
    State state_, resume_ = State::Content;   // resume_: where to continue if a tool-call marker turns out not to start a call
    std::string buf_;            // generated text not yet assigned to a block
    std::string held_;           // trailing whitespace of the open text / reasoning block, kept back until more follows
    bool open_ = false;          // a text or reasoning block is open
    bool open_reasoning_ = false;
    bool wrapped_ = true;        // the tool call being read started with <tool_call>
    bool content_started_ = false;
    Json blocks_ = Json::array();
    std::string cur_text_;
    // tool call in progress
    Sub sub_ = Sub::Name;
    bool tool_open_ = false;
    std::string tool_name_, tool_id_, arg_name_, tool_json_;
    Json tool_input_;
    int n_tools_ = 0, n_args_ = 0;
    bool stopped_ = false;
    std::string stop_hit_;
};

std::string random_id(const char *prefix, int length = 24);

}  // namespace gfx
