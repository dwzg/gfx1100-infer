#include "reply.h"

#include <algorithm>
#include <random>
#include <stdexcept>

#include "core/unicode.h"

namespace gfx {

namespace {
const std::string_view k_ws = " \t\n\r\v\f";
const std::string_view k_think_open = "<think>", k_think_close = "</think>", k_call_open = "<tool_call>", k_call_close = "</tool_call>";
const std::string_view k_fn_open = "<function=", k_fn_close = "</function>", k_par_open = "<parameter=", k_par_close = "</parameter>";

// Length of the longest proper prefix of `marker` that `text` ends with.
size_t partial_tail(std::string_view text, std::string_view marker) {
    for (size_t n = std::min(text.size(), marker.size() - 1); n > 0; n--)
        if (text.compare(text.size() - n, n, marker.substr(0, n)) == 0) return n;
    return 0;
}
bool starts_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
// `s` could still become `marker` with more text.
bool is_prefix_of(std::string_view s, std::string_view marker) { return s.size() < marker.size() && marker.compare(0, s.size(), s) == 0; }
std::string_view strip(std::string_view s) {
    size_t b = s.find_first_not_of(k_ws);
    if (b == std::string_view::npos) return {};
    return s.substr(b, s.find_last_not_of(k_ws) - b + 1);
}
}  // namespace

std::string random_id(const char *prefix, int length) {
    static const char alphabet[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::string id = prefix;
    for (int i = 0; i < length; i++) id += alphabet[rng() % 62];
    return id;
}

ReplyParser::ReplyParser(const ChatTemplate &t, const Json &tools, bool in_reasoning, std::vector<std::string> stop_sequences)
    : xml_(t.tool_style == ChatTemplate::ToolStyle::Xml), tools_(tools), stops_(std::move(stop_sequences)), state_(in_reasoning ? State::Reasoning : State::Content) {
    stops_.erase(std::remove_if(stops_.begin(), stops_.end(), [](const std::string &s) { return s.empty(); }), stops_.end());
    content_started_ = in_reasoning || !t.reasoning;   // otherwise the reply may still open with <think>
    inkling_ = t.family == ChatTemplate::Family::Inkling;
    if (inkling_) xml_ = false;
}

void ReplyParser::feed(std::string_view text, std::vector<ReplyEvent> &out) {
    if (stopped_) return;
    buf_ += text;
    step(out, false);
}

// Add text to the open reasoning or text block (opening one if needed). Whitespace at the start
// of a block is dropped; whitespace at its end is held back until something follows it.
void ReplyParser::emit_text(State st, std::string_view text, std::vector<ReplyEvent> &out) {
    if (!open_) {
        size_t b = text.find_first_not_of(k_ws);
        if (b == std::string_view::npos) return;
        text.remove_prefix(b);
        open_reasoning_ = st == State::Reasoning;
        out.push_back({open_reasoning_ ? ReplyEvent::ThinkingStart : ReplyEvent::TextStart, {}, {}});
        open_ = true; held_.clear(); cur_text_.clear();
    }
    std::string all = held_ + std::string(text);
    size_t e = all.find_last_not_of(k_ws);
    e = e == std::string::npos ? 0 : e + 1;
    held_ = all.substr(e);
    if (e == 0) return;
    all.resize(e);
    cur_text_ += all;
    out.push_back({open_reasoning_ ? ReplyEvent::ThinkingDelta : ReplyEvent::TextDelta, std::move(all), {}});
}

void ReplyParser::close_block(std::vector<ReplyEvent> &out) {
    if (!open_) return;
    open_ = false; held_.clear();
    Json b = Json::object();
    if (open_reasoning_) b.set("type", "thinking").set("thinking", cur_text_).set("signature", "local");
    else b.set("type", "text").set("text", cur_text_);
    blocks_.push(std::move(b));
    out.push_back({ReplyEvent::BlockStop, {}, {}});
}

void ReplyParser::start_tool(const std::string &name, std::vector<ReplyEvent> &out) {
    close_block(out);   // the text or reasoning before the call
    tool_name_ = name; tool_id_ = random_id("toolu_"); tool_json_.clear(); tool_input_ = Json::object(); n_args_ = 0; tool_open_ = true;
    out.push_back({ReplyEvent::ToolStart, name, tool_id_});
}
void ReplyParser::tool_delta(const std::string &json, std::vector<ReplyEvent> &out) {
    tool_json_ += json;
    out.push_back({ReplyEvent::ToolDelta, json, {}});
}
void ReplyParser::end_tool(bool closed, std::vector<ReplyEvent> &out) {
    if (!tool_open_) return;
    if (closed && xml_) tool_delta("}", out);
    tool_open_ = false; n_tools_++;
    blocks_.push(Json::object().set("type", "tool_use").set("id", tool_id_).set("name", tool_name_).set("input", tool_input_));
    out.push_back({ReplyEvent::BlockStop, {}, {}});
}

// An argument as the model wrote it between <parameter=...> tags: text for string parameters,
// JSON for everything else. The tool's schema decides.
Json ReplyParser::typed_value(const std::string &name, std::string_view raw) const {
    const Json *schema = nullptr;
    for (const Json &tool : tools_.items()) if (tool["name"].str() == tool_name_) schema = &tool["input_schema"]["properties"][name];
    bool known = false, allow_string = false, only_string = false, allow_bool = false;
    if (schema) {
        const Json &type = (*schema)["type"];
        if (type.is_string()) { known = true; allow_string = only_string = type.str() == "string"; allow_bool = type.str() == "boolean"; }
        else if (type.is_array() && !type.items().empty()) {
            known = true; only_string = true;
            for (const Json &ty : type.items()) { allow_string |= ty.str() == "string"; allow_bool |= ty.str() == "boolean"; only_string &= ty.str() == "string"; }
        }
    }
    if (known && only_string) return Json(raw);
    const std::string_view body = strip(raw);
    if (allow_bool && (body == "True" || body == "False")) return Json(body == "True");
    try {
        Json v = Json::parse(body);
        if (v.is_string() && (allow_string || !known)) return known ? Json(raw) : v;
        return v;
    } catch (const std::exception &) {
        return Json(raw);
    }
}

// One step inside <tool_call>. Returns false when more text is needed.
bool ReplyParser::step_tool(std::vector<ReplyEvent> &out, bool final) {
    const size_t lead = std::min(buf_.find_first_not_of(k_ws), buf_.size());
    const std::string_view rest = std::string_view(buf_).substr(lead);
    if (!xml_ || (sub_ == Sub::Name && !rest.empty() && rest[0] == '{')) {
        // JSON notation: {"name": ..., "arguments": {...}} up to the closing tag.
        size_t end = buf_.find(k_call_close);
        if (end == std::string::npos && !final) return false;
        const std::string body = buf_.substr(0, std::min(end, buf_.size()));
        buf_.erase(0, end == std::string::npos ? buf_.size() : end + k_call_close.size());
        try {
            Json call = Json::parse(strip(body));
            Json args = call["arguments"].is_string() ? Json::parse(call["arguments"].str()) : call["arguments"];
            if (!call["name"].is_string() || !(args.is_object() || args.is_null())) throw std::runtime_error("not a call");
            start_tool(call["name"].str(), out);
            tool_input_ = args.is_object() ? args : Json::object();
            tool_delta(tool_input_.dump(), out);
            const bool was_xml = xml_; xml_ = false; end_tool(true, out); xml_ = was_xml;
            state_ = State::Content;
        } catch (const std::exception &) {
            state_ = resume_;
            emit_text(state_, std::string(k_call_open) + body + (end == std::string::npos ? "" : std::string(k_call_close)), out);   // not a call after all
        }
        return true;
    }
    switch (sub_) {
        case Sub::Name: {
            if (starts_with(rest, k_fn_open)) {
                size_t gt = rest.find('>');
                if (gt == std::string_view::npos) { if (!final) return false; buf_.clear(); state_ = resume_; return true; }
                std::string name(strip(rest.substr(k_fn_open.size(), gt - k_fn_open.size())));
                bool known = false;
                for (const Json &tool : tools_.items()) known |= tool["name"].str() == name;
                if (!wrapped_ && !known) {
                    // "<function=" at the start of a line that names no tool: ordinary text.
                    state_ = resume_;
                    emit_text(state_, std::string(std::string_view(buf_).substr(0, lead + gt + 1)), out);
                    buf_.erase(0, lead + gt + 1);
                    return true;
                }
                buf_.erase(0, lead + gt + 1);
                start_tool(name, out);
                tool_delta("{", out);
                sub_ = Sub::Args;
                return true;
            }
            if (!final && is_prefix_of(rest, k_fn_open)) return false;
            // Neither notation: give the text back as it is.
            size_t end = buf_.find(k_call_close);
            if (end == std::string::npos && !final) return false;
            std::string body = buf_.substr(0, std::min(end, buf_.size()));
            buf_.erase(0, end == std::string::npos ? buf_.size() : end + k_call_close.size());
            state_ = resume_;
            emit_text(state_, (wrapped_ ? std::string(k_call_open) : std::string()) + body + (end == std::string::npos ? "" : std::string(k_call_close)), out);
            return true;
        }
        case Sub::Args: {
            if (starts_with(rest, k_par_open)) {
                size_t gt = rest.find('>');
                if (gt == std::string_view::npos) { if (!final) return false; buf_.clear(); return true; }
                arg_name_ = std::string(strip(rest.substr(k_par_open.size(), gt - k_par_open.size())));
                buf_.erase(0, lead + gt + 1);
                sub_ = Sub::Value;
                return true;
            }
            if (starts_with(rest, k_fn_close)) { buf_.erase(0, lead + k_fn_close.size()); sub_ = Sub::Tail; return true; }
            if (starts_with(rest, k_call_close)) { buf_.erase(0, lead + k_call_close.size()); end_tool(true, out); state_ = State::Content; return true; }
            if (rest.empty() || is_prefix_of(rest, k_par_open) || is_prefix_of(rest, k_fn_close) || is_prefix_of(rest, k_call_close)) {
                if (!final) return false;
                buf_.clear();
                return false;
            }
            // Something else between arguments: skip to the next tag.
            size_t lt = rest.find('<', 1);
            buf_.erase(0, lt == std::string_view::npos ? buf_.size() : lead + lt);
            return !buf_.empty();
        }
        case Sub::Value: {
            size_t end = buf_.find(k_par_close);
            if (end == std::string::npos) return false;
            std::string_view raw = std::string_view(buf_).substr(0, end);
            if (!raw.empty() && raw.front() == '\n') raw.remove_prefix(1);
            if (!raw.empty() && raw.back() == '\n') raw.remove_suffix(1);
            Json v = typed_value(arg_name_, raw);
            std::string piece = n_args_++ ? "," : "";
            json_quote(piece, arg_name_); piece += ":"; v.dump_to(piece, false);
            tool_input_.set(arg_name_, std::move(v));
            tool_delta(piece, out);
            buf_.erase(0, end + k_par_close.size());
            sub_ = Sub::Args;
            return true;
        }
        case Sub::Tail: {
            if (starts_with(rest, k_call_close)) buf_.erase(0, lead + k_call_close.size());
            else if (!final && (rest.empty() || is_prefix_of(rest, k_call_close))) return false;
            end_tool(true, out);
            state_ = State::Content;
            return true;
        }
    }
    return false;
}

// Inkling: the reply is a series of messages, <|message_model|>[tool name]<|content_KIND|> ... <|end_message|>,
// with KIND one of thinking, text, or invoke_tool_json ({"name": ..., "args": {...}}).
void ReplyParser::step_inkling(std::vector<ReplyEvent> &out, bool final) {
    static const std::string_view k_open = "<|message_model|>", k_end = "<|end_message|>";
    static const std::string_view k_kinds[3] = {"<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>"};
    for (;;) {
        if (stopped_) return;
        if (msg_ == Msg::Outside) {
            size_t at = buf_.find(k_open);
            // Anything outside a message is ordinary text (the model should not write any).
            size_t n = at != std::string::npos ? at : buf_.size() - (final ? 0 : partial_tail(buf_, k_open));
            if (at == std::string::npos && !final) n = utf8_complete_prefix(std::string_view(buf_).substr(0, n));
            if (n > 0) { emit_text(State::Content, std::string_view(buf_).substr(0, n), out); buf_.erase(0, n); }
            if (at == std::string::npos) return;
            close_block(out);
            buf_.erase(0, k_open.size());
            msg_ = Msg::Header;
        } else if (msg_ == Msg::Header) {
            size_t at = std::string::npos; int kind = -1;
            for (int i = 0; i < 3; i++) { size_t p = buf_.find(k_kinds[i]); if (p < at) { at = p; kind = i; } }
            if (kind < 0) { if (final) buf_.clear(); return; }
            msg_name_ = std::string(strip(std::string_view(buf_).substr(0, at)));
            buf_.erase(0, at + k_kinds[kind].size());
            msg_ = kind == 0 ? Msg::Thinking : kind == 1 ? Msg::Text : Msg::Tool;
        } else if (msg_ == Msg::Tool) {
            size_t end = buf_.find(k_end);
            if (end == std::string::npos && !final) return;
            const std::string body = buf_.substr(0, std::min(end, buf_.size()));
            buf_.erase(0, end == std::string::npos ? buf_.size() : end + k_end.size());
            msg_ = Msg::Outside;
            try {
                Json call = Json::parse(strip(body));
                const Json &args = call["args"];
                if (!(args.is_object() || args.is_null())) throw std::runtime_error("not a call");
                start_tool(call["name"].is_string() ? call["name"].str() : msg_name_, out);
                tool_input_ = args.is_object() ? args : Json::object();
                tool_delta(tool_input_.dump(), out);
                end_tool(true, out);
            } catch (const std::exception &) {
                if (strip(body).empty()) continue;
                emit_text(State::Content, body, out);   // not a call after all
                close_block(out);
            }
        } else {
            const State st = msg_ == Msg::Thinking ? State::Reasoning : State::Content;
            size_t at = buf_.find(k_end), stop_index = 0; bool is_stop = false;
            if (msg_ == Msg::Text)
                for (size_t i = 0; i < stops_.size(); i++) { size_t p = buf_.find(stops_[i]); if (p < at) { at = p; is_stop = true; stop_index = i; } }
            if (at == std::string::npos) {
                size_t keep = final ? 0 : partial_tail(buf_, k_end);
                if (!final && msg_ == Msg::Text) for (const std::string &s : stops_) keep = std::max(keep, partial_tail(buf_, s));
                size_t n = buf_.size() - keep;
                if (!final) n = utf8_complete_prefix(std::string_view(buf_).substr(0, n));
                if (n > 0) { emit_text(st, std::string_view(buf_).substr(0, n), out); buf_.erase(0, n); }
                return;
            }
            emit_text(st, std::string_view(buf_).substr(0, at), out);
            close_block(out);
            if (is_stop) { stopped_ = true; stop_hit_ = stops_[stop_index]; buf_.clear(); return; }
            buf_.erase(0, at + k_end.size());
            msg_ = Msg::Outside;
        }
    }
}

void ReplyParser::step(std::vector<ReplyEvent> &out, bool final) {
    if (inkling_) { step_inkling(out, final); return; }
    for (;;) {
        if (stopped_) return;
        if (state_ == State::Tool) { if (!step_tool(out, final)) return; continue; }
        if (buf_.empty()) return;

        if (state_ == State::Content && !content_started_) {
            // A reply that was not started inside <think> may still open with it.
            const size_t lead = std::min(buf_.find_first_not_of(k_ws), buf_.size());
            const std::string_view rest = std::string_view(buf_).substr(lead);
            if (starts_with(rest, k_think_open)) { buf_.erase(0, lead + k_think_open.size()); state_ = State::Reasoning; content_started_ = true; continue; }
            if (!final && (rest.empty() || is_prefix_of(rest, k_think_open))) return;
            content_started_ = true;
        }

        // The first thing in the buffer that ends the current block.
        size_t at = std::string::npos, skip = 0; enum { None, ThinkEnd, Call, Function, Stop } what = None;
        auto look = [&](std::string_view marker, int kind, size_t consume) {
            size_t p = buf_.find(marker);
            if (p < at) { at = p; what = (decltype(what))kind; skip = consume; }
        };
        if (state_ == State::Reasoning) look(k_think_close, ThinkEnd, k_think_close.size());
        look(k_call_open, Call, k_call_open.size());
        if (xml_ && state_ == State::Content && !tools_.items().empty()) {
            // A call written without its wrapper: "<function=" at the start of a line.
            for (size_t p = buf_.find(k_fn_open); p < at; p = buf_.find(k_fn_open, p + 1)) {
                const bool line_start = p > 0 ? buf_[p - 1] == '\n' : (!open_ || (!held_.empty() && held_.back() == '\n'));
                if (line_start) { at = p; what = Function; skip = 0; break; }
            }
        }
        size_t stop_index = 0;
        if (state_ == State::Content)
            for (size_t i = 0; i < stops_.size(); i++) { size_t before = at; look(stops_[i], Stop, stops_[i].size()); if (at != before) stop_index = i; }

        if (what == None) {
            // Give out what cannot be the start of a marker (or half a character).
            size_t keep = 0;
            if (!final) {
                if (state_ == State::Reasoning) keep = partial_tail(buf_, k_think_close);
                keep = std::max(keep, partial_tail(buf_, k_call_open));
                if (xml_ && state_ == State::Content && !tools_.items().empty()) keep = std::max(keep, partial_tail(buf_, k_fn_open));
                if (state_ == State::Content) for (const std::string &s : stops_) keep = std::max(keep, partial_tail(buf_, s));
            }
            size_t n = buf_.size() - keep;
            if (!final) n = utf8_complete_prefix(std::string_view(buf_).substr(0, n));
            if (n > 0) { emit_text(state_, std::string_view(buf_).substr(0, n), out); buf_.erase(0, n); }
            return;
        }
        emit_text(state_, std::string_view(buf_).substr(0, at), out);
        buf_.erase(0, at + skip);
        if (what == ThinkEnd) { close_block(out); state_ = State::Content; }
        else if (what == Stop) { close_block(out); stopped_ = true; stop_hit_ = stops_[stop_index]; buf_.clear(); return; }
        else { resume_ = state_; state_ = State::Tool; sub_ = Sub::Name; wrapped_ = what == Call; }   // the open block is closed once a call really starts
    }
}

void ReplyParser::finish(bool complete, std::vector<ReplyEvent> &out) {
    step(out, true);
    if (state_ == State::Tool && tool_open_) {
        if (complete && sub_ == Sub::Value) {
            // The turn ended inside an argument: take what is there.
            std::string_view raw = buf_;
            if (!raw.empty() && raw.front() == '\n') raw.remove_prefix(1);
            Json v = typed_value(arg_name_, raw);
            std::string piece = n_args_++ ? "," : "";
            json_quote(piece, arg_name_); piece += ":"; v.dump_to(piece, false);
            tool_input_.set(arg_name_, std::move(v));
            tool_delta(piece, out);
        }
        end_tool(complete, out);   // cut off: the input JSON stays unterminated, as upstream does
        state_ = State::Content;
    }
    buf_.clear();
    close_block(out);
}

}  // namespace gfx
