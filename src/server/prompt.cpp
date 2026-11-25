#include "prompt.h"

#include <stdexcept>

#include "core/gguf.h"

namespace gfx {

namespace {

// String literals of a Jinja template, in order, with where they are.
struct Literal { size_t begin, end; std::string text; };

std::vector<Literal> jinja_literals(const std::string &s) {
    std::vector<Literal> out;
    for (size_t i = 0; i + 1 < s.size();) {
        if (s[i] != '{' || (s[i + 1] != '{' && s[i + 1] != '%' && s[i + 1] != '#')) { i++; continue; }
        const char close = s[i + 1] == '{' ? '}' : s[i + 1];
        i += 2;
        while (i + 1 < s.size() && !(s[i] == close && s[i + 1] == '}')) {
            if (close != '#' && (s[i] == '\'' || s[i] == '"')) {
                const char q = s[i];
                Literal lit{i, 0, {}};
                for (i++; i < s.size() && s[i] != q; i++) {
                    if (s[i] != '\\' || i + 1 >= s.size()) { lit.text += s[i]; continue; }
                    char c = s[++i];
                    lit.text += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
                }
                lit.end = ++i;
                out.push_back(std::move(lit));
            } else i++;
        }
        i += 2;
    }
    return out;
}

std::string_view strip(std::string_view s, std::string_view chars = " \t\n\r\v\f") {
    size_t b = s.find_first_not_of(chars);
    if (b == std::string_view::npos) return {};
    return s.substr(b, s.find_last_not_of(chars) - b + 1);
}

const char *k_hermes_head = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const char *k_hermes_tail = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

// Text of a content value: a string, or the text blocks of an array (other block types are named).
std::string plain_text(const Json &content, const char *sep) {
    if (content.is_string()) return content.str();
    std::string out;
    auto add = [&](std::string_view t) { if (!out.empty()) out += sep; out += t; };
    for (const Json &b : content.items()) {
        const std::string &type = b["type"].str();
        if (type == "text") add(b["text"].str());
        else if (type == "image") add("[image]");
        else if (type == "document") add("[document]");
    }
    return out;
}

// A tool as the chat templates expect it (the OpenAI function format), written as JSON.
std::string tool_definition(const Json &tool) {
    Json fn = Json::object();
    fn.set("name", tool["name"]);
    if (tool.has("description")) fn.set("description", tool["description"]);
    fn.set("parameters", tool["input_schema"]);
    return Json::object().set("type", "function").set("function", std::move(fn)).dump(true);
}

// Text of a tool result. A result can name tools whose definitions were held back until now
// (deferred loading); their definitions are written out, since this is where the model learns them.
std::string result_text(const Json &content, const Json &tools) {
    if (content.is_string()) return content.str();
    std::string out;
    auto add = [&](std::string_view t) { if (!out.empty()) out += "\n"; out += t; };
    for (const Json &b : content.items()) {
        const std::string &type = b["type"].str();
        if (type == "text") add(b["text"].str());
        else if (type == "image") add("[image]");
        else if (type == "document") add("[document]");
        else if (type == "tool_reference") {
            bool found = false;
            for (const Json &tool : tools.items())
                if (tool["name"].str() == b["tool_name"].str() && tool["input_schema"].is_object()) { add("This function can be called from now on:\n" + tool_definition(tool)); found = true; break; }
            if (!found) add("(no function named " + b["tool_name"].str() + " is available)");
        }
    }
    return out;
}

struct Parts { std::string reasoning, text; std::vector<const Json *> calls; };
Parts split_assistant(const Json &content) {
    Parts p;
    if (content.is_string()) { p.text = content.str(); return p; }
    for (const Json &b : content.items()) {
        const std::string &type = b["type"].str();
        if (type == "thinking") { if (!p.reasoning.empty()) p.reasoning += "\n"; p.reasoning += b["thinking"].str(); }
        else if (type == "text") { if (!p.text.empty() && !b["text"].str().empty()) p.text += "\n\n"; p.text += b["text"].str(); }
        else if (type == "tool_use") p.calls.push_back(&b);
    }
    return p;
}

}  // namespace

ChatTemplate ChatTemplate::from_text(const std::string &jinja) {
    ChatTemplate t;
    if (jinja.find("<|im_start|>") == std::string::npos) { t.problem = "only ChatML-style chat templates (<|im_start|>role ... <|im_end|>) are supported"; return t; }
    t.supported = true;
    t.trim = jinja.find("|trim") != std::string::npos || jinja.find("| trim") != std::string::npos;
    t.reasoning = jinja.find("<think>") != std::string::npos;
    t.tool_style = jinja.find("<function=") != std::string::npos ? ToolStyle::Xml : ToolStyle::Json;
    const std::vector<Literal> lits = jinja_literals(jinja);
    for (size_t i = 0; i < lits.size(); i++) {
        const Literal &l = lits[i];
        if (l.text == "<think>\n") t.think_open = true;
        if (t.tools_head.empty() && l.text.find("# Tools") != std::string::npos && l.text.find("<tools>") != std::string::npos) {
            t.tools_head = l.text;
            // The tool list precedes the system text unless that text is written first.
            size_t open = jinja.rfind("<|im_start|>system", l.begin);
            t.tools_first = open == std::string::npos || jinja.substr(open, l.begin - open).find("content") == std::string::npos;
        } else if (t.tools_tail.empty() && !t.tools_head.empty() && l.text.find("</tools>") != std::string::npos) {
            // This literal and the ones written right after it, up to the next statement.
            size_t stop = jinja.find("{%", l.end);
            for (size_t k = i; k < lits.size() && lits[k].begin < stop; k++) t.tools_tail += lits[k].text;
            const std::string end = "<|im_end|>\n";
            if (t.tools_tail.size() >= end.size() && t.tools_tail.compare(t.tools_tail.size() - end.size(), end.size(), end) == 0) t.tools_tail.resize(t.tools_tail.size() - end.size());
        }
        {
            const std::string open = "<|im_start|>system\n", close = "<|im_end|>\n";
            if (l.text.size() > open.size() + close.size() && l.text.compare(0, open.size(), open) == 0 && l.text.compare(l.text.size() - close.size(), close.size(), close) == 0)
                t.default_system = l.text.substr(open.size(), l.text.size() - open.size() - close.size());
        }
        if (l.text.find("Reasoning effort is set to xhigh") == 0 || l.text.find("Reasoning effort is set to high") == 0) t.effort_high = l.text;
        if (l.text.find("Reasoning effort is set to low") == 0) t.effort_low = l.text;
    }
    if (t.tools_head.empty() || t.tools_tail.empty()) {
        // The template says nothing about tools: describe them the common (Hermes) way.
        t.tools_head = k_hermes_head; t.tools_tail = k_hermes_tail; t.tool_style = ToolStyle::Json; t.tools_first = false;
    }
    return t;
}

ChatTemplate ChatTemplate::detect(const GGUF &g) {
    if (!g.has("tokenizer.chat_template")) { ChatTemplate t; t.problem = "the model file has no chat template"; return t; }
    return from_text(g.get_str("tokenizer.chat_template"));
}

std::string ChatTemplate::describe() const {
    if (!supported) return "unsupported (" + problem + ")";
    std::string d = "ChatML, tool calls as ";
    d += tool_style == ToolStyle::Xml ? "<function=...> blocks" : "JSON";
    d += reasoning ? ", reasoning" : ", no reasoning";
    if (!effort_high.empty()) d += " with effort levels";
    return d;
}

std::string Prompt::text() const {
    std::string out;
    for (const Piece &p : pieces) out += p.text.s;
    return out + reply_prefix.s;
}

MarkedText render_assistant_body(const ChatTemplate &t, const Json &content, bool with_reasoning) {
    Parts p = split_assistant(content);
    MarkedText m;
    with_reasoning &= t.reasoning;
    if (t.tool_style == ChatTemplate::ToolStyle::Xml) {
        const std::string_view text = strip(p.text);
        if (with_reasoning) m.markup("<think>\n").quote(strip(p.reasoning)).markup("\n</think>\n\n");
        m.quote(text);
        for (size_t i = 0; i < p.calls.size(); i++) {
            const Json &call = *p.calls[i];
            m.markup(i > 0 ? "\n<tool_call>\n<function=" : text.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=").quote(call["name"].str()).markup(">\n");
            for (const Json::Member &arg : call["input"].members())
                m.markup("<parameter=").quote(arg.first).markup(">\n").quote(arg.second.is_string() ? arg.second.str() : arg.second.dump(true)).markup("\n</parameter>\n");
            m.markup("</function>\n</tool_call>");
        }
        return m;
    }
    std::string_view text = t.trim ? strip(p.text) : std::string_view(p.text);
    const std::string_view reasoning = strip(p.reasoning, "\n");
    if (with_reasoning && !strip(reasoning).empty()) {
        m.markup("<think>\n").quote(reasoning).markup("\n</think>\n\n");
        text = text.substr(std::min(text.find_first_not_of('\n'), text.size()));
    }
    m.quote(text);
    for (size_t i = 0; i < p.calls.size(); i++) {
        const Json &call = *p.calls[i];
        if (i > 0 || !text.empty()) m.markup("\n");
        const Json &input = call["input"];
        m.markup("<tool_call>\n{\"name\": \"").quote(call["name"].str()).markup("\", \"arguments\": ").quote(input.is_object() ? input.dump(true) : "{}").markup("}\n</tool_call>");
    }
    return m;
}

Prompt render_prompt(const ChatTemplate &t, const Json &req, const RenderOptions &o) {
    if (!t.supported) throw std::runtime_error("chat template not supported: " + t.problem);
    const Json &messages = req["messages"];
    if (!messages.is_array() || messages.items().empty()) throw std::runtime_error("messages: at least one message is required");
    const bool xml = t.tool_style == ChatTemplate::ToolStyle::Xml;
    Prompt out;

    // ---- System message: instructions and tool definitions ----
    std::string sys;
    const Json &system = req["system"];
    if (system.is_string()) sys = system.str();
    else for (const Json &b : system.items()) {
        const std::string &text = b["text"].str();
        // Claude Code's first system block is a billing note that changes with every request.
        if (b["type"].str() != "text" || text.empty() || text.compare(0, 27, "x-anthropic-billing-header:") == 0) continue;
        if (!sys.empty()) sys += "\n\n";
        sys += text;
    }
    // Leading system-role messages belong to the system text as well.
    size_t first = 0;
    for (; first < messages.items().size() && messages.items()[first]["role"].str() == "system"; first++) {
        std::string more = plain_text(messages.items()[first]["content"], "\n");
        if (!sys.empty() && !more.empty()) sys += "\n\n";
        sys += more;
    }
    if (sys.empty() && first == 0 && !system.is_string() && system.items().empty()) sys = t.default_system;
    const std::string_view sys_text = t.trim ? strip(sys) : std::string_view(sys);

    const std::string &choice = req["tool_choice"]["type"].str();
    std::vector<std::string> tool_defs;
    if (choice != "none")
        for (const Json &tool : req["tools"].items()) {
            if (!tool["input_schema"].is_object() || tool["name"].str().empty()) continue;   // server-side tools cannot run here
            if (tool["defer_loading"].boolean(false)) continue;                               // shown when a tool result refers to it
            tool_defs.push_back(tool_definition(tool));
        }
    const bool forced = !tool_defs.empty() && (choice == "any" || choice == "tool");
    const bool thinking = o.thinking && t.reasoning && !forced;
    std::string effort;
    if (thinking) effort = o.effort == "low" ? t.effort_low : o.effort == "medium" ? "" : t.effort_high;

    if (!tool_defs.empty() || !sys_text.empty() || !effort.empty()) {
        MarkedText m;
        m.markup("<|im_start|>system\n");
        if (!tool_defs.empty()) {
            if (!t.tools_first && !sys_text.empty()) m.quote(sys_text).markup(t.tools_head[0] == '\n' ? "" : "\n\n");
            if (t.tools_first && !effort.empty()) m.markup(effort).markup("\n\n");
            m.markup(t.tools_head);
            for (const std::string &def : tool_defs) m.markup("\n").quote(def);
            m.markup(t.tools_tail);
            if (t.tools_first) {
                out.tools_bytes = m.s.size();
                if (!sys_text.empty()) m.markup("\n\n").quote(sys_text);
            }
        } else {
            if (!effort.empty()) m.markup(effort).markup(sys_text.empty() ? "" : "\n\n");
            m.quote(sys_text);
        }
        m.markup("<|im_end|>\n");
        out.pieces.push_back({std::move(m), false});
        out.has_system = true;
    }

    // ---- Conversation ----
    const size_t n = messages.items().size();
    // Without keep_thinking, reasoning is only shown for the assistant messages after the last
    // real user message, as the templates do.
    size_t last_query = 0;
    for (size_t i = first; i < n; i++) {
        const Json &msg = messages.items()[i];
        if (msg["role"].str() != "user") continue;
        bool only_results = msg["content"].is_array() && !msg["content"].items().empty();
        for (const Json &b : msg["content"].items()) only_results &= b["type"].str() == "tool_result";
        if (!only_results) last_query = i;
    }
    const bool prefill = messages.items()[n - 1]["role"].str() == "assistant";
    for (size_t i = first; i < n; i++) {
        const Json &msg = messages.items()[i];
        const std::string &role = msg["role"].str();
        const Json &content = msg["content"];
        if (role == "assistant") {
            if (i == n - 1) break;   // continued below
            out.pieces.push_back({MarkedText().markup("<|im_start|>assistant\n"), false});
            out.pieces.push_back({render_assistant_body(t, content, o.keep_thinking || i > last_query), true});
            out.pieces.push_back({MarkedText().markup("<|im_end|>\n"), false});
        } else if (role == "user") {
            // Tool results form one turn; any other content of the message is a turn of its own.
            MarkedText results; std::string text;
            auto flush_text = [&] {
                if (text.empty()) return;
                out.pieces.push_back({MarkedText().markup("<|im_start|>user\n").quote(t.trim ? strip(text) : std::string_view(text)).markup("<|im_end|>\n"), false});
                text.clear();
            };
            auto flush_results = [&] {
                if (results.s.empty()) return;
                MarkedText m;
                m.markup("<|im_start|>user").append(results).markup("<|im_end|>\n");
                out.pieces.push_back({std::move(m), false});
                results = MarkedText();
            };
            if (content.is_string()) text = content.str();
            else for (const Json &b : content.items()) {
                const std::string &type = b["type"].str();
                if (type == "tool_result") {
                    flush_text();
                    const std::string body = result_text(b["content"], req["tools"]);
                    results.markup("\n<tool_response>\n").quote(t.trim ? strip(body) : std::string_view(body)).markup("\n</tool_response>");
                } else {
                    std::string piece = type == "text" ? b["text"].str() : type == "image" ? "[image]" : type == "document" ? "[document]" : "";
                    if (piece.empty()) continue;
                    flush_results();
                    if (!text.empty()) text += "\n";
                    text += piece;
                }
            }
            flush_results(); flush_text();
            if (content.is_string() && content.str().empty()) out.pieces.push_back({MarkedText().markup("<|im_start|>user\n<|im_end|>\n"), false});
        } else if (role == "system") {
            const std::string body = plain_text(content, "\n");
            out.pieces.push_back({MarkedText().markup("<|im_start|>system\n").quote(t.trim ? strip(body) : std::string_view(body)).markup("<|im_end|>\n"), false});
        } else {
            throw std::runtime_error("messages." + std::to_string(i) + ".role: expected \"user\" or \"assistant\"");
        }
    }

    // ---- Start of the reply ----
    out.pieces.push_back({MarkedText().markup("<|im_start|>assistant\n"), false});
    const char *no_think = t.reasoning ? "<think>\n\n</think>\n\n" : "";
    if (prefill) {
        Parts p = split_assistant(messages.items()[n - 1]["content"]);
        out.reply_prefix.markup(no_think).quote(p.text);
    } else if (forced) {
        const std::string &name = req["tool_choice"]["name"].str();
        std::string call = xml ? "<tool_call>\n<function=" : "<tool_call>\n{\"name\": \"";
        if (choice == "tool" && !name.empty()) call += name + (xml ? ">\n" : "\", \"arguments\": ");
        out.reply_prefix.markup(no_think).markup(call);
        out.fed_to_parser = call;
    } else if (thinking) {
        if (t.think_open) { out.reply_prefix.markup("<think>\n"); out.starts_reasoning = true; }
    } else {
        out.reply_prefix.markup(no_think);
    }
    return out;
}

}  // namespace gfx
