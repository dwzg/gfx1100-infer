#include "server.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

#include "core/tokenizer.h"
#include "engine/gpu_model.h"
#include "reply.h"

namespace gfx {

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

void logf(const char *fmt, ...) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    time_t now = time(nullptr); struct tm tm; localtime_r(&now, &tm);
    char stamp[16]; strftime(stamp, sizeof stamp, "%H:%M:%S", &tm);
    fprintf(stderr, "[%s] ", stamp);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
}

// ---- HTTP ----

struct HttpRequest {
    std::string method, path;
    std::map<std::string, std::string> headers;   // names in lower case
    std::string body;
    bool keep_alive = true;
    const std::string &header(const std::string &name) const { static const std::string none; auto it = headers.find(name); return it == headers.end() ? none : it->second; }
};

struct HttpError { int status; std::string type, message; };

const char *status_text(int s) {
    switch (s) {
        case 200: return "OK"; case 400: return "Bad Request"; case 401: return "Unauthorized"; case 404: return "Not Found"; case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large"; case 500: return "Internal Server Error"; case 503: return "Service Unavailable"; default: return "Error";
    }
}

class Conn {
  public:
    explicit Conn(int fd) : fd_(fd) {}
    ~Conn() { if (fd_ >= 0) close(fd_); }

    bool send_all(std::string_view data) {
        while (!data.empty()) {
            ssize_t n = ::send(fd_, data.data(), data.size(), MSG_NOSIGNAL);
            if (n < 0) { if (errno == EINTR) continue; return false; }
            data.remove_prefix((size_t)n);
        }
        return true;
    }
    // The client hung up (it does that to cancel a request).
    bool peer_closed() {
        struct pollfd p{fd_, POLLIN | POLLRDHUP, 0};
        if (poll(&p, 1, 0) <= 0) return false;
        if (p.revents & (POLLRDHUP | POLLHUP | POLLERR)) return true;
        char c;
        return (p.revents & POLLIN) && recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
    }
    // Reads one request. False when the connection ended before one arrived.
    bool read_request(HttpRequest &req) {
        size_t head_end;
        while ((head_end = in_.find("\r\n\r\n")) == std::string::npos) {
            if (in_.size() > (1u << 20)) throw HttpError{400, "invalid_request_error", "request headers too large"};
            if (!fill()) return false;
        }
        const std::string head = in_.substr(0, head_end);
        in_.erase(0, head_end + 4);
        size_t line_end = head.find("\r\n");
        const std::string first = head.substr(0, line_end);
        size_t sp1 = first.find(' '), sp2 = first.rfind(' ');
        if (sp1 == std::string::npos || sp2 == sp1) throw HttpError{400, "invalid_request_error", "malformed request line"};
        req.method = first.substr(0, sp1);
        req.path = first.substr(sp1 + 1, sp2 - sp1 - 1);
        req.path = req.path.substr(0, req.path.find('?'));
        req.keep_alive = first.substr(sp2 + 1) != "HTTP/1.0";
        req.headers.clear();
        for (size_t p = line_end; p != std::string::npos && p + 2 < head.size();) {
            size_t e = head.find("\r\n", p + 2);
            std::string line = head.substr(p + 2, e == std::string::npos ? std::string::npos : e - p - 2);
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(0, colon), value = line.substr(colon + 1);
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)tolower(c); });
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                req.headers[name] = value;
            }
            p = e;
        }
        std::string conn = req.header("connection");
        std::transform(conn.begin(), conn.end(), conn.begin(), [](unsigned char c) { return (char)tolower(c); });
        if (conn == "close") req.keep_alive = false;
        if (req.header("expect") == "100-continue") send_all("HTTP/1.1 100 Continue\r\n\r\n");
        req.body.clear();
        const size_t limit = 512u << 20;
        if (req.header("transfer-encoding").find("chunked") != std::string::npos) {
            for (;;) {
                size_t eol;
                while ((eol = in_.find("\r\n")) == std::string::npos) if (!fill()) return false;
                size_t len = strtoull(in_.c_str(), nullptr, 16);
                in_.erase(0, eol + 2);
                if (req.body.size() + len > limit) throw HttpError{413, "request_too_large", "request body too large"};
                while (in_.size() < len + 2) if (!fill()) return false;
                req.body.append(in_, 0, len);
                in_.erase(0, len + 2);
                if (len == 0) break;
            }
        } else if (!req.header("content-length").empty()) {
            size_t len = strtoull(req.header("content-length").c_str(), nullptr, 10);
            if (len > limit) throw HttpError{413, "request_too_large", "request body too large"};
            while (in_.size() < len) if (!fill()) return false;
            req.body = in_.substr(0, len);
            in_.erase(0, len);
        }
        return true;
    }
    void respond(int status, const std::string &content_type, const std::string &body, bool keep_alive, bool head_only = false) {
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\nContent-Type: " + content_type + "\r\nContent-Length: " + std::to_string(body.size()) +
                          "\r\nConnection: " + (keep_alive ? "keep-alive" : "close") + "\r\nrequest-id: " + random_id("req_") + "\r\n\r\n";
        if (!head_only) out += body;
        send_all(out);
    }

  private:
    bool fill() {
        char buf[65536];
        for (;;) {
            struct pollfd p{fd_, POLLIN, 0};
            int r = poll(&p, 1, 500);
            if (g_stop) return false;
            if (r < 0 && errno != EINTR) return false;
            if (r <= 0) continue;
            ssize_t n = recv(fd_, buf, sizeof buf, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            in_.append(buf, (size_t)n);
            return true;
        }
    }
    int fd_;
    std::string in_;
};

std::string error_body(const std::string &type, const std::string &message) {
    return Json::object().set("type", "error").set("error", Json::object().set("type", type).set("message", message)).set("request_id", random_id("req_")).dump();
}

// A server-sent-event response, sent with chunked transfer encoding.
class EventStream {
  public:
    explicit EventStream(Conn &c) : c_(c) {}
    bool started() const { return started_; }
    bool alive() const { return alive_; }
    void start() {
        started_ = true;
        alive_ = c_.send_all("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nTransfer-Encoding: chunked\r\nrequest-id: " +
                             random_id("req_") + "\r\n\r\n");
        last_ = Clock::now();
    }
    void event(const char *name, const Json &data) {
        std::string payload = std::string("event: ") + name + "\ndata: " + data.dump() + "\n\n";
        char len[20]; snprintf(len, sizeof len, "%zx\r\n", payload.size());
        if (alive_) alive_ = c_.send_all(std::string(len) + payload + "\r\n");
        last_ = Clock::now();
    }
    // Keeps the connection from looking dead while nothing else is being sent.
    void ping_if_quiet(double after_seconds = 5.0) {
        if (seconds_since(last_) >= after_seconds) event("ping", Json::object().set("type", "ping"));
    }
    void finish() { if (alive_) alive_ = c_.send_all("0\r\n\r\n"); }

  private:
    Conn &c_;
    bool started_ = false, alive_ = true;
    Clock::time_point last_;
};

// ---- The model behind the API ----

size_t common_prefix(const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) n++;
    return n;
}

class Service {
  public:
    Service(GpuModel &model, const Tokenizer &tok, const ChatTemplate &tmpl, const ServerOptions &opt) : tok_(tok), tmpl_(tmpl), opt_(opt), cache_(model, opt.cache) {
        if (!tmpl.body_end.empty()) body_end_ids_ = tok.encode(tmpl.body_end, false, true);
    }

    void handle(Conn &c, const HttpRequest &req) {
        const bool head = req.method == "HEAD";
        if (!opt_.api_key.empty() && req.path.compare(0, 4, "/v1/") == 0) {
            const std::string &bearer = req.header("authorization");
            if (req.header("x-api-key") != opt_.api_key && bearer != "Bearer " + opt_.api_key) throw HttpError{401, "authentication_error", "invalid x-api-key"};
        }
        if (req.path == "/v1/messages" && req.method == "POST") return messages(c, req);
        if (req.path == "/v1/messages/count_tokens" && req.method == "POST") {
            Json body = parse_body(req);
            Built b = build(body);
            return c.respond(200, "application/json", Json::object().set("input_tokens", (int64_t)b.ids.size()).dump(), req.keep_alive);
        }
        if (req.path == "/v1/models" && (req.method == "GET" || head)) {
            Json m = Json::object().set("type", "model").set("id", opt_.model_id).set("display_name", opt_.model_id).set("created_at", "2025-01-01T00:00:00Z")
                         .set("max_input_tokens", (int64_t)cache_.limit()).set("max_tokens", (int64_t)cache_.limit());
            Json list = Json::object().set("data", Json::array().push(m)).set("has_more", false).set("first_id", opt_.model_id).set("last_id", opt_.model_id);
            return c.respond(200, "application/json", list.dump(), req.keep_alive, head);
        }
        if (req.path == "/" || req.path == "/health" || req.path == "/api/hello")
            return c.respond(200, "application/json", Json::object().set("status", "ok").set("model", opt_.model_id).dump(), req.keep_alive, head);
        throw HttpError{404, "not_found_error", "no such endpoint: " + req.method + " " + req.path};
    }

  private:
    struct Built {
        Prompt prompt;
        std::vector<int32_t> ids;       // the whole prompt
        std::vector<int32_t> prefix;    // tokens of prompt.reply_prefix (the end of ids)
        std::vector<int> marks;
    };

    static Json parse_body(const HttpRequest &req) {
        try {
            Json j = Json::parse(req.body);
            if (!j.is_object()) throw std::runtime_error("the request body must be a JSON object");
            return j;
        } catch (const std::exception &e) { throw HttpError{400, "invalid_request_error", e.what()}; }
    }

    // Token ids of a piece of prompt text. The same messages come back with every request of a
    // conversation, so results are remembered.
    std::vector<int32_t> encode(const MarkedText &t) {
        std::string key = t.s;
        key += '\x01';
        for (auto [b, e] : t.literal) { key += std::to_string(b); key += '-'; key += std::to_string(e); key += ','; }
        auto it = token_cache_.find(key);
        if (it != token_cache_.end()) return it->second;
        if (token_cache_bytes_ > (256u << 20)) { token_cache_.clear(); token_cache_bytes_ = 0; }
        std::vector<int32_t> ids = tok_.encode(t.s, false, true, &t.literal);
        token_cache_bytes_ += key.size() + ids.size() * 4;
        return token_cache_.emplace(std::move(key), std::move(ids)).first->second;
    }

    Built build(const Json &body) {
        Built b;
        RenderOptions ro;
        const std::string &thinking = body["thinking"]["type"].str();
        ro.thinking = opt_.think == "on" || (opt_.think != "off" && (thinking == "enabled" || thinking == "adaptive"));
        const std::string &effort = body["output_config"]["effort"].str();
        ro.effort = !opt_.effort.empty() ? opt_.effort : effort == "low" || effort == "medium" ? effort : "high";
        ro.keep_thinking = opt_.keep_thinking;
        try { b.prompt = render_prompt(tmpl_, body, ro); } catch (const std::exception &e) { throw HttpError{400, "invalid_request_error", e.what()}; }

        std::lock_guard<std::mutex> lock(tok_mu_);
        for (size_t i = 0; i < b.prompt.pieces.size(); i++) {
            const Prompt::Piece &piece = b.prompt.pieces[i];
            if (piece.body) {
                // An earlier reply: the tokens it was generated as, if this server generated it.
                auto it = generated_.find(piece.text.s);
                if (it != generated_.end()) { b.ids.insert(b.ids.end(), it->second.begin(), it->second.end()); continue; }
            }
            const std::vector<int32_t> ids = encode(piece.text);
            if (i == 0 && b.prompt.has_system) {
                // Worth a checkpoint each: the end of the tool list (shared by conversations whose
                // instructions differ) and the end of the system message.
                if (b.prompt.tools_bytes > 0) {
                    MarkedText head;
                    head.s = piece.text.s.substr(0, b.prompt.tools_bytes);
                    for (auto [lb, le] : piece.text.literal) if (lb < head.s.size()) head.literal.emplace_back(lb, std::min(le, head.s.size()));
                    size_t same = common_prefix(encode(head), ids);
                    if (same > 64) b.marks.push_back((int)same - 2);   // the last tokens may spell differently when other text follows
                }
                b.marks.push_back((int)ids.size());
            }
            b.ids.insert(b.ids.end(), ids.begin(), ids.end());
            if (i < b.prompt.mark_pieces) b.marks.push_back((int)b.ids.size());
        }
        b.prefix = encode(b.prompt.reply_prefix);
        b.ids.insert(b.ids.end(), b.prefix.begin(), b.prefix.end());
        return b;
    }

    void remember(const Json &blocks, std::vector<int32_t> tokens) {
        std::string key = render_assistant_body(tmpl_, blocks, true).s;
        std::lock_guard<std::mutex> lock(tok_mu_);
        if (generated_.emplace(key, std::move(tokens)).second) generated_order_.push_back(std::move(key));
        while (generated_order_.size() > 4096) { generated_.erase(generated_order_.front()); generated_order_.pop_front(); }
    }

    void messages(Conn &c, const HttpRequest &req) {
        const Clock::time_point t_start = Clock::now();
        if (!opt_.dump_dir.empty()) {
            static std::atomic<int> serial{0};
            char name[64]; snprintf(name, sizeof name, "/request-%04d.json", ++serial);
            if (FILE *f = fopen((opt_.dump_dir + name).c_str(), "wb")) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
        }
        const Json body = parse_body(req);
        const bool stream = body["stream"].boolean(false);
        const std::string model = body["model"].is_string() ? body["model"].str() : opt_.model_id;
        Built b = build(body);
        const int n_prompt = (int)b.ids.size();
        if (n_prompt + 1 > cache_.limit()) throw ContextFull(n_prompt, cache_.limit() - 1);
        int max_tokens = (int)body["max_tokens"].integer(opt_.default_max_tokens);
        if (max_tokens < 1) throw HttpError{400, "invalid_request_error", "max_tokens: must be at least 1"};
        std::vector<std::string> stops;
        for (const Json &s : body["stop_sequences"].items()) if (s.is_string()) stops.push_back(s.str());
        SamplerParams sp = opt_.sampling;
        if (opt_.client_sampling) {
            if (body["temperature"].is_number()) sp.temperature = (float)body["temperature"].number();
            if (body["top_p"].is_number()) sp.top_p = (float)body["top_p"].number();
            if (body["top_k"].is_number()) sp.top_k = (int)body["top_k"].integer();
        }
        sp.seed = std::random_device{}();
        if (opt_.log_prompts) logf("prompt:\n%s", b.prompt.text().c_str());

        EventStream es(c);
        if (stream) es.start();
        bool cancelled = false;
        auto gone = [&] { return g_stop || c.peer_closed() || (stream && !es.alive()); };

        // One request uses the GPU at a time; the others wait their turn.
        std::unique_lock<std::mutex> gpu(gpu_mu_, std::try_to_lock);
        while (!gpu.owns_lock() && !gpu.try_lock()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (gone()) { logf("request cancelled while waiting for the GPU"); return; }
            if (stream) es.ping_if_quiet();
        }
        const double waited = seconds_since(t_start);

        const std::string id = random_id("msg_");
        Json usage = Json::object();
        Json blocks = Json::array();
        std::string stop_reason = "end_turn", stop_sequence;
        int n_out = 0;
        ContextCache::Result prep;
        double t_prompt = 0, t_gen = 0;
        try {
            const Clock::time_point t0 = Clock::now();
            prep = cache_.prepare(b.ids, b.marks, [&](int, int) {
                if (stream) es.ping_if_quiet();
                return !gone();
            });
            t_prompt = seconds_since(t0);
            cancelled = prep.cancelled;
            usage.set("input_tokens", (int64_t)(n_prompt - prep.reused)).set("cache_creation_input_tokens", 0).set("cache_read_input_tokens", (int64_t)prep.reused).set("output_tokens", 0);
            if (stream && !cancelled)
                es.event("message_start", Json::object().set("type", "message_start").set("message",
                         Json::object().set("id", id).set("type", "message").set("role", "assistant").set("model", model).set("content", Json::array())
                             .set("stop_reason", nullptr).set("stop_sequence", nullptr).set("usage", usage)));

            ReplyParser parser(tmpl_, body["tools"], b.prompt.starts_reasoning, stops);
            Sampler sampler(sp);
            std::vector<int32_t> raw = b.prefix;     // the reply's body as tokens, for remember()
            std::vector<ReplyEvent> events;
            int index = -1; bool open_thinking = false;
            auto deliver = [&] {
                if (stream) for (const ReplyEvent &e : events) {
                    switch (e.kind) {
                        case ReplyEvent::ThinkingStart: case ReplyEvent::TextStart: case ReplyEvent::ToolStart: {
                            Json block = Json::object();
                            if (e.kind == ReplyEvent::ThinkingStart) block.set("type", "thinking").set("thinking", "").set("signature", "");
                            else if (e.kind == ReplyEvent::TextStart) block.set("type", "text").set("text", "");
                            else block.set("type", "tool_use").set("id", e.id).set("name", e.text).set("input", Json::object());
                            open_thinking = e.kind == ReplyEvent::ThinkingStart;
                            es.event("content_block_start", Json::object().set("type", "content_block_start").set("index", ++index).set("content_block", block));
                            break;
                        }
                        case ReplyEvent::ThinkingDelta: case ReplyEvent::TextDelta: case ReplyEvent::ToolDelta: {
                            Json delta = Json::object();
                            if (e.kind == ReplyEvent::ThinkingDelta) delta.set("type", "thinking_delta").set("thinking", e.text);
                            else if (e.kind == ReplyEvent::TextDelta) delta.set("type", "text_delta").set("text", e.text);
                            else delta.set("type", "input_json_delta").set("partial_json", e.text);
                            es.event("content_block_delta", Json::object().set("type", "content_block_delta").set("index", index).set("delta", delta));
                            break;
                        }
                        case ReplyEvent::BlockStop:
                            if (open_thinking) es.event("content_block_delta", Json::object().set("type", "content_block_delta").set("index", index).set("delta",
                                                        Json::object().set("type", "signature_delta").set("signature", "local")));
                            open_thinking = false;
                            es.event("content_block_stop", Json::object().set("type", "content_block_stop").set("index", index));
                            break;
                    }
                }
                events.clear();
            };
            if (!b.prompt.fed_to_parser.empty()) { parser.feed(b.prompt.fed_to_parser, events); deliver(); }

            const Clock::time_point t1 = Clock::now();
            bool complete = false;
            while (!cancelled) {
                if (n_out >= max_tokens) { stop_reason = "max_tokens"; break; }
                const int32_t token = sampler.sample(cache_.logits().data(), (int)cache_.logits().size());
                if (tok_.is_eog(token)) { complete = true; break; }
                n_out++;
                raw.push_back(token);
                parser.feed(tok_.piece(token), events);
                deliver();
                if (parser.stopped()) { stop_reason = "stop_sequence"; stop_sequence = parser.stop_sequence(); break; }
                if (gone()) { cancelled = true; break; }
                if (stream) es.ping_if_quiet();
                if (!cache_.push(token)) { stop_reason = "max_tokens"; break; }   // the context is full
            }
            t_gen = seconds_since(t1);
            parser.finish(complete, events);
            deliver();
            if (complete && parser.has_tool_call()) stop_reason = "tool_use";
            blocks = parser.blocks();
            if (complete) raw.insert(raw.end(), body_end_ids_.begin(), body_end_ids_.end());   // part of the message as it is sent back
            if (!cancelled) remember(blocks, std::move(raw));
        } catch (const ContextFull &e) {
            if (!stream) throw;
            es.event("error", Json::object().set("type", "error").set("error", Json::object().set("type", "invalid_request_error").set("message", e.what())));
            es.finish();
            return;
        } catch (const std::exception &e) {
            // The engine may be anywhere now; start from a clean slate next time.
            try { cache_.clear(); } catch (const std::exception &) {}
            logf("error: %s", e.what());
            if (!stream) throw HttpError{500, "api_error", e.what()};
            es.event("error", Json::object().set("type", "error").set("error", Json::object().set("type", "api_error").set("message", e.what())));
            es.finish();
            return;
        }
        const std::string state = cache_.summary();
        gpu.unlock();

        logf("%s%d prompt tokens (%d reused, %d processed in %.1f s%s), %d generated in %.1f s (%.1f tok/s), %s; %s", cancelled ? "cancelled: " : "", n_prompt, prep.reused, prep.evaluated,
             t_prompt, waited > 0.5 ? (", waited " + std::to_string((int)waited) + " s").c_str() : "", n_out, t_gen, n_out > 1 && t_gen > 0 ? n_out / t_gen : 0.0, stop_reason.c_str(), state.c_str());
        if (opt_.log_prompts) logf("reply:\n%s", blocks.dump().c_str());
        if (cancelled) return;

        usage.set("output_tokens", (int64_t)n_out);
        const Json stop_seq = stop_sequence.empty() ? Json() : Json(stop_sequence);
        if (stream) {
            es.event("message_delta", Json::object().set("type", "message_delta").set("delta", Json::object().set("stop_reason", stop_reason).set("stop_sequence", stop_seq)).set("usage", usage));
            es.event("message_stop", Json::object().set("type", "message_stop"));
            es.finish();
        } else {
            Json msg = Json::object().set("id", id).set("type", "message").set("role", "assistant").set("model", model).set("content", blocks)
                           .set("stop_reason", stop_reason).set("stop_sequence", stop_seq).set("usage", usage);
            c.respond(200, "application/json", msg.dump(), req.keep_alive);
        }
    }

    const Tokenizer &tok_;
    const ChatTemplate &tmpl_;
    const ServerOptions &opt_;
    ContextCache cache_;
    std::mutex gpu_mu_, tok_mu_;
    std::unordered_map<std::string, std::vector<int32_t>> token_cache_;
    size_t token_cache_bytes_ = 0;
    std::unordered_map<std::string, std::vector<int32_t>> generated_;   // reply body text -> the tokens it was generated as
    std::deque<std::string> generated_order_;
    std::vector<int32_t> body_end_ids_;
};

void serve_connection(Service &svc, int fd) {
    Conn c(fd);
    for (;;) {
        HttpRequest req;
        try {
            if (!c.read_request(req)) return;
            svc.handle(c, req);
        } catch (const HttpError &e) {
            c.respond(e.status, "application/json", error_body(e.type, e.message), false);
            return;
        } catch (const ContextFull &e) {
            // Worded as upstream words it, so clients recognise it and shorten the conversation.
            c.respond(400, "application/json", error_body("invalid_request_error", e.what()), false);
            return;
        } catch (const std::exception &e) {
            logf("error: %s", e.what());
            c.respond(500, "application/json", error_body("api_error", e.what()), false);
            return;
        }
        if (!req.keep_alive) return;
    }
}

}  // namespace

int serve(GpuModel &model, const Tokenizer &tok, const ChatTemplate &tmpl, const ServerOptions &opt) {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(opt.host.c_str(), std::to_string(opt.port).c_str(), &hints, &res) != 0 || !res) throw std::runtime_error("cannot resolve " + opt.host);
    int ls = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (ls < 0 || bind(ls, res->ai_addr, res->ai_addrlen) != 0 || listen(ls, 64) != 0) {
        std::string why = strerror(errno);
        freeaddrinfo(res);
        if (ls >= 0) close(ls);
        throw std::runtime_error("cannot listen on " + opt.host + ":" + std::to_string(opt.port) + ": " + why);
    }
    freeaddrinfo(res);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{}; sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);

    Service svc(model, tok, tmpl, opt);
    std::atomic<int> active{0};
    logf("listening on http://%s:%d (model id \"%s\", context %d tokens)", opt.host.c_str(), opt.port, opt.model_id.c_str(), model.n_ctx());
    while (!g_stop) {
        struct pollfd p{ls, POLLIN, 0};
        if (poll(&p, 1, 200) <= 0) continue;
        int fd = accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        active++;
        std::thread([&svc, &active, fd] { serve_connection(svc, fd); active--; }).detach();
    }
    close(ls);
    logf("shutting down");
    for (int i = 0; i < 100 && active > 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

}  // namespace gfx
