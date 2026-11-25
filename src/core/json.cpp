#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace gfx {

namespace {
const Json k_null;
const std::string k_empty;

struct Parser {
    std::string_view t; size_t p = 0; int depth = 0;
    [[noreturn]] void fail(const char *what) const { throw std::runtime_error(std::string("JSON: ") + what + " at byte " + std::to_string(p)); }
    void ws() { while (p < t.size() && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++; }
    static void put_utf8(std::string &o, uint32_t cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    }
    uint32_t hex4() {
        if (p + 4 > t.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = t[p++]; v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10; else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else fail("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        std::string o;
        p++;   // opening quote
        for (;;) {
            size_t run = p;
            while (p < t.size() && t[p] != '"' && t[p] != '\\') p++;
            o.append(t.substr(run, p - run));
            if (p >= t.size()) fail("unterminated string");
            if (t[p++] == '"') return o;
            if (p >= t.size()) fail("unterminated string");
            char c = t[p++];
            switch (c) {
                case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
                case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break; case 'r': o += '\r'; break; case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < t.size() && t[p] == '\\' && t[p + 1] == 'u') {
                        size_t save = p; p += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); else p = save;
                    }
                    if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFD;   // lone surrogate
                    put_utf8(o, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json value() {
        ws();
        if (p >= t.size()) fail("unexpected end");
        if (++depth > 256) fail("nesting too deep");
        Json out;
        char c = t[p];
        if (c == '{') {
            out = Json::object(); p++; ws();
            if (p < t.size() && t[p] == '}') p++;
            else for (;;) {
                ws();
                if (p >= t.size() || t[p] != '"') fail("expected a member name");
                std::string k = string();
                ws();
                if (p >= t.size() || t[p++] != ':') fail("expected ':'");
                Json v = value();
                out.set(std::move(k), std::move(v));
                ws();
                if (p >= t.size()) fail("unterminated object");
                if (t[p] == ',') { p++; continue; }
                if (t[p] == '}') { p++; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            out = Json::array(); p++; ws();
            if (p < t.size() && t[p] == ']') p++;
            else for (;;) {
                out.push(value());
                ws();
                if (p >= t.size()) fail("unterminated array");
                if (t[p] == ',') { p++; continue; }
                if (t[p] == ']') { p++; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            out = Json(string());
        } else if (t.compare(p, 4, "true") == 0) { p += 4; out = Json(true); }
        else if (t.compare(p, 5, "false") == 0) { p += 5; out = Json(false); }
        else if (t.compare(p, 4, "null") == 0) { p += 4; }
        else if (c == '-' || (c >= '0' && c <= '9')) {
            size_t b = p;
            if (t[p] == '-') p++;
            if (p >= t.size() || t[p] < '0' || t[p] > '9') fail("bad number");
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++;
            if (p < t.size() && t[p] == '.') { p++; if (p >= t.size() || t[p] < '0' || t[p] > '9') fail("bad number"); while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++; }
            if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
                p++; if (p < t.size() && (t[p] == '+' || t[p] == '-')) p++;
                if (p >= t.size() || t[p] < '0' || t[p] > '9') fail("bad number");
                while (p < t.size() && t[p] >= '0' && t[p] <= '9') p++;
            }
            out = Json::parse_number_(t.substr(b, p - b));
        } else fail("unexpected character");
        depth--;
        return out;
    }
};
}  // namespace

Json Json::parse_number_(std::string_view lexeme) {
    Json j; j.type_ = Number; j.s_ = std::string(lexeme); j.n_ = strtod(j.s_.c_str(), nullptr);
    return j;
}

Json::Json(double v) : type_(Number), n_(v) {
    if (!std::isfinite(v)) { type_ = Null; return; }
    char buf[40];
    if (v == std::floor(v) && std::fabs(v) < 1e15) snprintf(buf, sizeof buf, "%.0f", v);
    else snprintf(buf, sizeof buf, "%.17g", v);
    s_ = buf;
}

Json Json::parse(std::string_view text) {
    Parser ps{text};
    Json v = ps.value();
    ps.ws();
    if (ps.p != text.size()) ps.fail("trailing characters");
    return v;
}

const std::string &Json::str() const { return type_ == String ? s_ : k_empty; }

const Json &Json::operator[](std::string_view key) const {
    if (type_ == Object) for (const Member &m : o_) if (m.first == key) return m.second;
    return k_null;
}
bool Json::has(std::string_view key) const {
    if (type_ == Object) for (const Member &m : o_) if (m.first == key) return true;
    return false;
}
Json &Json::set(std::string key, Json v) {
    if (type_ != Object) { *this = object(); }
    for (Member &m : o_) if (m.first == key) { m.second = std::move(v); return *this; }
    o_.emplace_back(std::move(key), std::move(v));
    return *this;
}
Json &Json::push(Json v) {
    if (type_ != Array) { *this = array(); }
    a_.push_back(std::move(v));
    return *this;
}

void json_quote(std::string &out, std::string_view s) {
    static const char hex[] = "0123456789abcdef";
    out += '"';
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            switch (c) {
                case '"': out += "\\\""; break; case '\\': out += "\\\\"; break; case '\n': out += "\\n"; break; case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break; case '\b': out += "\\b"; break; case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; } else out += (char)c;
            }
            i++; continue;
        }
        // Copy a well-formed multi-byte sequence, replace anything else.
        size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 ? 2 : 0;
        bool ok = len != 0 && i + len <= s.size() && c < 0xF5;
        for (size_t k = 1; ok && k < len; k++) ok = ((unsigned char)s[i + k] & 0xC0) == 0x80;
        if (ok && len == 3) { uint32_t cp = ((c & 15u) << 12) | (((unsigned char)s[i + 1] & 63u) << 6); ok = cp >= 0x800 && !(cp >= 0xD800 && cp < 0xE000); }
        if (ok && len == 4) { uint32_t cp = ((c & 7u) << 18) | (((unsigned char)s[i + 1] & 63u) << 12); ok = cp >= 0x10000 && cp < 0x110000; }
        if (ok) { out.append(s.substr(i, len)); i += len; }
        else { out += "\xEF\xBF\xBD"; i++; }
    }
    out += '"';
}

void Json::dump_to(std::string &out, bool spaced) const {
    switch (type_) {
        case Null: out += "null"; break;
        case Bool: out += b_ ? "true" : "false"; break;
        case Number: out += s_; break;
        case String: json_quote(out, s_); break;
        case Array:
            out += '[';
            for (size_t i = 0; i < a_.size(); i++) { if (i) out += spaced ? ", " : ","; a_[i].dump_to(out, spaced); }
            out += ']';
            break;
        case Object:
            out += '{';
            for (size_t i = 0; i < o_.size(); i++) {
                if (i) out += spaced ? ", " : ",";
                json_quote(out, o_[i].first);
                out += spaced ? ": " : ":";
                o_[i].second.dump_to(out, spaced);
            }
            out += '}';
            break;
    }
}

std::string Json::dump(bool spaced) const { std::string o; dump_to(o, spaced); return o; }

}  // namespace gfx
