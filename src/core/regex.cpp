#include "regex.h"

#include <functional>
#include <stdexcept>

#include "unicode.h"

namespace gfx {

namespace {

struct ClassItem {
    enum Kind { Range, Category, Space, Digit } kind = Range;
    uint32_t lo = 0, hi = 0;   // Range
    char major = 0, minor = 0; // Category: major letter, optional minor letter
    bool negate = false;       // \P{..} or \S or \D
    bool test(uint32_t c) const {
        bool r = false;
        switch (kind) {
            case Range: r = c >= lo && c <= hi; break;
            case Category: { const char *n = unicode_category_name(unicode_category(c)); r = n[0] == major && (!minor || n[1] == minor); break; }
            case Space: r = unicode_is_space(c); break;
            case Digit: r = c >= '0' && c <= '9'; break;
        }
        return r != negate;
    }
};

struct Node;
using Seq = std::vector<Node>;

struct Node {
    enum Kind { Class, Any, Group, LookAhead, NegLookAhead } kind = Class;
    std::vector<ClassItem> items; bool negate = false;  // Class
    std::vector<Seq> alts;                               // Group / lookaheads
    uint32_t min = 1, max = 1;                           // quantifier (UINT32_MAX = unbounded)
    bool single(uint32_t c) const {
        if (kind == Any) return c != '\n';
        for (const auto &it : items) if (it.test(c)) return !negate;
        return negate;
    }
};

struct Parser {
    std::vector<uint32_t> p; size_t i = 0;
    [[noreturn]] void fail(const char *what) const { throw std::runtime_error(std::string("regex: ") + what); }
    bool eof() const { return i >= p.size(); }
    uint32_t peek() const { return p[i]; }

    ClassItem category(bool negate) {
        if (eof() || p[i] != '{') fail("expected { after \\p");
        i++; ClassItem it; it.kind = ClassItem::Category; it.negate = negate;
        if (eof()) fail("bad \\p{}");
        it.major = (char)p[i++];
        if (!eof() && p[i] != '}') it.minor = (char)p[i++];
        if (eof() || p[i] != '}') fail("bad \\p{}");
        i++; return it;
    }
    // Parses an escape (after the backslash). Returns true and fills `it` for class-like escapes,
    // otherwise sets `lit` to the literal code point.
    bool escape(ClassItem &it, uint32_t &lit) {
        if (eof()) fail("trailing backslash");
        uint32_t c = p[i++];
        switch (c) {
            case 'p': it = category(false); return true;
            case 'P': it = category(true); return true;
            case 's': it = ClassItem{ClassItem::Space, 0, 0, 0, 0, false}; return true;
            case 'S': it = ClassItem{ClassItem::Space, 0, 0, 0, 0, true}; return true;
            case 'd': it = ClassItem{ClassItem::Digit, 0, 0, 0, 0, false}; return true;
            case 'D': it = ClassItem{ClassItem::Digit, 0, 0, 0, 0, true}; return true;
            case 'n': lit = '\n'; return false;
            case 'r': lit = '\r'; return false;
            case 't': lit = '\t'; return false;
            default:
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) fail("unsupported escape");
                lit = c; return false;
        }
    }
    Node char_class() {
        Node n; n.kind = Node::Class;
        if (!eof() && p[i] == '^') { n.negate = true; i++; }
        bool first = true;
        while (true) {
            if (eof()) fail("unterminated [");
            uint32_t c = p[i++];
            if (c == ']' && !first) break;
            first = false;
            ClassItem it; uint32_t lo = c;
            if (c == '\\') { if (escape(it, lo)) { n.items.push_back(it); continue; } }
            uint32_t hi = lo;
            if (i + 1 < p.size() && p[i] == '-' && p[i + 1] != ']') {
                i++; hi = p[i++];
                if (hi == '\\') { ClassItem tmp; if (escape(tmp, hi)) fail("bad range"); }
                if (hi < lo) fail("bad range");
            }
            it.kind = ClassItem::Range; it.lo = lo; it.hi = hi; n.items.push_back(it);
        }
        return n;
    }
    void quantifier(Node &n) {
        if (eof()) return;
        uint32_t c = p[i];
        if (c == '?') { n.min = 0; n.max = 1; i++; }
        else if (c == '*') { n.min = 0; n.max = UINT32_MAX; i++; }
        else if (c == '+') { n.min = 1; n.max = UINT32_MAX; i++; }
        else if (c == '{') {
            size_t save = i++; uint32_t a = 0, b = 0; bool any = false, comma = false, anyb = false;
            while (!eof() && p[i] >= '0' && p[i] <= '9') { a = a * 10 + (p[i++] - '0'); any = true; }
            if (!eof() && p[i] == ',') { comma = true; i++; while (!eof() && p[i] >= '0' && p[i] <= '9') { b = b * 10 + (p[i++] - '0'); anyb = true; } }
            if (!any || eof() || p[i] != '}') { i = save; return; }
            i++; n.min = a; n.max = comma ? (anyb ? b : UINT32_MAX) : a;
        } else return;
        if (!eof() && (p[i] == '?' || p[i] == '+')) fail("lazy and possessive quantifiers are not supported");
    }
    std::vector<Seq> alternation() {
        std::vector<Seq> alts(1);
        while (!eof() && p[i] != ')') {
            uint32_t c = p[i++];
            if (c == '|') { alts.emplace_back(); continue; }
            Node n;
            if (c == '(') {
                n.kind = Node::Group;
                if (i + 1 < p.size() && p[i] == '?') {
                    uint32_t k = p[i + 1];
                    if (k == ':') i += 2;
                    else if (k == '=') { n.kind = Node::LookAhead; i += 2; }
                    else if (k == '!') { n.kind = Node::NegLookAhead; i += 2; }
                    else fail("unsupported group type");
                }
                n.alts = alternation();
                if (eof() || p[i] != ')') fail("unterminated group");
                i++;
            } else if (c == '[') n = char_class();
            else if (c == '.') n.kind = Node::Any;
            else if (c == '\\') {
                ClassItem it; uint32_t lit;
                if (!escape(it, lit)) { it.kind = ClassItem::Range; it.lo = it.hi = lit; }
                n.items.push_back(it);
            } else if (c == '*' || c == '+' || c == '?') fail("nothing to repeat");
            else { ClassItem it; it.kind = ClassItem::Range; it.lo = it.hi = c; n.items.push_back(it); }
            if (n.kind != Node::LookAhead && n.kind != Node::NegLookAhead) quantifier(n);
            alts.back().push_back(std::move(n));
        }
        return alts;
    }
};

using Cont = std::function<long(size_t)>;

long match_seq(const Seq &seq, size_t idx, const std::vector<uint32_t> &t, size_t pos, const Cont &k);

long match_alts(const std::vector<Seq> &alts, const std::vector<uint32_t> &t, size_t pos, const Cont &k) {
    for (const Seq &a : alts) { long r = match_seq(a, 0, t, pos, k); if (r >= 0) return r; }
    return -1;
}

// Repeat a group between min and max times, greedily.
long match_group_rep(const Node &n, uint32_t count, const std::vector<uint32_t> &t, size_t pos, const Cont &k) {
    if (count < n.max) {
        long r = match_alts(n.alts, t, pos, [&](size_t p2) -> long {
            if (p2 == pos) return -1;  // empty iteration: stop repeating
            return match_group_rep(n, count + 1, t, p2, k);
        });
        if (r >= 0) return r;
    }
    return count >= n.min ? k(pos) : -1;
}

long match_seq(const Seq &seq, size_t idx, const std::vector<uint32_t> &t, size_t pos, const Cont &k) {
    if (idx == seq.size()) return k(pos);
    const Node &n = seq[idx];
    auto rest = [&](size_t p2) -> long { return match_seq(seq, idx + 1, t, p2, k); };
    switch (n.kind) {
        case Node::LookAhead:
        case Node::NegLookAhead: {
            bool ok = match_alts(n.alts, t, pos, [](size_t p2) -> long { return (long)p2; }) >= 0;
            return ok == (n.kind == Node::LookAhead) ? rest(pos) : -1;
        }
        case Node::Group:
            return match_group_rep(n, 0, t, pos, rest);
        default: {
            // Single code point node: consume greedily without recursion, then back off.
            size_t m = 0;
            while (m < n.max && pos + m < t.size() && n.single(t[pos + m])) m++;
            if (m < n.min) return -1;
            for (size_t take = m;; take--) {
                long r = rest(pos + take);
                if (r >= 0) return r;
                if (take == n.min) return -1;
            }
        }
    }
}

}  // namespace

struct Regex::Impl { std::vector<Seq> alts; };

Regex::Regex(std::string_view pattern) : impl_(new Impl) {
    Parser ps; ps.p = utf8_decode(pattern);
    impl_->alts = ps.alternation();
    if (!ps.eof()) throw std::runtime_error("regex: unbalanced )");
}
Regex::~Regex() = default;
Regex::Regex(Regex &&) noexcept = default;

long Regex::match_at(const std::vector<uint32_t> &text, size_t pos) const {
    long end = match_alts(impl_->alts, text, pos, [](size_t p2) -> long { return (long)p2; });
    return end < 0 ? -1 : end - (long)pos;
}

std::vector<std::pair<uint32_t, uint32_t>> regex_split(const Regex &re, const std::vector<uint32_t> &text, uint32_t begin, uint32_t end) {
    // Match against a view limited to [begin, end) so lookahead cannot see past the piece.
    std::vector<uint32_t> view(text.begin() + begin, text.begin() + end);
    std::vector<std::pair<uint32_t, uint32_t>> out;
    uint32_t gap = 0, pos = 0, n = (uint32_t)view.size();
    while (pos < n) {
        long len = re.match_at(view, pos);
        if (len <= 0) { pos++; continue; }
        if (gap < pos) out.emplace_back(begin + gap, begin + pos);
        out.emplace_back(begin + pos, begin + pos + (uint32_t)len);
        pos += (uint32_t)len; gap = pos;
    }
    if (gap < n) out.emplace_back(begin + gap, begin + n);
    return out;
}

}  // namespace gfx
