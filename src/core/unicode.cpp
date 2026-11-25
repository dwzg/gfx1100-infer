#include "unicode.h"

#include <algorithm>
#include <cstring>

namespace gfx {

#include "unicode_data.inc"

static const size_t k_n_ranges = sizeof(k_cat_ranges) / sizeof(k_cat_ranges[0]);
static const uint8_t k_cat_cn = sizeof(k_cat_names) / sizeof(k_cat_names[0]) - 1;

uint8_t unicode_category(uint32_t cp) {
    size_t lo = 0, hi = k_n_ranges;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (k_cat_ranges[mid].last < cp) lo = mid + 1; else hi = mid;
    }
    return (lo < k_n_ranges && k_cat_ranges[lo].first <= cp) ? k_cat_ranges[lo].cat : k_cat_cn;
}
const char *unicode_category_name(uint8_t cat) { return k_cat_names[cat <= k_cat_cn ? cat : k_cat_cn]; }
bool unicode_is_space(uint32_t cp) {
    for (uint32_t w : k_white_space) if (w == cp) return true;
    return false;
}

static uint32_t decode_one(const unsigned char *p, size_t n, size_t *len) {
    unsigned char c = p[0];
    *len = 1;
    if (c < 0x80) return c;
    int need = (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
    if (need < 0 || (size_t)need >= n) return 0xFFFD;
    uint32_t cp = c & (0x3F >> need);
    for (int k = 1; k <= need; k++) { if ((p[k] & 0xC0) != 0x80) return 0xFFFD; cp = (cp << 6) | (p[k] & 0x3F); }
    static const uint32_t k_min[] = {0, 0x80, 0x800, 0x10000};
    if (cp < k_min[need] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0xFFFD;
    *len = (size_t)need + 1;
    return cp;
}

std::vector<uint32_t> utf8_decode(std::string_view s, std::vector<uint32_t> &offsets) {
    std::vector<uint32_t> out; out.reserve(s.size()); offsets.clear(); offsets.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size();) {
        size_t len; uint32_t cp = decode_one((const unsigned char *)s.data() + i, s.size() - i, &len);
        offsets.push_back((uint32_t)i); out.push_back(cp); i += len;
    }
    offsets.push_back((uint32_t)s.size());
    return out;
}
std::vector<uint32_t> utf8_decode(std::string_view s) { std::vector<uint32_t> off; return utf8_decode(s, off); }

void utf8_append(std::string &out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
}

size_t utf8_complete_prefix(std::string_view s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;                       // continuation byte, keep looking for the lead
        int need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return (size_t)need > back ? n - back : n;
    }
    return n;
}

}  // namespace gfx
