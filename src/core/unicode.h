// UTF-8 helpers and Unicode general categories.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

// General category as an index into the table of two-letter names ("Lu", "Ll", ...); the major
// class is the first letter. Unassigned code points report "Cn".
uint8_t unicode_category(uint32_t cp);
const char *unicode_category_name(uint8_t cat);
bool unicode_is_space(uint32_t cp);  // Unicode White_Space

// Decode UTF-8; invalid bytes decode to U+FFFD one byte at a time.
std::vector<uint32_t> utf8_decode(std::string_view s);
// Same, also recording the byte offset of each code point (offsets has size()+1 entries).
std::vector<uint32_t> utf8_decode(std::string_view s, std::vector<uint32_t> &offsets);
void utf8_append(std::string &out, uint32_t cp);
// Length in bytes of the longest prefix of s that does not end inside a multi-byte sequence.
size_t utf8_complete_prefix(std::string_view s);

}  // namespace gfx
