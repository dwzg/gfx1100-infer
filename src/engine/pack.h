// GPU weight layouts.
//
// Quantised matrices keep exactly the bits of their GGUF encoding, re-laid-out for the GPU:
//   * each row is cut into "super-blocks" of SB columns, each stored as NW 32-bit words;
//   * storage is plane-major: word k of super-block s of row r lives at [(s * NW + k) * rows + r].
// With one lane per output row, the 64 lanes of a wave therefore read 64 adjacent words for every
// load, and the input vector is the same for all lanes (a scalar load).
#pragma once
#include <cstdint>
#include <vector>

#include "core/gguf.h"

namespace gfx {

struct PackFormat {
    uint32_t type;        // GGUF tensor type
    const char *kernel;   // matvec kernel name
    uint32_t sb_cols;     // columns per super-block
    uint32_t sb_words;    // 32-bit words per super-block per row
    uint32_t src_blocks;  // GGUF blocks per super-block
    bool needs_table;     // kernel reads a codebook buffer at binding 5 (see pack_table)
};

// Native layout for a tensor type, or nullptr if it has none (such tensors are stored as fp16).
const PackFormat *pack_format(uint32_t type);

// Codebook a format's kernel needs as a buffer of `count` words, or nullptr.
const uint32_t *pack_table(const PackFormat &f, uint32_t *count);

// Transcode rows [row0, row0 + rows) of a matrix with n_in columns into the plane-major layout.
// `rows_pad` (>= rows, a multiple of 64) is the plane stride; padding rows are zero.
// out must hold (n_in / sb_cols) * sb_words * rows_pad words.
void pack_rows(const PackFormat &f, const uint8_t *src, uint32_t n_in, uint64_t row0, uint32_t rows, uint32_t rows_pad, uint32_t *out);

}  // namespace gfx
