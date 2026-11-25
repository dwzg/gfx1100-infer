#include "pack.h"

#include <cstring>

#include "core/quant.h"

namespace gfx {

namespace {
const PackFormat k_formats[] = {
    {T_Q4_K, "matvec_q4_k", 256, 36, 1, false},
    {T_Q5_K, "matvec_q5_k", 256, 44, 1, false},
    {T_Q6_K, "matvec_q6_k", 256, 53, 1, false},
    {T_Q3_K, "matvec_q3_k", 256, 28, 1, false},
    {T_IQ4_XS, "matvec_iq4_xs", 256, 34, 1, false},
    {T_IQ4_NL, "matvec_iq4_nl", 64, 9, 2, false},
    {T_Q4_0, "matvec_q4_0", 64, 9, 2, false},
    {T_Q8_0, "matvec_q8_0", 64, 17, 2, false},
    {T_NVFP4, "matvec_nvfp4", 64, 9, 1, false},
    {T_IQ3_S, "matvec_iq3_s", 256, 28, 1, true},
    {T_IQ3_XXS, "matvec_iq3_xxs", 256, 25, 1, true},
    {T_IQ2_S, "matvec_iq2_s", 256, 21, 1, true},
    {T_IQ2_XS, "matvec_iq2_xs", 256, 19, 1, true},
    {T_IQ2_XXS, "matvec_iq2_xxs", 256, 17, 1, true},
};

inline void words(uint32_t *dst, const uint8_t *src, size_t nbytes) {
    // Little-endian bytes -> words; a trailing partial word is zero padded.
    size_t full = nbytes / 4;
    memcpy(dst, src, full * 4);
    if (nbytes % 4) { uint32_t w = 0; memcpy(&w, src + full * 4, nbytes % 4); dst[full] = w; }
}

// Fill `w` (sb_words words) from the GGUF block(s) of one super-block.
void pack_superblock(const PackFormat &f, const uint8_t *b, uint32_t *w) {
    switch (f.type) {
        case T_Q4_K:    // d, dmin | scales[12] | qs[128]   (already word aligned)
        case T_Q5_K:    // d, dmin | scales[12] | qh[32] | qs[128]
        case T_IQ4_XS:  // d, scales_h | scales_l[4] | qs[128]
        case T_NVFP4:   // d[4] | qs[32]
            memcpy(w, b, (size_t)f.sb_words * 4); break;
        case T_Q6_K:    // ql[128] | qh[64] | scales[16] | d
            words(w, b, 210); break;
        case T_Q3_K:    // hmask[32] | qs[64] | scales[12] | d
            words(w, b, 110); break;
        case T_IQ4_NL:
        case T_Q4_0: {  // two blocks of (d, qs[16]) -> d0 | d1<<16, qs0[16], qs1[16]
            uint16_t d0, d1; memcpy(&d0, b, 2); memcpy(&d1, b + 18, 2);
            w[0] = (uint32_t)d0 | ((uint32_t)d1 << 16);
            memcpy(w + 1, b + 2, 16); memcpy(w + 5, b + 20, 16); break;
        }
        case T_IQ3_S: {  // d | qs[64] | qh[8] | signs[32] | scales[4]  ->  d, scales, qh[2 words], qs[16 words], signs[8 words]
            uint16_t d; memcpy(&d, b, 2);
            w[0] = d; memcpy(w + 1, b + 106, 4); memcpy(w + 2, b + 66, 8); memcpy(w + 4, b + 2, 64); memcpy(w + 20, b + 74, 32);
            break;
        }
        case T_IQ3_XXS: {  // d | index bytes[64] | scale+sign words[8]
            uint16_t d; memcpy(&d, b, 2);
            w[0] = d; memcpy(w + 1, b + 2, 96);
            break;
        }
        case T_IQ2_S: {  // d | index bytes[32] | sign bytes[32] | qh[8] | scales[8]
            uint16_t d; memcpy(&d, b, 2);
            w[0] = d; memcpy(w + 1, b + 2, 80);
            break;
        }
        case T_IQ2_XS:    // d | 16-bit codes[32] | scales[8]
        case T_IQ2_XXS: { // d | (index word, scale+sign word)[8]
            uint16_t d; memcpy(&d, b, 2);
            w[0] = d; memcpy(w + 1, b + 2, f.type == T_IQ2_XS ? 72 : 64);
            break;
        }
        case T_Q8_0: {  // two blocks of (d, qs[32]); values are stored biased by 128 (unsigned)
            uint16_t d0, d1; memcpy(&d0, b, 2); memcpy(&d1, b + 34, 2);
            w[0] = (uint32_t)d0 | ((uint32_t)d1 << 16);
            memcpy(w + 1, b + 2, 32); memcpy(w + 9, b + 36, 32);
            for (int i = 1; i < 17; i++) w[i] ^= 0x80808080u;
            break;
        }
    }
}
}  // namespace

const PackFormat *pack_format(uint32_t type) {
    for (const PackFormat &f : k_formats) if (f.type == type) return &f;
    return nullptr;
}

const uint32_t *pack_table(const PackFormat &f, uint32_t *count) {
    // The 64-bit tables are read as pairs of words: low four magnitudes, then high four.
    switch (f.type) {
        case T_IQ3_S: *count = 512; return k_iq3s_grid;
        case T_IQ3_XXS: *count = 256; return k_iq3xxs_grid;
        case T_IQ2_S: *count = 2048; return (const uint32_t *)k_iq2s_grid;
        case T_IQ2_XS: *count = 1024; return (const uint32_t *)k_iq2xs_grid;
        case T_IQ2_XXS: *count = 512; return (const uint32_t *)k_iq2xxs_grid;
    }
    *count = 0;
    return nullptr;
}

void pack_rows(const PackFormat &f, const uint8_t *src, uint32_t n_in, uint64_t row0, uint32_t rows, uint32_t rows_pad, uint32_t *out) {
    const TypeInfo *ti = type_info(f.type);
    const uint32_t n_sb = n_in / f.sb_cols, nw = f.sb_words;
    const uint64_t sb_bytes = (uint64_t)f.src_blocks * ti->block_bytes, row_bytes = (uint64_t)n_sb * sb_bytes;
    std::vector<uint32_t> tmp((size_t)rows * nw);
    for (uint32_t s = 0; s < n_sb; s++) {
        for (uint32_t r = 0; r < rows; r++) pack_superblock(f, src + (row0 + r) * row_bytes + (uint64_t)s * sb_bytes, &tmp[(size_t)r * nw]);
        for (uint32_t k = 0; k < nw; k++) {
            uint32_t *plane = out + ((size_t)s * nw + k) * rows_pad;
            for (uint32_t r = 0; r < rows; r++) plane[r] = tmp[(size_t)r * nw + k];
            for (uint32_t r = rows; r < rows_pad; r++) plane[r] = 0;
        }
    }
}

}  // namespace gfx
