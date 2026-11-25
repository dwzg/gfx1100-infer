#include "quant.h"

#include <cmath>
#include <cstring>

#include "gguf.h"

namespace gfx {

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else { int e = -1; do { man <<= 1; e++; } while (!(man & 0x400)); u = sign | ((uint32_t)(112 - e) << 23) | ((man & 0x3FF) << 13); }
    } else if (exp == 31) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &u, 4); return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000; int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127; uint32_t man = u & 0x7FFFFF;
    if (exp == 128) return (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0));
    if (exp > 15) return (uint16_t)(sign | 0x7C00);
    if (exp >= -14) {
        uint32_t h = ((uint32_t)(exp + 15) << 10) | (man >> 13), rem = man & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    if (exp < -25) return (uint16_t)sign;
    man |= 0x800000; int shift = -exp - 1;  // 14..24: subnormal result
    uint32_t h = man >> shift, rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

// UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits. The NVFP4 codebook below stores
// E2M1 values doubled, so the scale is returned halved to compensate.
float ue4m3_to_fp32_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF, man = x & 7;
    float raw = exp == 0 ? std::ldexp((float)man, -9) : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

const int8_t k_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
const int8_t k_fp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const uint32_t k_iq3s_grid[512] = {
#include "iq3s_grid.inc"
};

static inline float h16(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return fp16_to_fp32(h); }

// 6-bit (scale, min) pair j of a K-quant super-block's 12-byte scale field.
static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

static void dq_q4_0(const uint8_t *b, float *y) {
    float d = h16(b); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) { y[j] = d * ((qs[j] & 15) - 8); y[j + 16] = d * ((qs[j] >> 4) - 8); }
}
static void dq_q4_1(const uint8_t *b, float *y) {
    float d = h16(b), m = h16(b + 2); const uint8_t *qs = b + 4;
    for (int j = 0; j < 16; j++) { y[j] = d * (qs[j] & 15) + m; y[j + 16] = d * (qs[j] >> 4) + m; }
}
static void dq_q5(const uint8_t *b, float *y, bool has_min) {
    float d = h16(b), m = has_min ? h16(b + 2) : 0.0f; b += has_min ? 4 : 2;
    uint32_t qh; memcpy(&qh, b, 4); const uint8_t *qs = b + 4; int off = has_min ? 0 : 16;
    for (int j = 0; j < 16; j++) {
        int x0 = (qs[j] & 15) | (((qh >> j) << 4) & 0x10), x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
        y[j] = d * (x0 - off) + m; y[j + 16] = d * (x1 - off) + m;
    }
}
static void dq_q8_0(const uint8_t *b, float *y) {
    float d = h16(b); const int8_t *qs = (const int8_t *)b + 2;
    for (int j = 0; j < 32; j++) y[j] = d * qs[j];
}
static void dq_q2_k(const uint8_t *b, float *y) {
    const uint8_t *scales = b, *q = b + 16; float d = h16(b + 80), mn = h16(b + 82); int is = 0;
    for (int n = 0; n < 256; n += 128, q += 32)
        for (int j = 0, shift = 0; j < 4; j++, shift += 2)
            for (int h = 0; h < 2; h++) {
                uint8_t sc = scales[is++]; float dl = d * (sc & 15), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16 * h] >> shift) & 3) - ml;
            }
}
static void dq_q3_k(const uint8_t *b, float *y) {
    const uint8_t *hm = b, *q = b + 32; float d_all = h16(b + 108);
    uint32_t aux[4]; memcpy(aux, b + 96, 12); uint32_t tmp = aux[2];
    const uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f;
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4); aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);        aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    int8_t scales[16]; memcpy(scales, aux, 16);
    int is = 0; uint8_t m = 1;
    for (int n = 0; n < 256; n += 128, q += 32)
        for (int j = 0, shift = 0; j < 4; j++, shift += 2, m = (uint8_t)(m << 1))
            for (int h = 0; h < 2; h++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int)((q[l + 16 * h] >> shift) & 3) - ((hm[l + 16 * h] & m) ? 0 : 4));
            }
}
static void dq_q4_k(const uint8_t *b, float *y) {
    float d = h16(b), mn = h16(b + 2); const uint8_t *scales = b + 4, *q = b + 16; int is = 0;
    for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
        uint8_t sc, m; scale_min_k4(is, scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        scale_min_k4(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 15) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
    }
}
static void dq_q5_k(const uint8_t *b, float *y) {
    float d = h16(b), mn = h16(b + 2); const uint8_t *scales = b + 4, *qh = b + 16, *ql = b + 48; int is = 0; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 = (uint8_t)(u1 << 2), u2 = (uint8_t)(u2 << 2)) {
        uint8_t sc, m; scale_min_k4(is, scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        scale_min_k4(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 15) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    }
}
static void dq_q6_k(const uint8_t *b, float *y) {
    const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)b + 192; float d = h16(b + 208);
    for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = (int)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32,  q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2; y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
        }
}
static void dq_iq4_nl(const uint8_t *b, float *y) {
    float d = h16(b); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) { y[j] = d * k_iq4nl_values[qs[j] & 15]; y[j + 16] = d * k_iq4nl_values[qs[j] >> 4]; }
}
static void dq_iq4_xs(const uint8_t *b, float *y) {
    float d = h16(b); uint16_t scales_h; memcpy(&scales_h, b + 2, 2); const uint8_t *scales_l = b + 4, *qs = b + 8;
    for (int ib = 0; ib < 8; ib++, y += 32, qs += 16) {
        int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 15) | (((scales_h >> (2 * ib)) & 3) << 4);
        float dl = d * (ls - 32);
        for (int j = 0; j < 16; j++) { y[j] = dl * k_iq4nl_values[qs[j] & 15]; y[j + 16] = dl * k_iq4nl_values[qs[j] >> 4]; }
    }
}
static void dq_iq3_s(const uint8_t *b, float *y) {
    float d = h16(b); const uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
    for (int ib32 = 0; ib32 < 8; ib32++, qs += 8, signs += 4) {
        int sc = ib32 & 1 ? scales[ib32 / 2] >> 4 : scales[ib32 / 2] & 15; float db = d * (1 + 2 * sc); uint8_t h = qh[ib32];
        for (int l = 0; l < 4; l++, y += 8) {
            uint32_t g1 = k_iq3s_grid[qs[2 * l] | ((h << (8 - 2 * l)) & 256)], g2 = k_iq3s_grid[qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
            for (int j = 0; j < 4; j++) {
                y[j] = db * (float)((g1 >> (8 * j)) & 255) * ((signs[l] & (1 << j)) ? -1.f : 1.f);
                y[j + 4] = db * (float)((g2 >> (8 * j)) & 255) * ((signs[l] & (1 << (j + 4))) ? -1.f : 1.f);
            }
        }
    }
}
static void dq_mxfp4(const uint8_t *b, float *y) {
    // E8M0 scale, halved (codebook is doubled): 2^(e-127) * 0.5
    float d = std::ldexp(1.0f, (int)b[0] - 128); const uint8_t *qs = b + 1;
    for (int j = 0; j < 16; j++) { y[j] = d * k_fp4_values[qs[j] & 15]; y[j + 16] = d * k_fp4_values[qs[j] >> 4]; }
}
static void dq_nvfp4(const uint8_t *b, float *y) {
    const uint8_t *qs = b + 4;
    for (int s = 0; s < 4; s++, y += 16) {
        float d = ue4m3_to_fp32_half(b[s]);
        for (int j = 0; j < 8; j++) { y[j] = d * k_fp4_values[qs[s * 8 + j] & 15]; y[j + 8] = d * k_fp4_values[qs[s * 8 + j] >> 4]; }
    }
}

bool dequantize(uint32_t type, const uint8_t *src, float *dst, uint64_t n) {
    const TypeInfo *ti = type_info(type);
    if (!ti || !ti->decodable) return false;
    if (type == T_F32) { memcpy(dst, src, n * 4); return true; }
    if (type == T_F16 || type == T_BF16) {
        for (uint64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, src + 2 * i, 2); dst[i] = type == T_F16 ? fp16_to_fp32(h) : bf16_to_fp32(h); }
        return true;
    }
    uint64_t nb = n / ti->block_elems;
    for (uint64_t i = 0; i < nb; i++, src += ti->block_bytes, dst += ti->block_elems) {
        switch (type) {
            case T_Q4_0: dq_q4_0(src, dst); break;
            case T_Q4_1: dq_q4_1(src, dst); break;
            case T_Q5_0: dq_q5(src, dst, false); break;
            case T_Q5_1: dq_q5(src, dst, true); break;
            case T_Q8_0: dq_q8_0(src, dst); break;
            case T_Q2_K: dq_q2_k(src, dst); break;
            case T_Q3_K: dq_q3_k(src, dst); break;
            case T_Q4_K: dq_q4_k(src, dst); break;
            case T_Q5_K: dq_q5_k(src, dst); break;
            case T_Q6_K: dq_q6_k(src, dst); break;
            case T_IQ4_NL: dq_iq4_nl(src, dst); break;
            case T_IQ4_XS: dq_iq4_xs(src, dst); break;
            case T_IQ3_S: dq_iq3_s(src, dst); break;
            case T_MXFP4: dq_mxfp4(src, dst); break;
            case T_NVFP4: dq_nvfp4(src, dst); break;
            default: return false;
        }
    }
    return true;
}

}  // namespace gfx
