// Decoders for GGUF weight encodings. These are the scalar reference decoders: the GPU kernels
// and the weight transcoder are validated against them.
#pragma once
#include <cstdint>

namespace gfx {

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);   // round-to-nearest-even
inline float bf16_to_fp32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
float ue4m3_to_fp32_half(uint8_t x);  // NVFP4 sub-block scale (already halved, see quant.cpp)

extern const int8_t k_iq4nl_values[16];  // IQ4_NL / IQ4_XS codebook
extern const int8_t k_fp4_values[16];    // MXFP4 / NVFP4 codebook (E2M1 values, doubled)

// Decode `n` elements (a multiple of the type's block size) starting at a block boundary.
// Returns false if the type has no decoder.
bool dequantize(uint32_t type, const uint8_t *src, float *dst, uint64_t n);

}  // namespace gfx
