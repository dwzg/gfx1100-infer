// Shared preamble for all kernels.
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require

float silu(float x) { return x / (1.0 + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float softplus(float x) { return x > 20.0 ? x : log(1.0 + exp(x)); }
