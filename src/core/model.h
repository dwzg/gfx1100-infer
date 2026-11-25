// Model description: hyper-parameters and the tensors of each layer, resolved from a GGUF file.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"

namespace gfx {

enum class Arch {
    Llama,   // pre-norm decoder with rotary attention and SwiGLU (llama, qwen2, qwen3 families)
    Qwen35,  // hybrid: gated delta-net (linear attention) layers with periodic gated full attention
};

struct LayerDesc {
    bool recurrent = false;  // true: gated delta-net layer; false: full attention layer
    const TensorInfo *attn_norm = nullptr, *ffn_norm = nullptr;
    // full attention
    const TensorInfo *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const TensorInfo *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    const TensorInfo *q_norm = nullptr, *k_norm = nullptr;
    // gated delta-net
    const TensorInfo *wqkv = nullptr, *wz = nullptr, *conv = nullptr, *dt_bias = nullptr, *ssm_a = nullptr;
    const TensorInfo *wbeta = nullptr, *walpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
    // feed-forward
    const TensorInfo *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
};

struct ModelDesc {
    Arch arch = Arch::Llama;
    std::string arch_name, name;
    int n_vocab = 0, n_embd = 0, n_layer = 0, n_ff = 0, n_ctx_train = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    float norm_eps = 1e-5f;
    // rotary position encoding
    int rope_dim = 0;            // number of leading dimensions of each head that are rotated
    float rope_base = 10000.0f;
    bool rope_half_split = false;  // false: pairs (2i, 2i+1); true: pairs (i, i + rope_dim/2)
    bool q_has_gate = false;       // Q projection also produces a per-head output gate
    // gated delta-net
    int dn_head_dim = 0, dn_k_heads = 0, dn_v_heads = 0, dn_conv = 0;
    int dn_key_dim() const { return dn_head_dim * dn_k_heads; }
    int dn_value_dim() const { return dn_head_dim * dn_v_heads; }
    int dn_conv_dim() const { return 2 * dn_key_dim() + dn_value_dim(); }

    const TensorInfo *tok_embd = nullptr, *output_norm = nullptr, *output = nullptr;
    std::vector<LayerDesc> layers;

    int n_attn_layers() const { int n = 0; for (const auto &l : layers) n += !l.recurrent; return n; }
    // Bytes of key/value cache one token occupies, for a given bytes-per-element.
    uint64_t kv_bytes_per_token(int elem_bytes) const { return (uint64_t)n_attn_layers() * 2 * n_head_kv * head_dim * elem_bytes; }
    // Bytes of context-independent recurrent state (fp32).
    uint64_t recurrent_state_bytes() const;
};

// Throws std::runtime_error naming what is missing if the architecture is not supported.
ModelDesc describe_model(const GGUF &g);

}  // namespace gfx
