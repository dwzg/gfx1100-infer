#include "model.h"

#include <stdexcept>

namespace gfx {

uint64_t ModelDesc::recurrent_state_bytes() const {
    uint64_t n = 0;
    for (const auto &l : layers)
        if (l.recurrent) n += ((uint64_t)dn_head_dim * dn_head_dim * dn_v_heads + (uint64_t)(dn_conv - 1) * dn_conv_dim()) * 4;
    return n;
}

namespace {
struct Finder {
    const GGUF &g;
    const TensorInfo *get(const std::string &name, bool required = true) const {
        const TensorInfo *t = g.tensor(name);
        if (!t && required) throw std::runtime_error("model is missing tensor " + name);
        return t;
    }
    static void expect(const TensorInfo *t, uint64_t d0, uint64_t d1) {
        if (t && (t->dim(0) != d0 || t->dim(1) != d1))
            throw std::runtime_error("tensor " + t->name + " has shape [" + std::to_string(t->dim(0)) + ", " + std::to_string(t->dim(1)) +
                                     "], expected [" + std::to_string(d0) + ", " + std::to_string(d1) + "]");
    }
};
}  // namespace

ModelDesc describe_model(const GGUF &g) {
    ModelDesc m;
    m.arch_name = g.architecture;
    m.name = g.get_str("general.name");
    const std::string &a = g.architecture;
    if (a == "llama" || a == "qwen2" || a == "qwen3") m.arch = Arch::Llama;
    else if (a == "qwen35") m.arch = Arch::Qwen35;
    else throw std::runtime_error("architecture '" + a + "' is not supported yet (supported: qwen35, llama, qwen2, qwen3)");

    auto key = [&](const char *s) { return g.arch_key(s); };
    m.n_embd = (int)g.get_int(key("embedding_length"), 0);
    m.n_ff = (int)g.get_int(key("feed_forward_length"), 0);
    m.n_head = (int)g.get_int(key("attention.head_count"), 0);
    m.n_head_kv = (int)g.get_int(key("attention.head_count_kv"), m.n_head);
    m.n_ctx_train = (int)g.get_int(key("context_length"), 0);
    m.norm_eps = (float)g.get_float(key("attention.layer_norm_rms_epsilon"), 1e-5);
    m.rope_base = (float)g.get_float(key("rope.freq_base"), 10000.0);
    int n_block = (int)g.get_int(key("block_count"), 0);
    if (m.n_embd <= 0 || m.n_head <= 0 || n_block <= 0) throw std::runtime_error("model metadata is incomplete");
    m.head_dim = (int)g.get_int(key("attention.key_length"), m.n_embd / m.n_head);
    if (g.get_int(key("attention.value_length"), m.head_dim) != m.head_dim) throw std::runtime_error("key and value head sizes differ; not supported");
    m.rope_dim = (int)g.get_int(key("rope.dimension_count"), m.head_dim);
    if (g.get_float(key("rope.scaling.factor"), 1.0) != 1.0 && g.get_str(key("rope.scaling.type")) == "yarn")
        throw std::runtime_error("YaRN-scaled position encoding is not supported yet");

    Finder f{g};
    m.tok_embd = f.get("token_embd.weight");
    m.n_vocab = (int)m.tok_embd->dim(1);
    Finder::expect(m.tok_embd, m.n_embd, m.n_vocab);
    m.output_norm = f.get("output_norm.weight");
    m.output = f.get("output.weight", false);
    if (!m.output) m.output = m.tok_embd;  // tied input/output embeddings

    if (m.arch == Arch::Qwen35) {
        // Extra trailing blocks are multi-token-prediction draft heads; they are not part of the main pass.
        m.n_layer = n_block - (int)g.get_int(key("nextn_predict_layers"), 0);
        m.rope_half_split = true; m.q_has_gate = true;
        m.dn_head_dim = (int)g.get_int(key("ssm.state_size"), 0);
        m.dn_k_heads = (int)g.get_int(key("ssm.group_count"), 0);
        m.dn_v_heads = (int)g.get_int(key("ssm.time_step_rank"), 0);
        m.dn_conv = (int)g.get_int(key("ssm.conv_kernel"), 0);
        if (m.dn_head_dim <= 0 || m.dn_k_heads <= 0 || m.dn_v_heads <= 0 || m.dn_conv <= 1 || m.dn_v_heads % m.dn_k_heads)
            throw std::runtime_error("model metadata for the linear-attention layers is incomplete");
        if (g.get_int(key("ssm.inner_size"), m.dn_value_dim()) != m.dn_value_dim()) throw std::runtime_error("unexpected linear-attention inner size");
    } else {
        m.n_layer = n_block;
    }
    int interval = (int)g.get_int(key("full_attention_interval"), 4);

    m.layers.resize(m.n_layer);
    for (int il = 0; il < m.n_layer; il++) {
        LayerDesc &l = m.layers[il];
        std::string p = "blk." + std::to_string(il) + ".";
        l.attn_norm = f.get(p + "attn_norm.weight");
        l.ffn_gate = f.get(p + "ffn_gate.weight"); l.ffn_up = f.get(p + "ffn_up.weight"); l.ffn_down = f.get(p + "ffn_down.weight");
        if (m.n_ff <= 0) m.n_ff = (int)l.ffn_gate->dim(1);
        Finder::expect(l.ffn_gate, m.n_embd, m.n_ff); Finder::expect(l.ffn_up, m.n_embd, m.n_ff); Finder::expect(l.ffn_down, m.n_ff, m.n_embd);
        if (m.arch == Arch::Qwen35) {
            l.ffn_norm = f.get(p + "post_attention_norm.weight");
            l.recurrent = (il + 1) % interval != 0;
            if (const MetaValue *rl = g.find(key("attention.recurrent_layers")); rl && rl->type == M_ARR && (uint64_t)il < rl->count) l.recurrent = rl->arr_int(il) != 0;
        } else {
            l.ffn_norm = f.get(p + "ffn_norm.weight");
        }
        if (l.recurrent) {
            if (g.tensor(p + "attn_qkv_gate.weight") || g.tensor(p + "ssm_beta_alpha.weight"))
                throw std::runtime_error("fused linear-attention projections are not supported yet");
            l.wqkv = f.get(p + "attn_qkv.weight"); l.wz = f.get(p + "attn_gate.weight");
            l.conv = f.get(p + "ssm_conv1d.weight"); l.dt_bias = f.get(p + "ssm_dt.bias"); l.ssm_a = f.get(p + "ssm_a");
            l.wbeta = f.get(p + "ssm_beta.weight"); l.walpha = f.get(p + "ssm_alpha.weight");
            l.ssm_norm = f.get(p + "ssm_norm.weight"); l.ssm_out = f.get(p + "ssm_out.weight");
            Finder::expect(l.wqkv, m.n_embd, m.dn_conv_dim()); Finder::expect(l.wz, m.n_embd, m.dn_value_dim());
            Finder::expect(l.conv, m.dn_conv, m.dn_conv_dim()); Finder::expect(l.ssm_out, m.dn_value_dim(), m.n_embd);
            Finder::expect(l.wbeta, m.n_embd, m.dn_v_heads); Finder::expect(l.walpha, m.n_embd, m.dn_v_heads);
        } else {
            int q_out = m.head_dim * m.n_head * (m.q_has_gate ? 2 : 1), kv_out = m.head_dim * m.n_head_kv;
            l.wq = f.get(p + "attn_q.weight"); l.wk = f.get(p + "attn_k.weight"); l.wv = f.get(p + "attn_v.weight"); l.wo = f.get(p + "attn_output.weight");
            Finder::expect(l.wq, m.n_embd, q_out); Finder::expect(l.wk, m.n_embd, kv_out); Finder::expect(l.wv, m.n_embd, kv_out);
            Finder::expect(l.wo, m.head_dim * m.n_head, m.n_embd);
            l.bq = f.get(p + "attn_q.bias", false); l.bk = f.get(p + "attn_k.bias", false);
            l.bv = f.get(p + "attn_v.bias", false); l.bo = f.get(p + "attn_output.bias", false);
            l.q_norm = f.get(p + "attn_q_norm.weight", m.arch == Arch::Qwen35); l.k_norm = f.get(p + "attn_k_norm.weight", m.arch == Arch::Qwen35);
        }
    }
    if (m.rope_dim > m.head_dim || m.rope_dim % 2) throw std::runtime_error("unexpected rotary dimension count");
    return m;
}

}  // namespace gfx
