// GPU inference engine: uploads the model to the device and runs the forward pass as a
// pre-recorded sequence of compute dispatches, one submission per token.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "core/model.h"
#include "vk/vk.h"

namespace gfx {

class GpuModel {
  public:
    // force_fp16: store every matrix as half-precision rows (the simple baseline path, for testing).
    // kv8: keep the key/value cache as 8-bit integers with a scale per 32 values (about half the
    //      memory and memory traffic of the half-precision cache, at a small accuracy cost).
    GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16 = false, bool kv8 = false);
    ~GpuModel();

    void reset();
    int n_past() const { return n_past_; }
    int n_ctx() const { return n_ctx_; }
    // Process tokens at the next positions. Returns the logits for the last one.
    const std::vector<float> &eval(const int32_t *tokens, int n);

    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t state_bytes() const { return state_bytes_; }
    size_t dispatches_per_token() const;
    size_t barriers_per_token() const;

  private:
    struct Matrix {
        uint32_t n_in = 0, n_out = 0;
        const struct PackFormat *fmt = nullptr;  // native quantised layout, or nullptr for fp16 rows
        std::vector<Buffer> chunks;       // consecutive row ranges
        std::vector<uint32_t> chunk_rows; // rows in each chunk (its plane stride is rows rounded up to 64)
        Buffer bias;                      // fp32, optional
    };
    // An activation vector that feeds matrix products, with its 16-column block sums.
    struct Vec { Buffer v, s; uint32_t n = 0; };
    struct Layer {
        bool recurrent = false;
        Buffer attn_norm, ffn_norm, qk_norm;
        Matrix wq, wk, wv, wo, wqkv, wz, wbeta, walpha, ssm_out, ffn_gate, ffn_up, ffn_down;
        Buffer kc, vc;                    // key / value cache
        Buffer ks, vs;                    // per-block scales (8-bit cache only)
        Buffer conv_w, dt, a, ssm_norm, state, hist;
        bool has_qk_norm = false;
    };
    Matrix load_matrix(const TensorInfo *w, const TensorInfo *bias = nullptr);
    Buffer load_vector(const TensorInfo *t);
    Buffer scratch(uint64_t n_floats);
    Vec scratch_vec(uint32_t n);
    void blocksum(Program &p, const Vec &x);
    void matvec(Program &p, const Matrix &m, const Vec &x, const Buffer &y);
    void build_program();
    void eval_one(int32_t token);

    Device &dev_;
    const GGUF &g_;
    ModelDesc d_;
    int n_ctx_, n_past_ = 0;
    bool force_fp16_ = false, kv8_ = false;
    std::vector<Layer> layers_;
    Buffer output_norm_;
    Matrix output_;
    Buffer ctl_, embd_, logits_host_, dummy_;
    std::vector<Buffer> zero_on_reset_;
    std::unique_ptr<Program> program_;
    std::vector<float> logits_;
    uint64_t weight_bytes_ = 0, state_bytes_ = 0;
};

}  // namespace gfx
