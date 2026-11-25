// GPU inference engine: uploads the model to the device and runs the forward pass as a
// pre-recorded sequence of compute dispatches, one submission per token.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/model.h"
#include "vk/vk.h"

namespace gfx {

class GpuModel {
  public:
    // force_fp16: store every matrix as half-precision rows (the simple baseline path, for testing).
    // kv8: keep the key/value cache as 8-bit integers with a scale per 32 values (about half the
    //      memory and memory traffic of the half-precision cache, at a small accuracy cost).
    // n_ctx <= 0: use as much context as fits in the GPU memory that is free after loading the
    //      weights (capped at the model's trained context), keeping `reserve_bytes` unallocated.
    // batch: how many prompt tokens are processed together (1 disables batching).
    GpuModel(Device &dev, const GGUF &g, const ModelDesc &desc, int n_ctx, bool force_fp16 = false, bool kv8 = false, uint64_t reserve_bytes = 256ull << 20, int batch = 32, bool allow_wmma = true);
    int batch() const { return batch_; }
    // Which weight formats passed the on-device check of the matrix-instruction kernels (empty if
    // the device has none, or they are disabled).
    const std::string &wmma_report() const { return wmma_report_; }
    // GPU bytes one token of context costs, for a given cache type.
    static uint64_t bytes_per_token(const ModelDesc &d, bool kv8);
    ~GpuModel();

    void reset();
    // Test aid: continue from position n as if n all-zero cache entries preceded it.
    void skip_to(int n);
    int n_past() const { return n_past_; }
    int n_ctx() const { return n_ctx_; }
    // Process tokens at the next positions. Returns the logits for the last one.
    const std::vector<float> &eval(const int32_t *tokens, int n);

    // Save / restore everything that depends on the processed tokens (attention caches up to the
    // current position, linear-attention state, last logits), so a long prompt is only paid for once.
    // The file is tied to the model and cache type it was written with.
    void save_state(const std::string &path, const std::vector<int32_t> &tokens);
    std::vector<int32_t> load_state(const std::string &path);   // returns the tokens it was saved with
    const std::vector<float> &logits() const { return logits_; }

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
        std::vector<uint32_t> chunk_stride;  // optional explicit plane stride (when only some rows are computed)
        uint32_t stride(size_t c) const { return c < chunk_stride.size() ? chunk_stride[c] : (chunk_rows[c] + 63) / 64 * 64; }
        Buffer bias;                      // fp32, optional
    };
    // An activation vector that feeds matrix products, with its 16-column block sums.
    struct Vec { Buffer v, s, h; uint32_t n = 0; };   // h: half-precision copy (matrix-instruction path only)
    struct Layer {
        bool recurrent = false;
        Buffer attn_norm, ffn_norm, qk_norm;
        Matrix wq, wk, wv, wo, wqkv, wz, wbeta, walpha, ssm_out, ffn_gate, ffn_up, ffn_down;
        Buffer kc, vc;                    // key cache (plane-major) / value cache (row-major)
        Buffer conv_w, dt, a, ssm_norm, state, hist;
        bool has_qk_norm = false;
    };
    Matrix load_matrix(const TensorInfo *w, const TensorInfo *bias = nullptr);
    Buffer load_vector(const TensorInfo *t);
    Buffer scratch(uint64_t n_floats);
    Vec scratch_vec(uint64_t n);
    // y[t] = M x[t] for the first n_tokens vectors of x; y holds vectors of m.n_out elements.
    void matmul(Program &p, const Matrix &m, const Vec &x, const Buffer &y, uint32_t n_tokens);
    bool use_wmma(const Matrix &m, uint32_t n_tokens) const;
    void dispatch_wmma(Program &p, const Matrix &m, size_t chunk, uint32_t row0, const Buffer &xh, const Buffer &y, uint32_t n_tokens);
    void to_half(Program &p, const Vec &x, uint64_t n);
    void verify_wmma();
    void alloc_activations();
    Program &program(int n_tokens);
    void eval_batch(const int32_t *tokens, int n);

    Device &dev_;
    const GGUF &g_;
    ModelDesc d_;
    int n_ctx_, n_past_ = 0;
    bool force_fp16_ = false, kv8_ = false;
    std::vector<Layer> layers_;
    Buffer output_norm_;
    Matrix output_;
    Buffer ctl_, ind_, embd_, logits_host_, dummy_;
    std::vector<std::pair<uint32_t, Buffer>> tables_;   // codebooks, by tensor type
    uint32_t stride_ = 0, kv_words_ = 0;   // cache plane stride (positions, padded) and words per head per position
    std::vector<Buffer> zero_on_reset_;
    // Activations for up to batch_cap_ tokens (one vector per token, consecutive).
    struct Act { Buffer x, a, g, u, qfull, kraw, vraw, q, gate, scores, zsum, part, mixed, z, alpha, beta, qkv, dn_o, logits; Vec h, act, mix, dn_on, h_last; } act_;
    int batch_ = 1, batch_cap_ = 8, q_batch_ = 1;   // tokens per batch; allocated capacity; queries scored together
    uint32_t part_stride_ = 0;
    bool wmma_ = false;                    // device has the matrix instructions and they are allowed
    bool wmma_emu_ = false;                // testing: run that path with the matrix calls emulated
    std::vector<uint32_t> wmma_ok_;        // tensor types whose kernel passed the on-device check
    std::string wmma_report_;
    std::map<int, std::unique_ptr<Program>> programs_;   // by number of tokens
    std::vector<float> logits_;
    uint64_t weight_bytes_ = 0, state_bytes_ = 0;
};

}  // namespace gfx
