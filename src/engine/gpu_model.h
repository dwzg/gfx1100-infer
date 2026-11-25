// GPU inference engine: uploads the model to the device and runs the forward pass as a
// pre-recorded sequence of compute dispatches, one submission per token.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/inkling.h"
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
    // Inkling 2 models. `want_expert`, a testing aid, limits which experts are uploaded (the rest
    // of each layer's expert buffer is left unwritten), for machines that cannot hold them all.
    using ExpertFilter = std::function<bool(int layer, int expert)>;
    GpuModel(Device &dev, const GGUF &g, const InklingDesc &desc, int n_ctx, bool kv8 = false, uint64_t reserve_bytes = 256ull << 20, int batch = 32, bool allow_wmma = true,
             const ExpertFilter &want_expert = {});
    static uint64_t bytes_per_token(const InklingDesc &d, bool kv8);
    // The experts the last token used, per cache slot (Inkling; for tests).
    std::vector<std::vector<int>> last_experts();
    int batch() const { return batch_; }
    // Which weight formats passed the on-device check of the matrix-instruction kernels (empty if
    // the device has none, or they are disabled).
    const std::string &wmma_report() const { return wmma_report_; }
    // GPU bytes one token of context costs, for a given cache type.
    static uint64_t bytes_per_token(const ModelDesc &d, bool kv8);
    ~GpuModel();

    // ---- Token streams ----
    // The attention caches are divided into pages that token streams ("sequences") take as they
    // grow, so several conversations share the context memory, and one can branch off another
    // without copying what they have in common. eval() and the calls below act on the selected
    // sequence; sequence 0 exists from the start.
    int seq_create();                        // a new, empty sequence
    void seq_destroy(int id);                // give its pages back (not the selected one)
    // A new sequence that starts as the first n positions of `src`, sharing their cache pages
    // (a shared page is copied when one of the two writes to it).
    int seq_fork(int src, int n);
    // Select the sequence eval() continues. The linear-attention state on the GPU is not part
    // of a sequence: the caller pairs sequences with state_save()/state_load().
    void seq_select(int id);
    int seq_selected() const { return active_; }
    int seq_len(int id) const { return seqs_[id].len; }
    void seq_truncate(int n);                // selected sequence: forget positions >= n
    // Cache pages that extending sequence `id` to `upto` positions would take from the free ones.
    int pages_needed(int id, int upto) const;
    int pages_owned(int id) const;           // pages only sequence `id` uses (what destroying it would free)
    int pages_free() const { return (int)free_pages_.size(); }
    int pages_total() const { return (int)page_ref_.size(); }
    int page_size() const { return (int)page_; }

    // Everything outside the attention caches that depends on the tokens processed so far: the
    // linear-attention state and the logits of the last token. Saved at position n of a sequence
    // and loaded back later, it lets that sequence continue from n (models without
    // linear-attention layers can continue from any position; for them this is just the logits).
    bool has_recurrent_state() const { return recurrent_bytes_ > 0; }
    size_t state_size() const { return recurrent_bytes_ + 4 * logits_.size(); }
    void state_save(std::vector<uint8_t> &out);
    void state_load(const std::vector<uint8_t> &in);
    void state_clear();                      // the state before the first token

    void reset();                            // selected sequence back to empty
    // Test aid: continue from position n as if n all-zero cache entries preceded it.
    void skip_to(int n);
    int n_past() const { return seqs_[active_].len; }
    int n_ctx() const { return n_ctx_; }     // longest sequence; also the positions all sequences share
    // Process tokens at the next positions. Returns the logits for the last one.
    const std::vector<float> &eval(const int32_t *tokens, int n);

    // Save / restore the selected sequence with its state in a file, so a long prompt is only
    // paid for once. The file is tied to the model and cache type it was written with.
    void save_state(const std::string &path, const std::vector<int32_t> &tokens);
    std::vector<int32_t> load_state(const std::string &path);   // returns the tokens it was saved with
    // The tokens a saved context holds, without loading it; empty if the file is not a saved
    // context of this model and cache type.
    std::vector<int32_t> saved_tokens(const std::string &path) const;
    const std::vector<float> &logits() const { return logits_; }

    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t state_bytes() const { return state_bytes_; }
    size_t dispatches_per_token() const;
    size_t barriers_per_token() const;

  private:
    uint32_t tile_ = 4;                                         // tokens per workgroup in the batched matrix kernels (T in the shaders): 4, or 2 with GFX_TILE=2
    static constexpr uint32_t k_wmma_tile = 16;                 // tokens per workgroup in the matrix-instruction kernels
    static constexpr uint32_t k_wmma_min_tokens = 12;           // below this the portable batched kernels are used
    static constexpr uint64_t k_scores_budget = 64ull << 20;    // bytes of score scratch (bounds how many queries are scored together)
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
        Matrix ffn_gu;                    // gate and up with their rows interleaved (then ffn_gate and ffn_up are empty)
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
    void init_batching(int batch, bool allow_wmma);
    void init_context(uint64_t reserve_bytes, uint64_t fixed, int n_caches);
    void finish_init();
    Program &program(int n_tokens);
    struct Ink;                                    // everything specific to Inkling 2 models (gpu_inkling.cpp)
    std::shared_ptr<Ink> ink_;
    void record_inkling(Program &p, uint32_t N);
    void eval_batch(const int32_t *tokens, int n);
    struct Seq { bool live = false; int len = 0; std::vector<uint32_t> pages; };
    void take_pages(uint32_t pos0, uint32_t n);
    void release_pages(Seq &s, size_t keep);
    void copy_page(uint32_t from, uint32_t to);
    void write_page_table();
    using Regions = std::vector<std::pair<uint64_t, uint64_t>>;   // (offset, bytes) inside a buffer
    void read_regions(const Buffer &b, const Regions &r, std::vector<uint8_t> &out);
    void write_regions(const Buffer &b, const Regions &r, const uint8_t *data);
    Regions key_regions(const Seq &s, int n) const;
    Regions value_regions(const Seq &s, int n) const;

    Device &dev_;
    const GGUF &g_;
    ModelDesc d_;
    int n_ctx_;
    bool force_fp16_ = false, kv8_ = false;
    std::vector<Layer> layers_;
    Buffer output_norm_;
    Matrix output_;
    Buffer ctl_, ind_, embd_, logits_host_, dummy_;
    std::vector<Seq> seqs_;
    int active_ = 0;
    uint32_t page_ = 1024, page_shift_ = 10;       // cache positions per page
    std::vector<uint32_t> page_ref_, free_pages_;  // sequences using each page; unused pages
    Buffer pt_;                                    // page table of the selected sequence, read by the attention kernels
    Buffer snap_, xfer_;                           // host copies: linear-attention state; transfers
    std::unique_ptr<Program> snap_save_, snap_load_;
    uint64_t recurrent_bytes_ = 0;
    float ink_tau_alpha_ = 0, ink_tau_floor_ = 1, ink_softcap_ = 0;
    std::vector<std::pair<uint32_t, Buffer>> tables_;   // codebooks, by tensor type
    uint32_t stride_ = 0, kv_words_ = 0;   // cache plane stride (positions, padded) and words per head per position
    std::vector<std::pair<Buffer, Buffer>> caches_;   // paged key / value caches, one pair per layer that attends to everything
    std::vector<Buffer> state_bufs_;                  // everything else that depends on the tokens processed (cleared by reset)
    // Activations for up to batch_cap_ tokens (one vector per token, consecutive).
    struct Act { Buffer x, a, g, u, gu, qfull, kraw, vraw, q, gate, scores, zsum, part, mixed, z, alpha, beta, qkv, dn_o, logits; Vec h, act, mix, dn_on, h_last; } act_;
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
