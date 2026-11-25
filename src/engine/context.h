// Keeps processed conversations on the GPU and brings the engine to a new prompt by the cheapest
// route: continuing a conversation, branching off one at the last point they have in common, or
// starting over. This is what makes a client that resends its whole conversation with every
// request (as chat and agent front ends do) pay only for what is new.
//
// Attention layers can resume at any position: their cache holds every position separately.
// Linear-attention layers cannot: their state is a running summary, so a conversation can only be
// resumed where a copy of that state was kept (a checkpoint). Checkpoints are taken at positions
// the caller marks, and wherever enough has been processed since the last one.
#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfx {

class GpuModel;

struct ContextFull : std::runtime_error {
    int tokens, limit;
    ContextFull(int t, int l) : std::runtime_error("prompt is too long: " + std::to_string(t) + " tokens > " + std::to_string(l) + " maximum"), tokens(t), limit(l) {}
};

class ContextCache {
  public:
    struct Options {
        int max_conversations = 8;   // kept at once
        int stale_seconds = 3600;    // a conversation unused for this long is the first to go
        int max_checkpoints = 3;     // per conversation, not counting the marked positions
        int min_gap = 256;           // tokens processed since the last checkpoint that justify another
        int leave_gap = 16;          // the same, when the GPU is about to be handed to another conversation
        // Keep the processed context at marked positions in files in this directory, and start from
        // such a file when a prompt begins the same way (after a restart, or in a new conversation).
        std::string store_dir;
        int store_min_tokens = 512;  // only what took at least this many tokens to process is stored
        int store_max_files = 8;     // the oldest files go first
    };
    explicit ContextCache(GpuModel &m);
    ContextCache(GpuModel &m, Options o);
    ~ContextCache();

    // Called between batches with the number of prompt tokens done; return false to stop early.
    using Progress = std::function<bool(int done, int total)>;
    struct Result {
        int reused = 0;       // leading tokens that were not processed again
        int evaluated = 0;
        bool cancelled = false;
        int from_disk = 0;    // of the reused tokens, how many came from a stored file
    };
    // Process `prompt` (at least one token), leaving its last token's logits in logits().
    // `marks` are positions (token counts) worth a checkpoint, e.g. the end of a system prompt
    // that later conversations will share. Throws ContextFull if the prompt cannot fit.
    Result prepare(const std::vector<int32_t> &prompt, const std::vector<int> &marks = {}, const Progress &progress = {});
    const std::vector<float> &logits() const;
    // Append one token to the conversation prepare() selected. False when the context is used up.
    bool push(int32_t token);
    int length() const;                  // tokens in the current conversation
    int limit() const;                   // longest conversation the engine can hold
    void clear();                        // forget everything (after an error)

    // For logs and tests.
    int conversations() const { return (int)convs_.size(); }
    int checkpoints() const;
    std::string summary() const;

  private:
    struct Checkpoint { int pos; bool pinned; std::shared_ptr<const std::vector<uint8_t>> state; };
    struct Conv { int seq = 0; std::vector<int32_t> tokens; std::vector<Checkpoint> cps; uint64_t used = 0; std::chrono::steady_clock::time_point last; };
    void checkpoint(Conv &c, bool pinned);
    void leave_current();
    void destroy(Conv *c);
    bool evict_one(const Conv *keep);
    void make_room(int upto);
    Conv *add_conv(int seq);
    std::string store_path(const std::vector<int32_t> &prompt, int n) const;
    void store_prune() const;

    GpuModel &m_;
    Options o_;
    bool recurrent_;
    std::vector<std::unique_ptr<Conv>> convs_;
    Conv *cur_ = nullptr;      // the conversation whose state is on the GPU, at its last token
    uint64_t clock_ = 0;
};

}  // namespace gfx
