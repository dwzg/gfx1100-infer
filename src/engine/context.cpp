#include "context.h"

#include <algorithm>

#include "gpu_model.h"

namespace gfx {

ContextCache::ContextCache(GpuModel &m) : ContextCache(m, Options{}) {}
ContextCache::ContextCache(GpuModel &m, Options o) : m_(m), o_(o), recurrent_(m.has_recurrent_state()) {
    m_.reset();
    cur_ = add_conv(m_.seq_selected());
}
ContextCache::~ContextCache() = default;

ContextCache::Conv *ContextCache::add_conv(int seq) {
    convs_.push_back(std::make_unique<Conv>());
    convs_.back()->seq = seq; convs_.back()->used = ++clock_; convs_.back()->last = std::chrono::steady_clock::now();
    return convs_.back().get();
}

const std::vector<float> &ContextCache::logits() const { return m_.logits(); }
int ContextCache::length() const { return (int)cur_->tokens.size(); }
int ContextCache::limit() const { return m_.n_ctx(); }

int ContextCache::checkpoints() const {
    int n = 0;
    for (auto &c : convs_) n += (int)c->cps.size();
    return n;
}

std::string ContextCache::summary() const {
    return std::to_string(convs_.size()) + " conversation(s), " + std::to_string(m_.pages_total() - m_.pages_free()) + "/" + std::to_string(m_.pages_total()) + " pages";
}

void ContextCache::clear() {
    m_.reset();
    for (auto &c : convs_) if (c.get() != cur_) m_.seq_destroy(c->seq);
    Conv *keep = cur_;
    convs_.erase(std::remove_if(convs_.begin(), convs_.end(), [&](auto &c) { return c.get() != keep; }), convs_.end());
    cur_->tokens.clear(); cur_->cps.clear();
}

// Keep the state at the current conversation's last token.
void ContextCache::checkpoint(Conv &c, bool pinned) {
    const int pos = (int)c.tokens.size();
    for (Checkpoint &cp : c.cps) if (cp.pos == pos) { cp.pinned |= pinned; return; }
    auto blob = std::make_shared<std::vector<uint8_t>>();
    m_.state_save(*blob);
    c.cps.push_back({pos, pinned, std::move(blob)});
    std::sort(c.cps.begin(), c.cps.end(), [](const Checkpoint &a, const Checkpoint &b) { return a.pos < b.pos; });
    // Of the unpinned ones, the latest are the useful ones.
    int unpinned = 0;
    for (const Checkpoint &cp : c.cps) unpinned += !cp.pinned;
    for (size_t i = 0; i < c.cps.size() && unpinned > o_.max_checkpoints;)
        if (!c.cps[i].pinned) { c.cps.erase(c.cps.begin() + i); unpinned--; } else i++;
}

void ContextCache::destroy(Conv *c) {
    m_.seq_destroy(c->seq);
    convs_.erase(std::find_if(convs_.begin(), convs_.end(), [&](auto &p) { return p.get() == c; }));
}

// The state on the GPU is about to be replaced. With linear-attention layers, what the current
// conversation holds beyond its last checkpoint can then no longer be continued, so it is dropped.
void ContextCache::leave_current() {
    if (!recurrent_ || !cur_) return;
    Conv &c = *cur_;
    const int tip = (int)c.tokens.size(), last = c.cps.empty() ? 0 : c.cps.back().pos;
    if (tip - last >= o_.leave_gap) checkpoint(c, false);
    const int end = c.cps.empty() ? 0 : c.cps.back().pos;
    if (end < tip) { m_.seq_truncate(end); c.tokens.resize(end); }
}

// Give up one conversation: one that has not been used for a long time if there is one, otherwise
// the one that is cheapest to bring back (the fewest pages of its own), the older of equals. A
// long conversation therefore outlives the short side requests made while it waits.
bool ContextCache::evict_one(const Conv *keep) {
    Conv *victim = nullptr; bool victim_stale = false; int victim_pages = 0;
    const auto now = std::chrono::steady_clock::now();
    for (auto &c : convs_) {
        if (c.get() == keep || c.get() == cur_) continue;
        const bool stale = now - c->last > std::chrono::seconds(o_.stale_seconds);
        const int pages = m_.pages_owned(c->seq);
        const bool better = !victim || (stale != victim_stale ? stale : stale ? c->used < victim->used : pages != victim_pages ? pages < victim_pages : c->used < victim->used);
        if (better) { victim = c.get(); victim_stale = stale; victim_pages = pages; }
    }
    if (!victim) return false;
    destroy(victim);
    return true;
}

// Free enough cache pages for the current conversation to grow to `upto` tokens.
void ContextCache::make_room(int upto) {
    while (m_.pages_needed(cur_->seq, upto) > m_.pages_free())
        if (!evict_one(cur_)) throw ContextFull(upto, m_.n_ctx());
}

ContextCache::Result ContextCache::prepare(const std::vector<int32_t> &prompt, const std::vector<int> &marks, const Progress &progress) {
    const int n = (int)prompt.size();
    if (n == 0) throw std::runtime_error("empty prompt");
    if (n + 1 > m_.n_ctx()) throw ContextFull(n, m_.n_ctx() - 1);

    // Where can each conversation be resumed without leaving the prompt?
    Conv *src = nullptr; int p = 0; const Checkpoint *from = nullptr; bool live = false;
    auto consider = [&](Conv *c, int pos, const Checkpoint *cp, bool is_live) {
        const bool better = pos > p || (pos == p && pos > 0 && ((is_live && !live) || (!live && !is_live && src && (int)c->tokens.size() == pos && (int)src->tokens.size() != pos)));
        if (better) { src = c; p = pos; from = cp; live = is_live; }
    };
    for (auto &up : convs_) {
        Conv *c = up.get();
        int common = 0;
        const int lim = std::min(n, (int)c->tokens.size());
        while (common < lim && prompt[common] == c->tokens[common]) common++;
        if (!recurrent_) {
            // Any position works; the last prompt token is processed again for its logits.
            consider(c, std::min(common, n - 1), nullptr, c == cur_ && common == (int)c->tokens.size() && common < n);
            continue;
        }
        if (c == cur_ && common == (int)c->tokens.size()) consider(c, common, nullptr, true);
        for (const Checkpoint &cp : c->cps) if (cp.pos <= common) consider(c, cp.pos, &cp, false);
    }

    Result res;
    std::shared_ptr<const std::vector<uint8_t>> state = from ? from->state : nullptr;   // survives changes to the checkpoint lists
    if (!live || p == 0) {
        leave_current();
        Conv *target;
        if (p == 0) {
            // Nothing to reuse: an empty conversation.
            target = nullptr;
            for (auto &c : convs_) if (c->tokens.empty()) target = c.get();
            if (!target) target = add_conv(m_.seq_create());
        } else if ((int)src->tokens.size() == p) {
            target = src;
        } else {
            // Branch off, sharing the cache pages of the common part.
            target = add_conv(m_.seq_fork(src->seq, p));
            target->tokens.assign(src->tokens.begin(), src->tokens.begin() + p);
            for (const Checkpoint &cp : src->cps) if (cp.pos <= p) target->cps.push_back(cp);
        }
        m_.seq_select(target->seq);
        if (recurrent_) { if (p == 0) m_.state_clear(); else m_.state_load(*state); }
        Conv *old = cur_;
        cur_ = target;
        if (old && old != target && old->tokens.empty()) destroy(old);
    } else if (recurrent_) {
        // Continuing where the engine already is. What follows may be abandoned again (a side
        // request that extends the conversation), so this point is worth keeping.
        const int last = cur_->cps.empty() ? 0 : cur_->cps.back().pos;
        if (p - last >= o_.min_gap) checkpoint(*cur_, false);
    }
    cur_->used = ++clock_; cur_->last = std::chrono::steady_clock::now();
    while ((int)convs_.size() > o_.max_conversations && evict_one(cur_)) {}
    make_room(n + 1);
    res.reused = p;

    const int chunk = std::max(1, m_.batch());
    size_t next_mark = 0;
    std::vector<int> stops(marks);
    std::sort(stops.begin(), stops.end());
    int done = p;
    while (done < n) {
        while (next_mark < stops.size() && stops[next_mark] <= done) next_mark++;
        int upto = std::min(n, done + chunk);
        if (next_mark < stops.size()) upto = std::min(upto, stops[next_mark]);
        m_.eval(prompt.data() + done, upto - done);
        cur_->tokens.insert(cur_->tokens.end(), prompt.begin() + done, prompt.begin() + upto);
        done = upto;
        if (recurrent_ && next_mark < stops.size() && stops[next_mark] == done) checkpoint(*cur_, true);
        if (progress && done < n && !progress(done, n)) { res.cancelled = true; break; }
    }
    res.evaluated = done - p;
    if (recurrent_ && !res.cancelled) {
        const int last = cur_->cps.empty() ? 0 : cur_->cps.back().pos;
        if (n - last >= o_.min_gap) checkpoint(*cur_, false);
    }
    return res;
}

bool ContextCache::push(int32_t token) {
    const int len = (int)cur_->tokens.size();
    if (len + 1 > m_.n_ctx()) return false;
    while (m_.pages_needed(cur_->seq, len + 1) > m_.pages_free())
        if (!evict_one(cur_)) return false;
    m_.eval(&token, 1);
    cur_->tokens.push_back(token);
    return true;
}

}  // namespace gfx
