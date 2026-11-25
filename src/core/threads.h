// A minimal fork-join thread pool for the CPU reference paths.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gfx {

// Minimal fork-join pool: run(n, fn) calls fn(i) for i in [0, n) across the workers.
class ThreadPool {
  public:
    explicit ThreadPool(int n) {
        for (int i = 1; i < n; i++) workers_.emplace_back([this] { loop(); });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; gen_++; }
        cv_.notify_all();
        for (auto &t : workers_) t.join();
    }
    void run(size_t n, const std::function<void(size_t)> &fn) {
        if (workers_.empty() || n < 2) { for (size_t i = 0; i < n; i++) fn(i); return; }
        { std::lock_guard<std::mutex> l(m_); fn_ = &fn; n_ = n; next_ = 0; active_ = workers_.size(); gen_++; }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [this] { return active_ == 0; });
        fn_ = nullptr;
    }
  private:
    void work() {
        for (;;) {
            size_t i;
            { std::lock_guard<std::mutex> l(m_); if (next_ >= n_) return; i = next_++; }
            (*fn_)(i);
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            { std::unique_lock<std::mutex> l(m_); cv_.wait(l, [&] { return gen_ != seen; }); seen = gen_; if (stop_) return; }
            work();
            { std::lock_guard<std::mutex> l(m_); active_--; }
            done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_; std::condition_variable cv_, done_;
    const std::function<void(size_t)> *fn_ = nullptr;
    size_t n_ = 0, next_ = 0, active_ = 0; uint64_t gen_ = 0; bool stop_ = false;
};

}  // namespace gfx
