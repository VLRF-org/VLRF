// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Thread Pool  (thread_pool.hpp)
//
// Problem: the original parallel_for() spawned a fresh std::thread for every
// call and joined them before returning. A single token's forward pass makes
// roughly (4 attention projections + 3 FFN matrices) x n_layers GEMM calls —
// for Granite-3B that's ~224 GEMM calls per generated token, each spawning
// and joining a full set of OS threads.
//
// std::thread creation is notably expensive on Windows (kernel object alloc,
// TEB/TLS setup) — commonly 50-200us per thread, vs ~10-20us on Linux. With
// 4+ threads spawned per call, 224 calls per token, that overhead alone can
// reach 50-200ms per token: pure waste, no actual compute happening.
//
// Fix: a small persistent pool. Worker threads are created ONCE at process
// start (lazily, on first use) and parked on a condition variable between
// jobs. parallel_for() submits work and blocks until that batch completes,
// but never tears down or recreates OS threads.
// =============================================================================
#include <thread>
#include <vector>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>

namespace vlrf {
namespace ops {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) {
        workers_.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_task_.notify_all();
        for (auto& w : workers_) w.join();
    }

    int size() const { return (int)workers_.size(); }

    // Run fn(lo, hi) across n_jobs slices, splitting [0, total) into
    // n_jobs contiguous ranges. Blocks until all slices complete.
    //
    // REENTRANCY: if called from a thread that is already executing a job
    // submitted by THIS pool (e.g. softmax_inplace calling parallel_for
    // from inside a per-head job that attention's parallel_for is running),
    // we run the work INLINE on the calling thread instead of submitting
    // to the pool and waiting. Submitting+waiting in that situation is a
    // guaranteed deadlock: every worker thread can be busy running outer
    // jobs, so the nested submission has no free worker to service it,
    // while the calling worker sits blocked on cv_done_ forever — and if
    // ALL workers hit this simultaneously (the common case when n_jobs
    // for the outer call equals pool size), the pool is fully deadlocked.
    void run(int64_t total, int n_jobs,
            std::function<void(int64_t, int64_t)> const& fn) {
        if (total <= 0 || n_jobs <= 0) return;

        if (in_pool_job_) {
            // Nested call from inside a worker — run serially, no submission.
            fn(0, total);
            return;
        }

        n_jobs = std::min<int64_t>(n_jobs, total);
        if (n_jobs <= 1) { fn(0, total); return; }

        int64_t chunk = (total + n_jobs - 1) / n_jobs;

        pending_.store(n_jobs, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (int j = 0; j < n_jobs; ++j) {
                int64_t lo = (int64_t)j * chunk;
                int64_t hi = std::min(lo + chunk, total);
                if (lo >= hi) { pending_.fetch_sub(1, std::memory_order_relaxed); continue; }
                queue_.push_back([fn, lo, hi] { fn(lo, hi); });
            }
        }
        cv_task_.notify_all();

        // Wait for this batch to finish
        std::unique_lock<std::mutex> lk(done_mtx_);
        cv_done_.wait(lk, [this] {
            return pending_.load(std::memory_order_relaxed) == 0;
        });
    }

private:
    // Thread-local reentrancy guard: true while the CURRENT thread is
    // executing a job dequeued from this pool. Each worker thread gets
    // its own instance (thread_local), so this never needs synchronisation.
    static thread_local bool in_pool_job_;

    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_task_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            in_pool_job_ = true;
            job();
            in_pool_job_ = false;
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Last job in this batch — wake the waiting caller
                std::lock_guard<std::mutex> lk(done_mtx_);
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread>           workers_;
    std::deque<std::function<void()>>  queue_;
    std::mutex                         mtx_;
    std::condition_variable            cv_task_;
    std::mutex                         done_mtx_;
    std::condition_variable            cv_done_;
    std::atomic<int>                   pending_{0};
    bool                                stop_{false};
};

inline thread_local bool ThreadPool::in_pool_job_ = false;

// Global pool, sized to hardware concurrency, created on first use.
// Thread-safe lazy init (C++11 guarantees static local init is thread-safe).
inline ThreadPool& global_thread_pool() {
    static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

} // namespace ops
} // namespace vlrf
