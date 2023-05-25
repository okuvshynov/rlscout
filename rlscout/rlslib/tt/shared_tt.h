#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <mutex>
#include <atomic>

#include "utils/py_log.h"

template <typename State, typename score_t, size_t log_size>
struct SharedTT {
  using Self = SharedTT<State, score_t, log_size>;
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();
  static constexpr size_t log_buckets = 3;
  static constexpr size_t buckets = (1ull << log_buckets);

  static Self& instance() {
    static Self self;
    return self;
  }
  
  struct Entry {
    State state;
    score_t low, high;
    bool free = true;
  };

  static constexpr size_t kLevels = 37;
  std::atomic<uint64_t> tt_hits[kLevels] = {0ull};
  std::atomic<uint64_t> uniques[kLevels] = {0ull};

  std::atomic<uint64_t> accesses[buckets] = {0ull};

  template <uint32_t stones>
  bool lookup_and_init(const State& state, score_t& alpha,
                       score_t& beta, score_t& value) {
    auto h = state.hash();
    auto& d = data[h % buckets];
    accesses[h % buckets]++;
    std::lock_guard<std::mutex> g(bucket_mutex[h % buckets]);

    size_t slot = (h / buckets) % d.size();
    while (true) {
      if (d[slot].free) {
        break;
      }
      if (d[slot].state == state) {
        break;
      }
      slot = (slot + 1) % d.size();
    }

    auto& entry = d[slot];
    if (!entry.free) {
      if (entry.low >= beta) {
        tt_hits[stones]++;
        value = entry.low;
        return true;
      }
      if (entry.high <= alpha) {
        tt_hits[stones]++;
        value = entry.high;
        return true;
      }
      alpha = std::max(alpha, entry.low);
      beta = std::min(beta, entry.high);
    } else {
      entry.low = min_score;
      entry.high = max_score;
    }
    return false;
  }

  template <uint32_t stones>
  void update(const State& state, score_t& alpha, score_t& beta,
              score_t& value) {
    auto h = state.hash();
    auto& d = data[h % buckets];
    accesses[h % buckets]++;

    std::lock_guard<std::mutex> g(bucket_mutex[h % buckets]);

    size_t slot = (h / buckets) % d.size();
    while (true) {
      if (d[slot].free) {
        break;
      }
      if (d[slot].state == state) {
        break;
      }
      slot = (slot + 1) % d.size();
    }
    
    auto& entry = d[slot];
    if (entry.free) {
      uniques[stones]++;
    }
    if constexpr (stones < 12) {
      PYLOG << "updating " << stones << " " << h;
    }
    entry.free = false;
    entry.state = state;
    if (value <= alpha) {
      entry.high = value;
    }
    if (value >= beta) {
      entry.low = value;
    }
    if (value > alpha && value < beta) {
      entry.high = value;
      entry.low = value;
    }
  }

  void print_stats() const {
    for (size_t d = 0; d < kLevels; d++) {
      if (uniques[d] == 0ull) {
        continue;
      }
      PYLOG << d << " tt_hits " << tt_hits[d] << " unique " << uniques[d];
    }
  }

 private:
  SharedTT() {
    for (size_t b = 0; b < buckets; b++) {
      data[b].resize(1ull << (log_size - log_buckets));
      PYLOG << b << " " << data[b].size();
    }
  }
  std::mutex bucket_mutex[buckets];
  std::vector<Entry> data[buckets];
};