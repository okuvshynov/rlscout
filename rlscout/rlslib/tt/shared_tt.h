#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

template <typename State, typename score_t>
struct SharedTT {
  using Self = SharedTT<State, score_t>;
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  SharedTT(size_t log_size) : log_size_(log_size) {
    data.resize(1ull << log_size);
  }

  struct Entry {
    State state;
    score_t low, high;
    bool free = true;
  };

  static constexpr size_t kLevels = 37;
  uint64_t tt_hits[kLevels] = {0ull};

  size_t find_slot(const State& state) const {
    size_t slot = state.hash() % data.size();
    while (true) {
      if (data[slot].free) {
        return slot;
      }
      if (data[slot].state == state) {
        return slot;
      }
      slot = (slot + 1) % data.size();
    }
  }

  template <uint32_t stones>
  bool lookup_and_init(const State& state, score_t& alpha,
                       score_t& beta, score_t& value) {
    size_t slot = find_slot(state);
    auto& entry = data[slot];
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
    size_t slot = find_slot(state);
    
    auto& entry = data[slot];
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
      if (tt_hits[d] == 0ull) {
        continue;
      }
      std::cout << d << " tt_hits " << tt_hits[d] << std::endl;
    }
  }

  const size_t log_size_;

  std::vector<Entry> data;
};