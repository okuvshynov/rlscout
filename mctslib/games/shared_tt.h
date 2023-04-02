#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

// For 8x8 case this would be remote and there's not too much point optimizing it a lot.

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

  void save_to(const std::string& filename) const {
    auto out = std::ofstream(filename);

    out << log_size_ << std::endl;

    for (size_t i = 0; i < data.size(); i++) {
      const auto& entry = data[i];
      if (!entry.free) {
        out << i << " " << entry.state << " " << int64_t(entry.low) << " "
            << int64_t(entry.high) << std::endl;
      }
    }
  }

  static Self load_from(const std::string& filename) {
    auto in = std::ifstream(filename);
    size_t log_size, i;
    in >> log_size;
    Self res(log_size);
    std::cout << log_size << std::endl;
    uint64_t low, high;
    while (in >> i) {
      auto& entry = res.data[i];
      entry.free = false;
      in >> entry.state >> low >> high;
      entry.low = low;
      entry.high = high;
      if (entry.state.stones_played() < 8) {
        entry.state.p();
        std::cout << entry.low << " " << entry.high << std::endl;
      }
    }
    return res;
  }

  template <uint32_t stones>
  bool lookup_and_init(const State& state, size_t& slot, score_t& alpha,
                     score_t& beta, score_t& value) {
    slot = find_slot(state);
    auto& entry = data[slot];
    if (!entry.free) {
      if (entry.low >= beta) {
        // tt_hits[stones]++;
        value = entry.low;
        return true;
      }
      if (entry.high <= alpha) {
        // tt_hits[stones]++;
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
  void update(const State& state, size_t& slot, score_t& alpha,
                      score_t& beta, score_t& value) {
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


  const size_t log_size_;

  std::vector<Entry> data;
};