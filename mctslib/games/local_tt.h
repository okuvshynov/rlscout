#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <array>

template<size_t N, size_t B1, size_t B2, size_t TSize, size_t... Is>
constexpr std::array<size_t, N> gen_ttable_sizes(std::index_sequence<Is...>) {
  return {{(Is < B1 ? 1 : (Is < B2 ? TSize : TSize * 2))...}};
}

template <typename State, typename score_t, size_t kFullLevel, size_t kLevels>
struct LocalTT {
  static constexpr size_t tt_size = 1 << 24;

  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  struct TTEntry {
    State state;
    score_t low, high;
  } __attribute__((packed));

  std::vector<TTEntry> data[kLevels];
  
  static constexpr std::array<size_t, kLevels> tt_sizes = gen_ttable_sizes<kLevels, kFullLevel, 30, tt_size>(std::make_index_sequence<kLevels>{});

  uint64_t tt_hits[kLevels] = {0ull};
  uint64_t evictions[kLevels] = {0ull};

  void init_tt() {
    for (size_t i = 0; i < kLevels; i++) {
      data[i].resize(tt_sizes[i]);
      for (auto& p : data[i]) {
        p.state.board[0] = 0ull;
        p.state.board[1] = 0ull;
      }
    }
  }

  void print_stats() const {
    for (size_t d = 0; d < kLevels; d++) {
      if (tt_hits[d] == 0ull) {
        continue;
      }
      std::cout << d << " tt_hits " << tt_hits[d] << " evictions "
                << evictions[d] << std::endl;
    }
  }

  template <uint32_t stones>
  bool lookup_and_init(const State& state, size_t& slot, score_t& alpha,
                       score_t& beta, score_t& value) {
    static_assert(stones < kLevels);
    slot = state.hash() % tt_sizes[stones];

    if (data[stones][slot].state == state) {
      if (data[stones][slot].low >= beta) {
        tt_hits[stones]++;
        value = data[stones][slot].low;
        return true;
      }
      if (data[stones][slot].high <= alpha) {
        tt_hits[stones]++;
        value = data[stones][slot].high;
        return true;
      }

      
      alpha = std::max(alpha, data[stones][slot].low);
      beta = std::min(beta, data[stones][slot].high);
    } else {
      if (!data[stones][slot].state.empty()) {
        evictions[stones]++;
      }
      // override / init slot
      data[stones][slot].low = min_score;
      data[stones][slot].high = max_score;
    }
    return false;
  }

  template <uint32_t stones>
  void update(const State& state, size_t& slot, score_t& alpha, score_t& beta,
              score_t& value) {
    //std::cout << int(value) << " " << int(alpha) << " " << int(beta) << " " << int(data[stones][slot].low) << " " << int(data[stones][slot].high) << std::endl;
    static_assert(stones < kLevels);
    data[stones][slot].state = state;

    if (value <= alpha) {
      data[stones][slot].high = value;
    }
    if (value >= beta) {
      data[stones][slot].low = value;
    }
    if (value > alpha && value < beta) {
      data[stones][slot].high = value;
      data[stones][slot].low = value;
    }
  }
};
