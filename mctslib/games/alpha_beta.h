#pragma once

#include <cstdint>
#include <iostream>

#include "tt.h"

template <typename State, typename score_t>
class AlphaBeta {
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  static constexpr size_t kLevels = 37;
  std::chrono::steady_clock::time_point start;

  uint64_t leaves = 0ull;
  uint64_t tt_hits[kLevels] = {0ull};
  uint64_t completions[kLevels] = {0ull};
  uint64_t cutoffs[kLevels][kLevels] = {{0ull}};
  uint64_t evictions[kLevels] = {0ull};
  TT<State, score_t> full_tt = TT<State, score_t>{27};
  static constexpr uint32_t tt_full_level = 24;

  static constexpr int32_t tt_max_level = 33;
  static constexpr int32_t log_max_level = 11;
  static constexpr int32_t canonical_max_level = 30;

  static constexpr size_t tt_size = 1 << 24;

  struct TTEntry {
    State state;
    score_t low, high;
  };

  std::vector<TTEntry> transposition_table[kLevels];

  static constexpr size_t tt_sizes[kLevels] = {
      17,          17,          17,          17,      17,          17,
      17,          17,          17,          17,      17,  // 0-10
      1,           1,           1,           1,       1,           1,
      1,           1,           1,           1,                     // 11-20
      1,           1,           1,           tt_size, tt_size,      // 21 - 25
      tt_size,     tt_size,     tt_size,     tt_size, tt_size * 2,  // 26-30
      tt_size * 2, tt_size * 2, tt_size * 2,                        // 31-33
      17,          17,          17};

 public:
  AlphaBeta() {
    start = std::chrono::steady_clock::now();
    init_tt();
    auto curr = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = curr - start;
    std::cout << "TT init done at " << diff.count() << std::endl;
  }
  void init_tt() {
    for (size_t i = 0; i < kLevels; i++) {
      transposition_table[i].resize(tt_sizes[i]);
      for (auto& p : transposition_table[i]) {
        p.state.board[0] = 0ull;
        p.state.board[1] = 0ull;
      }
    }
  }

  void log_stats_by_depth() {
    auto curr = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = curr - start;
    std::cout << diff.count() << std::endl;
    for (size_t d = 0; d < kLevels; d++) {
      if (completions[d] == 0ull && tt_hits[d] == 0ull) {
        continue;
      }
      std::cout << d << " tt_hits " << tt_hits[d] << " completions "
                << completions[d] << " evictions " << evictions[d] << std::endl;
    }

    std::cout << "cutoff idxs " << std::endl;
    for (size_t d = 0; d < kLevels; d++) {
      std::cout << d << ": ";
      for (size_t l = 0; l < kLevels; l++) {
        std::cout << cutoffs[d][l] << " ";
      }
      std::cout << std::endl;
    }
  }

  template <uint32_t stones, bool do_max>
  score_t alpha_beta(State state, score_t alpha, score_t beta) {
    if (state.finished() || state.full()) {
      return state.score(0);
    }

    size_t slot = 0;

    if constexpr (stones < tt_full_level) {
      slot = full_tt.find_slot(state);
      auto& entry = full_tt.data[slot];
      if (!entry.free) {
        if (entry.low >= beta) {
          tt_hits[stones]++;
          return entry.low;
        }
        if (entry.high <= alpha) {
          tt_hits[stones]++;
          return entry.high;
        }
        alpha = std::max(alpha, entry.low);
        beta = std::min(beta, entry.high);
      } else {
        entry.low = min_score;
        entry.high = max_score;
      }
    } else if constexpr (stones < tt_max_level) {
      slot = state.hash() % tt_sizes[stones];

      if (transposition_table[stones][slot].state == state) {
        if (transposition_table[stones][slot].low >= beta) {
          tt_hits[stones]++;
          return transposition_table[stones][slot].low;
        }
        if (transposition_table[stones][slot].high <= alpha) {
          tt_hits[stones]++;
          return transposition_table[stones][slot].high;
        }

        alpha = std::max(alpha, transposition_table[stones][slot].low);
        beta = std::min(beta, transposition_table[stones][slot].high);
      } else {
        if (!transposition_table[stones][slot].state.empty()) {
          evictions[stones]++;
        }
        // override / init slot
        transposition_table[stones][slot].low = min_score;
        transposition_table[stones][slot].high = max_score;
      }
    }
    auto alpha0 = alpha;
    auto beta0 = beta;
    auto moves = state.valid_actions();
    score_t value;
    if (moves == 0ull) {
      State new_state = state;
      new_state.apply_skip();
      value = alpha_beta<stones, !do_max>(new_state, alpha, beta);
    } else if constexpr (stones + 1 == State::M * State::N) {
      auto score = state.score(0);
      if constexpr (do_max) {
        if (score + 2 >= beta) {
          return beta;
        }
        if (score + state.max_flip_score() <= alpha) {
          return alpha;
        }
      } else {
        if (score - 2 <= alpha) {
          return alpha;
        }
        if (score - state.max_flip_score() >= beta) {
          return beta;
        }
      }
      State new_state = state;
      new_state.apply_move_mask(moves);
      return new_state.score(0);
    } else {
      value = do_max ? min_score : max_score;
      int32_t move_idx = 0;
      while (moves) {
        uint64_t other_moves = (moves & (moves - 1));
        uint64_t move = moves ^ other_moves;
        State new_state = state;
        new_state.apply_move_mask(move);
        if constexpr (stones + 1 < canonical_max_level) {
          new_state = new_state.to_canonical();
        }

        auto new_value =
            alpha_beta<stones + 1, !do_max>(new_state, alpha, beta);

        if constexpr (do_max) {
          value = std::max(value, new_value);
          alpha = std::max(alpha, value);
          if (value >= beta) {
            cutoffs[stones][move_idx]++;
            break;
          }
        } else {
          value = std::min(value, new_value);
          beta = std::min(beta, value);
          if (value <= alpha) {
            cutoffs[stones][move_idx]++;
            break;
          }
        }

        moves = other_moves;
        move_idx++;
      }
    }

    completions[stones]++;
    if constexpr (stones < tt_full_level) {
      auto& entry = full_tt.data[slot];
      entry.free = false;
      entry.state = state;
      if (value <= alpha0) {
        entry.high = value;
      }
      if (value >= beta0) {
        entry.low = value;
      }
      if (value > alpha0 && value < beta0) {
        entry.high = value;
        entry.low = value;
      }
      if constexpr (stones < log_max_level) {
        log_stats_by_depth();
      }
    } else if constexpr (stones < tt_max_level) {
      transposition_table[stones][slot].state = state;

      if (value <= alpha0) {
        transposition_table[stones][slot].high = value;
      }
      if (value >= beta0) {
        transposition_table[stones][slot].low = value;
      }
      if (value > alpha0 && value < beta0) {
        transposition_table[stones][slot].high = value;
        transposition_table[stones][slot].low = value;
      }
      if constexpr (stones < log_max_level) {
        log_stats_by_depth();
      }
    }
    return value;
  }
};