#pragma once

#include <bit>
#include <cstdint>
#include <iostream>

#include "tt.h"

template <typename State, typename score_t>
class AlphaBeta {
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  static constexpr size_t kLevels = 37;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  static constexpr int32_t log_max_level = 11;
  static constexpr int32_t canonical_max_level = 30;

  uint64_t completions[kLevels] = {0ull};
  uint64_t cutoffs[kLevels][kLevels] = {{0ull}};

  TT<State, score_t> tt;

 public:
  void log_stats_by_depth() {
    auto curr = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = curr - start;
    std::cout << diff.count() << std::endl;

    for (size_t d = 0; d < kLevels; d++) {
      if (completions[d] == 0ull) {
        continue;
      }
      std::cout << d << " completions " << completions[d] << std::endl;
    }
    tt.log_stats();
  }

  template <uint32_t stones, bool do_max>
  score_t alpha_beta(State state, score_t alpha, score_t beta) {
    if (state.finished() || state.full()) {
      return state.score(0);
    }

    size_t slot = 0;
    score_t value;

    if (tt.template lookup_and_init<stones>(state, slot, alpha, beta, value)) {
      return value;
    }
    auto alpha0 = alpha;
    auto beta0 = beta;
    auto moves = state.valid_actions();

    /*if constexpr (stones + 2 == State::M * State::N) {
      std::cout << std::popcount(moves) << " " << state.score(0) << std::endl;
      state.p();
      std::cout << std::endl;
    }*/

    if (moves == 0ull) {
      State new_state = state;
      new_state.apply_skip();
      value = alpha_beta<stones, !do_max>(new_state, alpha, beta);
    } else if constexpr (stones + 1 == State::M * State::N) {
      // TODO: This is othello-specific cutoff. Do we want to move it somewhere?
      auto score = state.score(0);
      if constexpr (do_max) {
        // valid move means we'll at least add our own + swap one
        if (score + 3 >= beta) {
          return beta;
        }
      } else {
        if (score - 3  <= alpha) {
          return alpha;
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

        // for lower depth we pick the smallest of the symmetries
        // for high depth that operation itself becomes expensive enough
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
    tt.template update<stones>(state, slot, alpha0, beta0, value);

    if constexpr (stones < log_max_level) {
      log_stats_by_depth();
    }
    return value;
  }
};