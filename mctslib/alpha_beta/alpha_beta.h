#pragma once

#include <bit>
#include <cstdint>
#include <iostream>

#include "tt/tt.h"

// TODO: remove this dependency
#include "othello/othello_dumb7.h"

template <typename State, typename score_t>
class AlphaBeta {
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  static constexpr size_t kLevels = State::M * State::N + 1;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  static constexpr int32_t log_max_level = 11;
  static constexpr int32_t canonical_max_level = 32;

  uint64_t completions[kLevels] = {0ull};
  uint64_t cutoffs[kLevels][kLevels] = {{0ull}};

  TT<State, score_t> tt;

 public:
  void load_shared_tt(const std::string& file) {
    tt.full_tt.load_from(file);
  }
  void save_shared_tt(const std::string& file) {
    tt.full_tt.save_to(file);
  }


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
    if (state.finished()) {
      return state.score(0);
    }

    size_t slot = 0;
    score_t value;

    if (tt.template lookup_and_init<stones>(state, slot, alpha, beta, value)) {
      return value;
    }
    auto alpha0 = alpha;
    auto beta0 = beta;
    uint64_t moves;

    if constexpr (stones + 1 == State::M * State::N) {
      moves = OthelloDumb7Fill6x6::has_valid_move(state.board[state.player], state.board[1 - state.player]);
    } else {
      moves = state.valid_actions();
    }

    if (moves == 0ull) {
      State new_state = state;
      new_state.apply_skip();
      value = alpha_beta<stones, !do_max>(new_state, alpha, beta);
    } else if constexpr (stones + 1 == State::M * State::N) {
      // TODO: This is othello-specific cutoff. Do we want to move it somewhere?
      auto curr_score = state.score(0);
      if constexpr (do_max) {
        // valid move means we'll at least add our own + swap one
        if (curr_score + 3 >= beta) {
          return beta;
        }
      } else {
        if (curr_score - 3  <= alpha) {
          return alpha;
        }
      }
      State new_state = state;
      new_state.apply_move_mask(moves);
      value = new_state.score(0);
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
          //new_state = new_state.maybe_to_canonical(state, move);
          //auto c_state = new_state.to_canonical();
          //std::cout << (c_state == new_state) << std::endl;
          if constexpr (stones + 4 < canonical_max_level) {
            new_state = new_state.to_canonical();
          } else {
            new_state = new_state.maybe_to_canonical(state, move);
          }
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