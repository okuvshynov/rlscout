#pragma once

#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>

#include "alpha_beta/move_generators.h"
#include "tt/tt.h"
#include "utils/model_evaluator.h"
#include "utils/py_log.h"

template <typename State, typename score_t, int32_t log_max_level = 11, int32_t canonical_max_level = 28, int32_t evaluate_nn_until_level = 20>
class AlphaBeta {
  static constexpr auto min_score = std::numeric_limits<score_t>::min();
  static constexpr auto max_score = std::numeric_limits<score_t>::max();

  static constexpr size_t kLevels = State::M * State::N + 1;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  uint64_t completions[kLevels] = {0ull};
  uint64_t cutoffs[kLevels][kLevels] = {{0ull}};

  TT<State, score_t> tt;

  ModelEvaluator* evaluator = nullptr;

 public:
  void set_model_evaluator(ModelEvaluator* evaluator) {
    this->evaluator = evaluator;
  }

  void log_stats_by_depth() {
    auto curr = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = curr - start;
    PYLOG << diff.count();

    for (size_t d = 0; d < kLevels; d++) {
      if (completions[d] == 0ull) {
        continue;
      }
      PYLOG << d << " completions " << completions[d];
    }
  }

  template <uint32_t stones, bool do_max>
  score_t alpha_beta(State state, score_t alpha, score_t beta) {
    if (state.finished()) {
      return state.score(0);
    }

    score_t value;

    if (tt.template lookup_and_init<stones>(state, alpha, beta, value)) {
      return value;
    }
    auto alpha0 = alpha;
    auto beta0 = beta;

    // no constexpr ternary operator; create a lambda and execute it.
    auto move_gen = [&]() {
      if constexpr (stones > evaluate_nn_until_level) {
        return PlainMoveGenerator<State, stones>(state);
      } else {
        return ActionModelMoveGenerator<State, stones>(state, *evaluator);
      }
    }();

    if (move_gen.moves() == 0ull) {
      State new_state = state;
      new_state.apply_skip();
      value = alpha_beta<stones, !do_max>(new_state, alpha, beta);
    } else if constexpr (stones + 1 == State::M * State::N) {
      state.apply_move_mask(move_gen.moves());
      return state.score(0);
    } else {
      value = do_max ? min_score : max_score;
      int32_t move_idx = 0;

      while (move_gen.moves()) {
        auto move = move_gen.next_move();

        State new_state = state;
        new_state.apply_move_mask(move);

        if constexpr (stones < canonical_max_level) {
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
        move_idx++;
      }
    }

    completions[stones]++;
    tt.template update<stones>(state, alpha0, beta0, value);

    if constexpr (stones < log_max_level) {
      log_stats_by_depth();
    }
    return value;
  }

  void print_tt_stats() { tt.log_stats(); }

  uint64_t total_completions() const {
    uint64_t res = 0ull;
    for (size_t i = 0; i < kLevels; i++) {
      res += completions[i];
    }
    return res;
  } 
};