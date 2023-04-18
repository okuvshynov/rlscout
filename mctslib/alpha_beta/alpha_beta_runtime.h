#pragma once

#include "alpha_beta/alpha_beta.h"

// this is a version which does a dispatch
// from run-time only available state

template <typename State, typename score_t>
class AlphaBetaRuntime {
  template <uint32_t stones>
  struct MaxRunner {
    score_t run(AlphaBeta<State, score_t> &ab, State s, double alpha,
                double beta) {
      score_t res = ab.template alpha_beta<stones, true>(s, alpha, beta);
      return res;
    }
  };

  template <uint32_t stones>
  struct MinRunner {
    score_t run(AlphaBeta<State, score_t> &ab, State s, double alpha,
                double beta) {
      return ab.template alpha_beta<stones, false>(s, alpha, beta);
    }
  };

  template <template <uint32_t V> class D, typename... ArgsT, uint32_t... X>
  score_t dispatch_impl(uint32_t x, ArgsT... args,
                        std::integer_sequence<uint32_t, X...>) {
    score_t res;
    ((x == X ? res = D<X>().run(args...) : score_t()), ...);
    return res;
  }

  template <template <uint32_t V> class D, uint32_t B, typename... ArgsT>
  score_t dispatch(uint32_t x, ArgsT... args) {
    if (x >= 0 && x < B) {
      return dispatch_impl<D, ArgsT...>(
          x, args..., std::make_integer_sequence<uint32_t, B>{});
    }
    return 0;
  }

 public:
  std::vector<std::pair<uint64_t, score_t>> get_move_scores(State state,
                                                            score_t alpha,
                                                            score_t beta) {
    bool do_max = (state.player == 0);
    uint32_t stones = state.stones_played();

    uint64_t moves = state.valid_actions();
    std::vector<std::pair<uint64_t, score_t>> res;

    while (moves) {
      uint64_t other_moves = (moves & (moves - 1));
      uint64_t move = moves ^ other_moves;
      State new_state = state;
      new_state.apply_move_mask(move);
      score_t score;
      if (do_max) {
        score = dispatch<MinRunner, State::M * State::N>(
            stones, std::ref(alpha_beta), new_state, alpha, beta);
      } else {
        score = dispatch<MaxRunner, State::M * State::N>(
            stones, std::ref(alpha_beta), new_state, alpha, beta);
      }
      res.emplace_back(move, score);
      moves = other_moves;
    }

    //alpha_beta.log_stats_by_depth();

    return res;
  }

 private:
  AlphaBeta<State, score_t> alpha_beta;
};