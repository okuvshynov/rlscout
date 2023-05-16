#pragma once

#include "othello/othello_6x6_dumb7.h"
#include "utils/model_evaluator.h"

template <typename State, uint32_t stones>
struct PlainMoveGenerator {
  PlainMoveGenerator(const State& state) {
    if constexpr (stones + 1 == State::M * State::N) {
      moves_ = OthelloDumb7Fill6x6::single_valid_move(
          state.board[state.player], state.board[1 - state.player]);
    } else {
      moves_ = state.valid_actions();
    }
  }

  __attribute__((always_inline)) uint64_t next_move() {
    uint64_t other_moves = (moves_ & (moves_ - 1));
    uint64_t move = moves_ ^ other_moves;
    moves_ = other_moves;
    return move;
  }

  __attribute__((always_inline)) uint64_t moves() const { return moves_; }

 private:
  uint64_t moves_;
};

template <typename State, uint32_t stones>
struct ActionModelMoveGenerator {
  ActionModelMoveGenerator(const State& state,
                           const ModelEvaluator& evaluator) {
    state.fill_boards(evaluator.boards_buffer);
    // model_id = 0, no adding extra noise
    evaluator.run(0, false);
    if constexpr (stones + 1 == State::M * State::N) {
      moves_ = OthelloDumb7Fill6x6::single_valid_move(
          state.board[state.player], state.board[1 - state.player]);
    } else {
      moves_ = state.valid_actions();
    }
    for (uint64_t k = 0; k < State::M * State::N; k++) {
      if ((1ull << k) & moves_) {
        ordered_moves_.emplace_back(evaluator.probs_buffer[k], (1ull << k));
      }
    }
    std::sort(ordered_moves_.begin(), ordered_moves_.end());
  }

  __attribute__((always_inline)) uint64_t moves() const { return moves_; }

  __attribute__((always_inline)) uint64_t next_move() {
    uint64_t res = ordered_moves_.back().second;
    ordered_moves_.pop_back();
    moves_ = moves_ ^ res;

    return res;
  }

 private:
  uint64_t moves_;
  std::vector<std::pair<float, uint64_t>> ordered_moves_;
};
