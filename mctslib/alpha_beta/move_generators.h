#pragma once

#include "othello/othello_dumb7.h"

template <typename State, uint32_t stones>
struct PlainMoveGenerator {
  PlainMoveGenerator(const State& state) {
    if constexpr (stones + 1 == State::M * State::N) {
      moves_ = OthelloDumb7Fill6x6::has_valid_move(state.board[state.player], state.board[1 - state.player]);
    } else {
      moves_ = state.valid_actions();
    }
  }

  __attribute__((always_inline))
  uint64_t next_move() {
    // This operation picks the next move to explore
    // when we add model evaluation here, we need to change this thing here.
    uint64_t other_moves = (moves_ & (moves_ - 1));
    uint64_t move = moves_ ^ other_moves;
    moves_ = other_moves;
    return move;
  }
  
  __attribute__((always_inline))
  uint64_t moves() const {
    return moves_;
  }
 private:
  uint64_t moves_;
};
