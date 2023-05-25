#pragma once

#include "othello/othello_6x6_dumb7.h"
#include "utils/model_evaluator.h"
#include "utils/random.h"
#include "utils/py_log.h"

#include <vector>
#include <deque>

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

template <typename State, uint32_t stones>
struct SamplingActionModelMoveGenerator {
  static constexpr float kTemp = 1.0f;
  SamplingActionModelMoveGenerator(const State& state,
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
    std::vector<int> moves(State::M * State::N);
    size_t total_moves = 0ll;
    for (uint64_t k = 0; k < State::M * State::N; k++) {
      if ((1ull << k) & moves_) {
        total_moves++;

        moves[k] = 1 + int(1000000 * std::pow(evaluator.probs_buffer[k], 1.0f / kTemp));
      } else {
        moves[k] = 0;
      }
    }

    while (total_moves > 0) {
      std::discrete_distribution<> d(moves.begin(), moves.end());
      uint64_t index = d(RandomGen::gen());
      if constexpr (stones < 12) {
        PYLOG << stones << " picking moves for " << state.hash() << " " << index << " " << moves[index];
      }
      ordered_moves_.push_back(1ull << index);
      moves[index] = 0;
      total_moves--;
    }
  }

  __attribute__((always_inline)) uint64_t moves() const { return moves_; }

  __attribute__((always_inline)) uint64_t next_move() {
    uint64_t res = ordered_moves_.front();
    ordered_moves_.pop_front();
    moves_ = moves_ ^ res;

    return res;
  }

 private:
  uint64_t moves_;
  std::deque<uint64_t> ordered_moves_;
};
