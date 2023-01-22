#pragma once

#include "state.h"
#include "mcts.h"

#include <vector>

// player maintains buffer between MCTS invokations.
template<int m, int n, int k>
struct MCTSPlayer {
  std::vector<MCTSNode> buf;
  int32_t *boards_buffer;
  float *probs_buffer;
  MCTSPlayer(int32_t* boards_buffer, float* probs_buffer) 
    : boards_buffer(boards_buffer), probs_buffer(probs_buffer) { }

  void get_moves(State<m, n, k>* state, double temp, int rollouts, double* moves_out, void (*eval_cb)()) {
    // TODO: overflow
    size_t buf_size = rollouts * m * n;
    if (buf_size > buf.size()) {
      buf.resize(buf_size);
    }
    MCTS<m, n, k> mcts(buf, boards_buffer, probs_buffer, eval_cb);
    auto moves = mcts.run(*state, temp, rollouts);
    for (int i = 0; i < m * n; i++) {
        moves_out[i] = 0.0;
    }
    for (auto move: moves) {
        moves_out[move.first] = move.second;
    }
  }
};
