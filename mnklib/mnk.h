#pragma once

#include "state.h"
#include "mcts.h"

#include <vector>

// player maintains buffer between MCTS invokations.
template<int m, int n, int k>
struct MCTSPlayer {
  std::vector<MCTSNode> buf;

  void get_moves(State<m, n, k>* state, double temp, int rollouts, double* moves_out) {
    // TODO: overflow
    size_t buf_size = rollouts * m * n;
    if (buf_size > buf.size()) {
      buf.resize(buf_size);
    }
    MCTS<m, n, k> mcts(buf);
    auto moves = mcts.run(*state, temp, rollouts);
    for (int i = 0; i < m * n; i++) {
        moves_out[i] = 0.0;
    }
    for (auto move: moves) {
        moves_out[move.first] = move.second;
    }
  }
};
