#pragma once

#include <iostream>
#include <memory>
#include <cmath>

#include "state.h"

struct MCTSNode {
  uint64_t in_action;
  double N = 0.0;
  double Q = 0.0;
  double P = 0.0;
  int parent = -1;

  // rather than creating nodes dynamically we'll create a contiguous buffer
  // in advance and fill it in. Children of current node will be stored in that
  // buffer at indicies [children_from; children_from + children_count)

  int children_from = 0;
  int children_count = 0;
  MCTSNode(double prior) : P(prior) {};
  MCTSNode() : P(1.0) {};
};

template<int m, int n, int k>
struct MCTS {
  std::vector<MCTSNode>& nodes;
  int size = 0;
  int root_id = 0;
  int32_t* boards_buffer;
  float* probs_buffer;
  void (*eval_cb)();

  // no ownership over buffer
  MCTS(std::vector<MCTSNode>& nodes, int32_t *boards_buffer, float *probs_buffer, void (*eval_cb)())
      : nodes(nodes), boards_buffer(boards_buffer), probs_buffer(probs_buffer), eval_cb(eval_cb) {
    nodes[0] = MCTSNode(1.0);
    size = 1;
  }

  ~MCTS() {
  }

  double value(const MCTSNode& node, double temp) const {
    double res = node.Q;
    if (node.parent != -1) {
      res += node.P * temp * std::sqrt(nodes[node.parent].N) / (node.N + 1.0);
    }
    return res;
  }

  int select(MCTSNode& node, double temp) const {
    double mx = -10000.0;
    int32_t mxi = -1;
    for (int i = 0; i < node.children_count; i++) {
      double val = value(nodes[node.children_from + i], temp);
      if (val > mx) {
        mx = val;
        mxi = i;
      }
    }
    return node.children_from + mxi;
  }

  void expand(int node_id, State<m,n,k>& state) {
    uint64_t moves = state.valid_actions();
    nodes[node_id].children_from = size;
    int j = size;

    state.fill_boards(boards_buffer);
    if (eval_cb != nullptr) {
      eval_cb();
    }

    for (uint64_t i = 0; i < m * n; i++) {
      if ((1ull << i) & moves) {
        nodes[j] = MCTSNode(probs_buffer[i]);
        nodes[j].parent = node_id;
        nodes[j].in_action = i;
        j++;
      }
    }
    nodes[node_id].children_count = j - size;
    size = j;
  }

  void record(int node_id, double value) {
    auto& node = nodes[node_id];
    if (node.parent >= 0) {
      record(node.parent, -value);
    }
    node.Q = (node.Q * node.N + value) / (node.N + 1);
    node.N++;
  }

  std::vector<std::pair<int64_t, int64_t>> run(State<m, n, k> state, double temp, int32_t rollouts) {
    for (int i = 0; i < rollouts; i++) {
      State<m,n,k> s = state; // copy here
      int node_id = root_id;
      while (nodes[node_id].children_count > 0) {
        node_id = select(nodes[node_id], temp);
        s.apply_move(nodes[node_id].in_action);
      }

      int current_player = 1 - s.player;

      if (!s.finished()) {
        expand(node_id, s);
      }

      while (!s.finished()) {
        s.take_random_action();
      }

      double value = s.score(current_player);
      record(node_id, value);
    }
    std::vector<std::pair<int64_t, int64_t>> res;
    for (int i = 0; i < nodes[root_id].children_count; i++) {
      const auto& node = nodes[nodes[root_id].children_from + i];
      res.emplace_back(std::make_pair(node.in_action, node.N));
    }
    return res;
  }
};
