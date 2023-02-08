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

struct MCTS {
  std::vector<MCTSNode>& nodes;
  int size = 0;
  int root_id = 0;

  // no ownership over buffer
  MCTS(std::vector<MCTSNode>& nodes)
      : nodes(nodes) {
  }

  void reset() {
    nodes[0] = MCTSNode(1.0);
    size = 1;
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

  void record(int node_id, double value) {
    auto& node = nodes[node_id];
    if (node.parent >= 0) {
      record(node.parent, -value);
    }
    node.Q = (node.Q * node.N + value) / (node.N + 1);
    node.N++;
  }
};