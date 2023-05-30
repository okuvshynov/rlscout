#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "utils/model_evaluator.h"
#include "utils/py_log.h"
#include "utils/random.h"

/*

This is batched implementation of MCTS (or, rather, PUCT) algorithm.
It is more efficient as we can apply NN to a batch of samples rather than
individual sample.
Rather than parallelizing each individual MCTS instance we just play multiple
games at a time.

For self-play and player evaluation we care more about throughput than latency,
thus, we can serially evaluate and prepare the board for evaluation, evaluate
them in 1 batch and then continue serially.

Multithreading (and another layer of batching) can be applied on top if needed.

*/

using LogFn = void (*)(int64_t, int8_t, int8_t);
using GameDoneFn = bool (*)(int32_t, int64_t);
using ModelIdFn = uint32_t (*)();

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
  MCTSNode(double prior) : P(prior){};
  MCTSNode() : P(1.0){};
};

template <typename State>
struct GameSlot {
  GameSlot() {
    restart();
  }

  void restart() {
    state = State();
    auto dis = std::uniform_int_distribution<int64_t>(
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max());
    game_id = dis(RandomGen::gen());
  }

  // TODO: this returns 0 when there are no valid moves
  // it is not causing a visible bug because we do another
  // check in the state, but it needs to be fixed
  uint64_t get_move_index(uint32_t explore_for_n_moves) {
    static thread_local std::array<uint64_t, State::M * State::N> visits;
    std::fill(visits.begin(), visits.end(), 0ull);

    for (int i = 0; i < nodes[root_id].children_count; i++) {
      const auto &node = nodes[nodes[root_id].children_from + i];
      visits[node.in_action] = node.N;
    }

    if (state.stones_played() >= explore_for_n_moves) {
      // pick greedily
      auto it = std::max_element(visits.begin(), visits.end());
      return std::distance(visits.begin(), it);
    } else {
      // sample
      std::discrete_distribution<> d(visits.begin(), visits.end());
      return d(RandomGen::gen());
    }
  }

  void log_training_data(LogFn log_freq_cb, int32_t *boards_buffer,
                         float *freq_buffer) {
    if (log_freq_cb == nullptr) {
      return;
    }
    state.fill_boards(boards_buffer);
    for (int i = 0; i < State::N * State::M; i++) {
      freq_buffer[i] = 0.0f;
    }
    for (int i = 0; i < nodes[root_id].children_count; i++) {
      const auto &node = nodes[nodes[root_id].children_from + i];
      freq_buffer[node.in_action] = node.N;
    }
    log_freq_cb(game_id, state.player, state.skipped);
  }

  void reset(size_t buffer_size) {
    nodes.resize(buffer_size);
    nodes[0] = MCTSNode(1.0);
    size = 1;
  }

  double value(const MCTSNode &node, double temp) const {
    double res = node.Q;
    if (node.parent != -1) {
      res += node.P * temp * std::sqrt(nodes[node.parent].N) / (node.N + 1.0);
    }
    return res;
  }

  int select(MCTSNode &node, double temp) const {
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
    auto &node = nodes[node_id];
    if (node.parent >= 0) {
      record(node.parent, -value);
    }
    node.Q = (node.Q * node.N + value) / (node.N + 1);
    node.N++;
  }

  bool slot_active = true;

  // per game instance scope
  State state;
  int64_t game_id;

  // per each move scope
  std::vector<MCTSNode> nodes;
  int size = 0;
  // currently we do not reuse search tree during self-play at all. In this
  // case, there's no need to store root_id. We might want to add it in future
  // though.
  int root_id = 0;

  // within current search state scope
  State rollout_state;
  int node_id;
  int last_player;
};

// returns one move in all active games in all slots
template <typename State>
std::vector<uint64_t> get_moves(std::vector<GameSlot<State>> &game_slots,
                                int32_t rollouts, double temp,
                                ModelEvaluator evaluator, uint32_t model_id,
                                uint32_t explore_for_n_moves,
                                uint32_t random_rollouts) {
  for (auto &g : game_slots) {
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    g.reset(rollouts * State::M * State::N);
  }

  static const int kBoardElements = 2 * State::M * State::N;
  static const int kProbElements = State::M * State::N;

  for (int32_t r = 0; r < rollouts; r++) {
    // serially traverse the tree for each active game slot
    // and prepare the board for each game for evaluation
    for (size_t i = 0; i < game_slots.size(); ++i) {
      // puct_cycles++;
      auto &g = game_slots[i];
      if (!g.slot_active || g.state.finished()) {
        // puct_wasted_cycles++;
        continue;
      }

      // traverse the tree until we get to leaf node
      g.rollout_state = g.state;
      g.node_id = g.root_id;
      while (g.nodes[g.node_id].children_count > 0) {
        g.node_id = g.select(g.nodes[g.node_id], temp);
        g.rollout_state.apply_move(g.nodes[g.node_id].in_action);
      }

      g.last_player = 1 - g.rollout_state.player;

      // if leaf node was 'end-of-game' there's no need to evaluate action model
      if (g.rollout_state.finished()) {
        // puct_wasted_cycles++;
        continue;
      }

      if (evaluator.run != nullptr) {
        g.rollout_state.fill_boards(evaluator.boards_buffer +
                                    i * kBoardElements);
      }
    }

    // now evaluate model for all the game slots (including inactive ones)
    // we'll just ignore the output for that part of input.
    // eval_cb takes boards_buffer as an input and writes results to
    // probs_buffer and scores buffer

    bool use_scores_from_model = false;
    if (evaluator.run != nullptr && model_id != 0) {
      evaluator.run(model_id, r == 0);
      use_scores_from_model = true;
    }

    for (size_t i = 0; i < game_slots.size(); ++i) {
      auto &g = game_slots[i];
      if (!g.slot_active || g.state.finished()) {
        continue;
      }

      if (!g.rollout_state.finished()) {
        g.nodes[g.node_id].children_from = g.size;
        uint64_t moves = g.rollout_state.valid_actions();
        int j = g.size;
        for (uint64_t k = 0; k < State::M * State::N; k++) {
          if ((1ull << k) & moves) {
            g.nodes[j] = MCTSNode(
                model_id > 0 ? evaluator.probs_buffer[i * kProbElements + k]
                             : 1.0f);
            g.nodes[j].parent = g.node_id;
            g.nodes[j].in_action = k;
            j++;
          }
        }
        g.nodes[g.node_id].children_count = j - g.size;
        g.size = j;
        if (use_scores_from_model) {
          //PYLOG << "value estimate: " << evaluator.scores_buffer[i] << " " << val;
          g.record(g.node_id, evaluator.scores_buffer[i]);
          //continue;
        } else {
          std::uniform_real_distribution<> dis(-1.0, 1.0);
          g.record(g.node_id, dis(RandomGen::gen()));
        }

        /*
        double val = 0.0;
        for (uint32_t rr = 0; rr < random_rollouts; rr++) {
          auto temp_state = g.rollout_state;
          while (!temp_state.finished()) {
            temp_state.take_random_action();
          }
          auto v = temp_state.score(g.last_player);
          val += v;
        }
        val /= random_rollouts;
        g.record(g.node_id, val);*/
      } else {
        g.record(g.node_id, std::min(1.0, std::max(-1.0, double(g.rollout_state.score(g.last_player)))));
      }
    }
  }

  std::vector<uint64_t> result;
  result.resize(game_slots.size());
  for (size_t i = 0; i < game_slots.size(); ++i) {
    auto &g = game_slots[i];
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    result[i] = g.get_move_index(explore_for_n_moves);
  }
  return result;
}

// plays one move in all active games in all slots
template <typename State>
void single_move(std::vector<GameSlot<State>> &game_slots, int32_t rollouts,
                 double temp, int32_t *boards_buffer, float *probs_buffer,
                 float *scores_buffer, int32_t *log_boards_buffer,
                 float *log_probs_buffer, EvalFn eval_cb, LogFn log_freq_cb,
                 GameDoneFn log_game_done_cb, uint32_t model_id,
                 uint32_t explore_for_n_moves, uint32_t random_rollouts) {
  PYLOG << "Single move with model " << model_id;
  
  ModelEvaluator evaluator{.boards_buffer = boards_buffer,
                           .probs_buffer = probs_buffer,
                           .scores_buffer = scores_buffer,
                           .run = eval_cb};
  auto picked_moves =
      get_moves<State>(game_slots, rollouts, temp, evaluator, model_id,
                       explore_for_n_moves, random_rollouts);
  // now pick and apply moves
  for (size_t i = 0; i < game_slots.size(); ++i) {
    auto &g = game_slots[i];
    if (!g.slot_active || g.state.finished()) {
      continue;
    }

    g.log_training_data(log_freq_cb, log_boards_buffer, log_probs_buffer);

    g.state.apply_move(picked_moves[i]);

    if (g.state.finished()) {
      if (log_game_done_cb != nullptr) {
        // do we need to play another game in this slot?
        g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
      }
    }
  }
}
