#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "games/mnk_state.h"
#include "mcts_inv.h"

std::random_device batch_mcts_rd;

template<typename State>
struct GameSlot {
  GameSlot() : gen{batch_mcts_rd()} {}

  void restart() { state = State(); }

  uint64_t get_move_index(uint32_t explore_for_n_moves) {
    static thread_local std::array<uint64_t, State::M * State::N> visits;
    std::fill(visits.begin(), visits.end(), 0ull);

    for (int i = 0; i < mcts.nodes[mcts.root_id].children_count; i++) {
      const auto &node = mcts.nodes[mcts.nodes[mcts.root_id].children_from + i];
      visits[node.in_action] = node.N;
    }

    if (state.stones_played() >= explore_for_n_moves) {
      // pick greedily
      auto it = std::max_element(visits.begin(), visits.end());
      return std::distance(visits.begin(), it);
    } else {
      // sample
      std::discrete_distribution<> d(visits.begin(), visits.end());
      return d(gen);
    }
  }

  void fill_visits(float *freq_buffer) {
    for (int i = 0; i < State::N * State::M; i++) {
      freq_buffer[i] = 0.0f;
    }
    for (int i = 0; i < mcts.nodes[mcts.root_id].children_count; i++) {
      const auto &node = mcts.nodes[mcts.nodes[mcts.root_id].children_from + i];
      freq_buffer[node.in_action] = node.N;
    }
  }

  void make_move(void (*log_freq_cb)(int32_t),
                 bool (*log_game_done_cb)(int32_t), int32_t *log_boards_buffer,
                 float *log_probs_buffer, uint32_t explore_for_n_moves,
                 int32_t model_id) {
    // logging training data here
    if (log_freq_cb != nullptr) {
      state.fill_boards(log_boards_buffer);
      fill_visits(log_probs_buffer);
      log_freq_cb(model_id);
    }

    // applying move
    state.apply_move(get_move_index(explore_for_n_moves));
    total_moves++;
    
    if (state.finished()) {
      if (log_game_done_cb != nullptr) {
        // do we need to play another game in this slot?
        slot_active = log_game_done_cb(state.winner);
      }
    }
  }

  // global scope
  std::mt19937 gen;
  bool slot_active = true;
  uint64_t total_moves = 0ull;

  // per game instance scope
  State state;

  // per each move scope
  MCTS mcts;

  // within current search state scope
  State rollout_state;
  int node_id;
  int last_player;
};


// plays one move in all active games in all slots
template<typename State>
void single_move(std::vector<GameSlot<State>> &game_slots, int32_t rollouts, double temp,
                 int32_t *boards_buffer, float *probs_buffer,
                 int32_t *log_boards_buffer, float *log_probs_buffer,
                 void (*eval_cb)(int32_t), void (*log_freq_cb)(int32_t),
                 bool (*log_game_done_cb)(int32_t), uint32_t model_id,
                 uint32_t explore_for_n_moves) {
  for (auto &g : game_slots) {
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    g.mcts.reset(rollouts * State::M * State::N);
  }
  
  static const int kBoardElements = 2 * State::M * State::N;
  static const int kProbElements = State::M * State::N;
  for (int32_t r = 0; r < rollouts; r++) {
    for (auto &g : game_slots) {
      if (!g.slot_active || g.state.finished()) {
        continue;
      }
      g.rollout_state = g.state;
      g.node_id = g.mcts.root_id;
      while (g.mcts.nodes[g.node_id].children_count > 0) {
        g.node_id = g.mcts.select(g.mcts.nodes[g.node_id], temp);
        g.rollout_state.apply_move(g.mcts.nodes[g.node_id].in_action);
      }

      g.last_player = 1 - g.rollout_state.player;
    }

    // now expand all games where !rollout_state.finished
    for (size_t i = 0; i < game_slots.size(); ++i) {
      auto &g = game_slots[i];
      // slot_active -- new game can be started in that slot
      // finished -- this rollout is over, no need to evaluate
      if (!g.slot_active || g.rollout_state.finished() || g.state.finished()) {
        continue;
      }
      g.mcts.nodes[g.node_id].children_from = g.mcts.size;
      if (eval_cb != nullptr) {
        g.rollout_state.fill_boards(boards_buffer + i * kBoardElements);
      }
    }
    // eval_cb takes boards_buffer as an input and writes results to
    // probs_buffer.
    // Where do we pass value? 
    if (eval_cb != nullptr && model_id != 0) {
      eval_cb(model_id);
    }

    for (size_t i = 0; i < game_slots.size(); ++i) {
      auto &g = game_slots[i];
      if (!g.slot_active || g.rollout_state.finished() || g.state.finished()) {
        continue;
      }
      uint64_t moves = g.rollout_state.valid_actions();
      int j = g.mcts.size;
      for (uint64_t k = 0; k < State::M * State::N; k++) {
        if ((1ull << k) & moves) {
          g.mcts.nodes[j] = MCTSNode(model_id > 0 ? probs_buffer[i * kProbElements + k] : 1.0f);
          g.mcts.nodes[j].parent = g.node_id;
          g.mcts.nodes[j].in_action = k;
          j++;
        }
      }
      g.mcts.nodes[g.node_id].children_count = j - g.mcts.size;
      g.mcts.size = j;

      // TODO: get value from the model here
      while (!g.rollout_state.finished()) {
        g.rollout_state.take_random_action();
      }
    }

    for (auto &g : game_slots) {
      if (g.slot_active && !g.state.finished()) {
        g.mcts.record(g.node_id, g.rollout_state.score(g.last_player));
      }
    }
  }
  // now pick and apply moves
  for (auto &g : game_slots) {
    if (!g.slot_active || g.state.finished()) {
      continue;
    }

    g.make_move(log_freq_cb, log_game_done_cb, log_boards_buffer,
                log_probs_buffer, explore_for_n_moves, model_id);
  }
}

extern "C" {

// we add something like 'game name' here and dispatch to the 
// right implementation
void batch_mcts(uint32_t batch_size, int32_t *boards_buffer,
                float *probs_buffer, int32_t *log_boards_buffer,
                float *log_probs_buffer, void (*eval_cb)(int32_t),
                void (*log_freq_cb)(int32_t), bool (*log_game_done_cb)(int32_t),
                int32_t model_a, int32_t model_b, uint32_t explore_for_n_moves,
                uint32_t a_rollouts, double a_temp, uint32_t b_rollouts,
                double b_temp) {
  using State = MNKState<8, 8, 5>;
  
  std::vector<GameSlot<State>> games{batch_size};
  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;
    // first player
    single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer,
                log_boards_buffer, log_probs_buffer, eval_cb, log_freq_cb,
                log_game_done_cb, model_a, explore_for_n_moves);

    // second player
    single_move<State>(games, b_rollouts, b_temp, boards_buffer, probs_buffer,
                log_boards_buffer, log_probs_buffer, eval_cb, log_freq_cb,
                log_game_done_cb, model_b, explore_for_n_moves);

    // if a game is finished, but we need to play more games, restart the game
    // and reuse the slot
    for (auto &g : games) {
      if (g.slot_active) {
        has_active_games = true;
        if (g.state.finished()) {
          g.restart();
        }
      }
    }
  }
}
}
