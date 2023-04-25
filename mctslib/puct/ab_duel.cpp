#include <cstdint>

#include "othello/othello_move_selection_policy.h"
#include "othello/othello_state.h"
#include "puct/batch_puct.h"

template <typename State>
void process_mcts(std::vector<GameSlot<State>> &games, uint32_t rollouts,
                  double temp, int32_t *boards_buffer, float *probs_buffer,
                  float *scores_buffer, EvalFn eval_cb,
                  GameDoneFn log_game_done_cb, int32_t model_id,
                  uint32_t explore_for_n_moves, uint32_t random_rollouts) {
  auto picked_moves =
      get_moves<State>(games, rollouts, temp, boards_buffer, probs_buffer,
                       scores_buffer, eval_cb, model_id, explore_for_n_moves, random_rollouts);
  for (size_t i = 0; i < games.size(); ++i) {
    auto &g = games[i];
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    g.state.apply_move(picked_moves[i]);
    if (g.state.finished()) {
      g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
    }
    //g.state.p();
  }
}

void process_rand_ab(std::vector<GameSlot<State>> &games,
                     RandomABPlayer &player, GameDoneFn log_game_done_cb) {
  for (auto &g : games) {
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    auto move = player.get_move(g.state);
    if (move >= 0) {
      g.state.apply_move(move);
    } else {
      g.state.apply_skip();
    }
    if (g.state.finished()) {
      g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
    }
    //g.state.p();
  }
}

extern "C" {

// runs duel between MCTS+Model and Random + FullAB;
// Can be used as a baseline for the model we train.
void ab_duel(uint32_t batch_size,
             // IO for model prediction
             int32_t *boards_buffer, float *probs_buffer, float *scores_buffer,
             EvalFn eval_cb, GameDoneFn log_game_done_cb, int32_t model_id,
             uint32_t explore_for_n_moves, uint32_t rollouts, double temp,

             // config for AB search
             int8_t alpha, int8_t beta, uint32_t full_after_n_moves,

             bool inverse_first_player, uint32_t random_rollouts) {
  using State = OthelloState<6>;
  RandomABPlayer random_ab_player(full_after_n_moves, alpha, beta);
  random_ab_player.ab_policy_.ab_.load_tt("./db/6x6.tt");
  std::vector<GameSlot<State>> games{batch_size};

  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;

    // for now assume MCTS is first player
    if (inverse_first_player) {
      process_rand_ab(games, random_ab_player, log_game_done_cb);
      process_mcts<State>(games, rollouts, temp, boards_buffer, probs_buffer,
                          scores_buffer, eval_cb, log_game_done_cb, model_id,
                          explore_for_n_moves, random_rollouts);

    } else {
      process_mcts<State>(games, rollouts, temp, boards_buffer, probs_buffer,
                          scores_buffer, eval_cb, log_game_done_cb, model_id,
                          explore_for_n_moves, random_rollouts);
      process_rand_ab(games, random_ab_player, log_game_done_cb);
    }

    // now do second player

    // restart finished games in active slots
    for (auto &g : games) {
      if (g.slot_active) {
        has_active_games = true;
      }
      if (g.slot_active && g.state.finished()) {
        g.restart();
      }
    }
  }

  random_ab_player.ab_policy_.ab_.save_tt("./db/6x6.tt");
  random_ab_player.ab_policy_.ab_.print_tt_stats();
}
}