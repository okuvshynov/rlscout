#include <cstdint>

#include "othello/othello_move_selection_policy.h"
#include "othello/othello_state.h"
#include "puct/batch_puct.h"

extern "C" {

// runs duel between MCTS+Model and Random + FullAB;
// Can be used as a baseline for the model we train.
void ab_duel(uint32_t batch_size,
             // IO for model prediction
             int32_t *boards_buffer, float *probs_buffer, float *scores_buffer,
             EvalFn eval_cb, GameDoneFn log_game_done_cb, int32_t model_id,
             uint32_t explore_for_n_moves, uint32_t rollouts, double temp,

             // config for AB search
             int8_t alpha, int8_t beta, uint32_t full_after_n_moves

) {
  using State = OthelloState<6>;
  RandomABPlayer random_ab_player(full_after_n_moves, alpha, beta);
  std::vector<GameSlot<State>> games{batch_size};

  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;

    // for now assume MCTS is first player
    auto picked_moves =
        get_moves<State>(games, rollouts, temp, boards_buffer, probs_buffer,
                         scores_buffer, eval_cb, model_id, explore_for_n_moves);
    for (size_t i = 0; i < games.size(); ++i) {
        auto &g = games[i];
        if (!g.slot_active || g.state.finished()) {
            continue;
        }
        g.state.apply_move(picked_moves[i]);
        if (g.state.finished()) {
            g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
        }
    }

    // now do second player
    for (auto &g : games) {
        if (!g.slot_active || g.state.finished()) {
            continue;
        }
          auto move = random_ab_player.get_move(g.state);
          if (move >= 0) {
            g.state.apply_move(move);
          } else {
            g.state.apply_skip();
          }
        if (g.state.finished()) {
            g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
        }
    }

    // restart finished games in active slots
    for (auto& g: games) {
        if (g.slot_active) {
            has_active_games = true;
        }
        if (g.slot_active && g.state.finished()) {
            g.restart();
        }
    } 
  }
}
}