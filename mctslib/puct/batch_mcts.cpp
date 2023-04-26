#include <cstdint>

#include "othello/othello_state.h"
#include "puct/batch_puct.h"

extern "C" {

void init_py_logger(PyLogFn log_fn) {
    PyLog::instance().initialize(log_fn);
}

void batch_mcts(uint32_t batch_size, int32_t *boards_buffer,
                float *probs_buffer, float* scores_buffer, int32_t *log_boards_buffer,
                float *log_probs_buffer, EvalFn eval_cb,
                LogFn log_freq_cb, GameDoneFn log_game_done_cb,
                int32_t model_a, int32_t model_b, uint32_t explore_for_n_moves,
                uint32_t a_rollouts, double a_temp, uint32_t b_rollouts,
                double b_temp, uint32_t a_rr, uint32_t b_rr) {
  using State = OthelloState<6>;
  
  std::vector<GameSlot<State>> games{batch_size};
  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;
    //std::cout << model_a << " " << model_b << std::endl;
    // first player
    single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer, scores_buffer,
                log_boards_buffer, log_probs_buffer, eval_cb, log_freq_cb,
                log_game_done_cb, model_a, explore_for_n_moves, a_rr);

    // second player
    single_move<State>(games, b_rollouts, b_temp, boards_buffer, probs_buffer, scores_buffer,
                log_boards_buffer, log_probs_buffer, eval_cb, log_freq_cb,
                log_game_done_cb, model_b, explore_for_n_moves, b_rr);

    //games[0].state.p();

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
    //std::cout << "puct cycles: " << puct_cycles << ", " << puct_wasted_cycles << std::endl;
  }
}
}
