#include <cstdint>

#include "othello/othello_state.h"
#include "puct/batch_puct.h"
#include "utils/model_evaluator.h"
#include "utils/random.h"
#include "alpha_beta/alpha_beta.h"

extern "C" {

void init_py_logger(PyLogFn log_fn) { PyLog::instance().initialize(log_fn); }

void init_random_seed(int32_t seed) { RandomGen::init(seed); }

void batch_mcts(uint32_t batch_size, int32_t *boards_buffer,
                float *probs_buffer, float *scores_buffer,
                int32_t *log_boards_buffer, float *log_probs_buffer,
                EvalFn eval_cb, LogFn log_freq_cb, GameDoneFn log_game_done_cb,
                ModelIdFn model_a_cb, ModelIdFn model_b_cb, uint32_t explore_for_n_moves,
                uint32_t a_rollouts, double a_temp, uint32_t b_rollouts,
                double b_temp, uint32_t a_rr, uint32_t b_rr) {
  using State = OthelloState<6>;

  std::vector<GameSlot<State>> games{batch_size};
  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;
    single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer,
                       scores_buffer, log_boards_buffer, log_probs_buffer,
                       eval_cb, log_freq_cb, log_game_done_cb, model_a_cb(),
                       explore_for_n_moves, a_rr);

    PyLog::INFO("First player move done for batch");

    // second player
    single_move<State>(games, b_rollouts, b_temp, boards_buffer, probs_buffer,
                       scores_buffer, log_boards_buffer, log_probs_buffer,
                       eval_cb, log_freq_cb, log_game_done_cb, model_b_cb(),
                       explore_for_n_moves, b_rr);
    PyLog::INFO("Second player move done for batch");

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

uint64_t run_ab(int32_t *boards_buffer, float *probs_buffer, EvalFn eval_cb,
              int8_t alpha, int8_t beta) {
  PyLog::INFO("starting alpha-beta search");
  using State = OthelloState<6>;
  auto AB = AlphaBeta<State, int8_t>();
  State s;
  ModelEvaluator evaluator{.boards_buffer = boards_buffer,
                           .probs_buffer = probs_buffer,
                           .scores_buffer = nullptr,
                           .run = eval_cb};
  AB.set_model_evaluator(&evaluator);
  auto result = AB.alpha_beta<4, true>(s.to_canonical(), alpha, beta);
  PYLOG << "result = " << int32_t(result);
  return AB.total_completions();
}
}
