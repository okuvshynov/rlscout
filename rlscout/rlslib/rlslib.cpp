#include <cstdint>

#include "othello/othello_move_selection_policy.h"
#include "othello/othello_state.h"
#include "puct/batch_puct.h"
#include "utils/model_evaluator.h"
#include "utils/random.h"

template <typename State>
void process_mcts(std::vector<GameSlot<State>> &games, uint32_t rollouts,
                  double temp, int32_t *boards_buffer, float *probs_buffer,
                  float *scores_buffer, EvalFn eval_cb,
                  GameDoneFn log_game_done_cb, int32_t model_id,
                  uint32_t explore_for_n_moves, uint32_t random_rollouts) {
  ModelEvaluator evaluator{.boards_buffer = boards_buffer,
                           .probs_buffer = probs_buffer,
                           .scores_buffer = scores_buffer,
                           .run = eval_cb};
  auto picked_moves =
      get_moves<State>(games, rollouts, temp, evaluator, model_id,
                       explore_for_n_moves, random_rollouts);
  for (size_t i = 0; i < games.size(); ++i) {
    auto &g = games[i];
    if (!g.slot_active || g.state.finished()) {
      continue;
    }
    g.state.apply_move(picked_moves[i]);
    if (g.state.finished()) {
      g.slot_active = log_game_done_cb(g.state.score(0), g.game_id);
    }
    // g.state.p();
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
    // g.state.p();
  }
}

extern "C" {

void init_py_logger(PyLogFn log_fn) { PyLog::instance().initialize(log_fn); }

void init_random_seed(int32_t seed) { RandomGen::init(seed); }

void batch_mcts(uint32_t batch_size, int32_t *boards_buffer,
                float *probs_buffer, float *scores_buffer,
                int32_t *log_boards_buffer, float *log_probs_buffer,
                EvalFn eval_cb, LogFn log_freq_cb, GameDoneFn log_game_done_cb,
                int32_t model_a, int32_t model_b, uint32_t explore_for_n_moves,
                uint32_t a_rollouts, double a_temp, uint32_t b_rollouts,
                double b_temp, uint32_t a_rr, uint32_t b_rr) {
  using State = OthelloState<6>;

  std::vector<GameSlot<State>> games{batch_size};
  bool has_active_games = true;
  while (has_active_games) {
    has_active_games = false;
    single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer,
                       scores_buffer, log_boards_buffer, log_probs_buffer,
                       eval_cb, log_freq_cb, log_game_done_cb, model_a,
                       explore_for_n_moves, a_rr);

    PyLog::INFO("First player move done for batch");

    // second player
    single_move<State>(games, b_rollouts, b_temp, boards_buffer, probs_buffer,
                       scores_buffer, log_boards_buffer, log_probs_buffer,
                       eval_cb, log_freq_cb, log_game_done_cb, model_b,
                       explore_for_n_moves, b_rr);
    PyLog::INFO("First player move done for batch");

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
  PyLog::INFO("loading transposition table");
  random_ab_player.ab_policy_.ab_.load_tt("./db/6x6.tt");
  PyLog::INFO("loading transposition table done");
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

  PyLog::INFO("saving transposition table");
  random_ab_player.ab_policy_.ab_.save_tt("./db/6x6.tt");
  PyLog::INFO("saving transposition table done");
  // random_ab_player.ab_policy_.ab_.print_tt_stats();
}

int8_t run_ab(int32_t *boards_buffer, float *probs_buffer, EvalFn eval_cb,
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
  return AB.alpha_beta<4, true>(s.to_canonical(), alpha, beta);
}
}
