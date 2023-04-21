#include <cstdint>

#include "othello/othello_move_selection_policy.h"
#include "othello/othello_state.h"
#include "puct/batch_puct.h"

extern "C" {

void batch_duel(uint32_t batch_size, int32_t *boards_buffer,
                float *probs_buffer, float *scores_buffer,
                int32_t *log_boards_buffer, float *log_probs_buffer,
                EvalFn eval_cb, LogFn log_freq_cb,
                bool (*log_game_done_cb)(int32_t, int64_t), int32_t model_a,
                int32_t model_b, uint32_t explore_for_n_moves,
                uint32_t a_rollouts, double a_temp, uint32_t b_rollouts,
                double b_temp) {
  using State = OthelloState<6>;
  RandomABPlayer random_ab_player(20, -5, 5);
  int64_t score = 0;

  for (int rep = 0; rep < 10; rep ++) {
  {
    std::vector<GameSlot<State>> games{batch_size};
    bool has_active_games = true;
    while (has_active_games) {
      has_active_games = false;
      // first player
      single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer,
                         scores_buffer, log_boards_buffer, log_probs_buffer,
                         eval_cb, log_freq_cb, log_game_done_cb, model_a,
                         explore_for_n_moves);
      // second player
      for (auto &g : games) {
        if (g.slot_active && !g.state.finished()) {
          auto move = random_ab_player.get_move(g.state);
          if (move >= 0) {
            g.state.apply_move(move);
          } else {
            g.state.apply_skip();
          }
        }

        if (g.slot_active && g.state.finished()) {
          g.slot_active = false;
        }
      }
      for (auto &g : games) {
        if (g.slot_active) {
          has_active_games = true;
          if (g.state.finished()) {
            g.restart();
          }
        }
      }
      std::cout << ".";
      fflush(stdout);
    }
    for (auto &g : games) {
      std::cout << g.state.score(0) << std::endl;
      score += g.state.score(0);
    }
  }

  {
    std::vector<GameSlot<State>> games{batch_size};
    bool has_active_games = true;
    while (has_active_games) {
      has_active_games = false;
      // second player
      for (auto &g : games) {
        if (g.slot_active && !g.state.finished()) {
          auto move = random_ab_player.get_move(g.state);
          if (move >= 0) {
            g.state.apply_move(move);
          } else {
            g.state.apply_skip();
          }
        }

        if (g.slot_active && g.state.finished()) {
          g.slot_active = false;
        }
      }

      // first player
      single_move<State>(games, a_rollouts, a_temp, boards_buffer, probs_buffer,
                         scores_buffer, log_boards_buffer, log_probs_buffer,
                         eval_cb, log_freq_cb, log_game_done_cb, model_a,
                         explore_for_n_moves);

      for (auto &g : games) {
        if (g.slot_active) {
          has_active_games = true;
          if (g.state.finished()) {
            g.restart();
          }
        }
      }
      // std::cout << "puct cycles: " << puct_cycles << ", " <<
      // puct_wasted_cycles << std::endl;
      std::cout << ".";
      fflush(stdout);
    }

    for (auto &g : games) {
      std::cout << g.state.score(1) << std::endl;
      score += g.state.score(0);
    }
  }

  std::cout << std::endl << "### " << score << std::endl;
  }

  std::cout << std::endl << "### " << score << std::endl;
}
}
