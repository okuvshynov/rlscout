#include <bit>
#include <iostream>
#include <random>

#include "alpha_beta/alpha_beta_runtime.h"
#include "othello/othello_state.h"
#include "othello/othello_move_selection_policy.h"

using State = OthelloState<6>;
using score_t = int8_t;

int main() {
  Player* players[2] = {new RandomPlayer{}, new RandomABPlayer{25, -5, 5}};

  for (auto g = 0; g < 10000; g++) {
    OthelloState<6> state;
    while (!state.finished()) {
      auto move = players[state.player]->get_move(state);
      if (move == -1) {
        state.apply_skip();
      } else {
        state.apply_move(move);
      }
    }

    std::cout << state.score(g % 2) << std::endl;
    std::swap(players[0], players[1]);
  }

  return 0;
}