#include <bit>
#include <iostream>
#include <random>

#include "alpha_beta/alpha_beta_runtime.h"
#include "othello/othello_state.h"
#include "othello/othello_move_selection_policy.h"

using State = OthelloState<6>;
using score_t = int8_t;

struct Player {
  // negative number means skip
  virtual int64_t get_move(const State& state) = 0;
};

struct RandomPlayer : public Player {
  virtual int64_t get_move(const State& state) {
    return random_policy.get_move(state);
  }
    RandomSelectionPolicy random_policy;
};

struct RandomABPlayer : public Player {
  RandomABPlayer(int random_depth, score_t alpha, score_t beta) : ab_policy_(alpha, beta), random_depth_(random_depth) {
    
  }
  virtual int64_t get_move(const State& state) {
    return state.stones_played() < random_depth_ ? random_policy_.get_move(state) : ab_policy_.get_move(state);
  }

    RandomSelectionPolicy random_policy_;
    ABSelectionPolicy ab_policy_;
    int random_depth_;
};

int main() {
  Player* players[2] = {new RandomPlayer{}, new RandomABPlayer{25, -10, 10}};

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