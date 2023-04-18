#include <bit>
#include <iostream>
#include <random>

#include "alpha_beta/alpha_beta_runtime.h"
#include "othello/othello_state.h"

using State = OthelloState<6>;
using score_t = int8_t;

struct Player {
  // negative number means skip
  virtual int64_t get_move(const State& state) = 0;
};

struct RandomPlayer : public Player {
  virtual int64_t get_move(const State& state) {
    auto actions = state.valid_actions();
    if (actions == 0ull) {
      return -1ll;
    }

    auto action_size = std::popcount(actions);
    auto dis = std::uniform_int_distribution<int64_t>(0, action_size - 1);
    auto selected = dis(gen_);
    int set_bit_count = 0;
    for (int i = 0; i < 64; i++) {
      if (actions & (1ULL << i)) {
        if (set_bit_count == selected) {
          return i;
        }
        set_bit_count++;
      }
    }
    return -1;
  }
  int random_depth_;
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
};

struct RandomABPlayer : public Player {
  RandomABPlayer(int random_depth) : random_depth_(random_depth) {}

  int64_t pick_random_move(const State& state) {
    auto actions = state.valid_actions();
    if (actions == 0ull) {
      return -1ll;
    }

    auto action_size = std::popcount(actions);
    auto dis = std::uniform_int_distribution<int64_t>(0, action_size - 1);
    auto selected = dis(gen_);
    int set_bit_count = 0;
    for (int i = 0; i < 64; i++) {
      if (actions & (1ULL << i)) {
        if (set_bit_count == selected) {
          return i;
        }
        set_bit_count++;
      }
    }
    return -1;
  }

  virtual int64_t get_move(const State& state) {
    if (state.stones_played() < random_depth_) {
      return pick_random_move(state);
    }
    auto moves = ab_.get_move_scores(state, -10, 10);
    //std::cout << int(state.player) << " ab moves " << moves.size() << std::endl;
    if (moves.empty()) {
      return -1;
    }
    /*for (auto m : moves) {
      std::cout << m.first << " " << m.second << std::endl;
    }*/
    auto move =
        std::max_element(moves.begin(), moves.end(), [&state](auto a, auto b) {
          return state.player == 0 ? a.second < b.second : a.second > b.second;
        });

    return __builtin_ffsll(move->first) - 1;
  }
  int random_depth_;
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
  AlphaBetaRuntime<State, score_t> ab_;
};

int main() {
  Player* players[2] = {new RandomPlayer{}, new RandomABPlayer{20}};

  for (auto g = 0; g < 10000; g++) {
    OthelloState<6> state;
    while (!state.finished()) {
      auto move = players[state.player]->get_move(state);
      if (move == -1) {
        state.apply_skip();
      } else {
        state.apply_move(move);
      }
      //state.p();
      // state.to_canonical().p();
      //std::cout << std::endl;
    }

    std::cout << state.score(0) << std::endl;
  }

  return 0;
}