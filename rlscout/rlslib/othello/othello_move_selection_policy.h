#pragma once

#include <bit>
#include <iostream>
#include <random>

#include "alpha_beta/alpha_beta_runtime.h"
#include "othello/othello_state.h"

using State = OthelloState<6>;
using score_t = int8_t;

struct MoveSelectionPolicy {
  virtual int64_t get_move(const State& state) = 0;
};

struct RandomSelectionPolicy {
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
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
};

struct ABSelectionPolicy {
  ABSelectionPolicy(score_t alpha, score_t beta) : alpha_(alpha), beta_(beta) {}

  virtual int64_t get_move(const State& state) {
    auto moves = ab_.get_move_scores(state, -10, 10);
    if (moves.empty()) {
      return -1;
    }
    auto move =
        std::max_element(moves.begin(), moves.end(), [&state](auto a, auto b) {
          return state.player == 0 ? a.second < b.second : a.second > b.second;
        });
    return __builtin_ffsll(move->first) - 1;
  }

  AlphaBetaRuntime<State, score_t> ab_;
  score_t alpha_, beta_;
};

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
  RandomABPlayer(uint32_t random_depth, score_t alpha, score_t beta)
      : ab_policy_(alpha, beta), random_depth_(random_depth) {}
  virtual int64_t get_move(const State& state) {
    return state.stones_played() < random_depth_
               ? random_policy_.get_move(state)
               : ab_policy_.get_move(state);
  }

  void load_tt(const std::string& filename) {
    ab_policy_.ab_.load_tt(filename);
  }

  void save_tt(const std::string& filename) {
    ab_policy_.ab_.save_tt(filename);
  }

  RandomSelectionPolicy random_policy_;
  ABSelectionPolicy ab_policy_;
  uint32_t random_depth_;
};