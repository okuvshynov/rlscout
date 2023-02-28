#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "othello_dumb7.h"

template <uint8_t n>
struct OthelloState {
  using Self = OthelloState<n>;
  static_assert(n == 6 || n == 8, "6x6 or 8x8 board only");

  static constexpr auto M = n;
  static constexpr auto N = n;

  uint64_t board[2] = {0b000000000000001000000100000000000000ull,
                       0b000000000000000100001000000000000000ull};

  int32_t player = 0;   // 0 or 1
  int32_t winner = -1;  // -1 -- draw or not finished
  int32_t skipped = 0;  // if it becomes 2 the game is over

  uint64_t mask(uint64_t index) const { return (1ull << index); }

  uint64_t index(uint64_t x, uint64_t y) const { return (x * n + y); }

  bool operator==(const Self& other) const {
    return board[0] == other.board[0] && board[1] == other.board[1] &&
           player == other.player && winner == other.winner &&
           skipped == other.skipped;
  }

  uint64_t hash() const {
    return board[0] & (board[1] << 28);
  }

  void apply_skip() {
    skipped++;
    if (skipped == 2) {
      auto a = std::popcount(board[player]);
      auto b = std::popcount(board[1 - player]);
      if (a > b) {
        winner = player;
      }
      if (a < b) {
        winner = 1 - player;
      }
    }
    player = 1 - player;
  }

  bool apply_move(uint64_t index) {
    auto m = mask(index);

    // TODO: remove valid_actions call
    if (!(m & valid_actions())) {
      apply_skip();
      return false;
    }

    skipped = 0;

    auto to_flip = OthelloDumb7Fill6x6::to_flip(mask(index), board[player],
                                                board[1 - player]);

    board[player] |= m;
    board[player] |= to_flip;
    board[1 - player] ^= to_flip;

    player = 1 - player;
    return true;
  }

  bool apply_move_no_check(uint64_t index) {
    auto m = mask(index);

    skipped = 0;

    auto to_flip = OthelloDumb7Fill6x6::to_flip(m, board[player],
                                                board[1 - player]);

    board[player] |= m;
    board[player] |= to_flip;
    board[1 - player] ^= to_flip;

    player = 1 - player;
    return true;
  }

  bool finished() const { return skipped >= 2; }

  void fill_boards(int32_t* boards) const {
    for (uint64_t i = 0; i < n * n; i++) {
      boards[i] = boards[i + n * n] = 0;
      if (board[player] & mask(i)) {
        boards[i] = 1;
      }
      if (board[1 - player] & mask(i)) {
        boards[i + n * n] = 1;
      }
    }
  }

  uint64_t valid_actions() const {
    return OthelloDumb7Fill6x6::valid_moves(board[player], board[1 - player]);
  }

  // TODO: remove after we have value model
  void take_random_action() {
    auto actions = valid_actions();
    if (actions == 0ull) {
      apply_skip();
      return;
    }

    auto actions_size = std::popcount(actions);

    auto bit = rand() % actions_size;

    while (true) {
      uint64_t index = rand() % (n * n);
      if (mask(index) & actions) {
        apply_move(index);
        return;
      }
    }
  }

  double score(int32_t player) const {
    return std::popcount(board[player]) - std::popcount(board[1 - player]);
  }

  // includes 4 initial stones on the board
  uint32_t stones_played() const { return std::popcount(board[0] | board[1]); }

  // DEBUG printing
  int get(uint64_t x, uint64_t y) const {
    uint64_t idx = index(x, y);
    if (board[0] & mask(idx)) {
      return 0;
    }
    if (board[1] & mask(idx)) {
      return 1;
    }
    return -1;
  }

  void p() const {
    for (uint64_t y = 0; y < n; y++) {
      for (uint64_t x = 0; x < n; x++) {
        int cell = get(x, y);
        if (cell == -1) {
          printf(".");
        } else {
          printf("%c", cell == 0 ? '0' : 'X');
        }
      }
      printf("\n");
    }
  }
};
