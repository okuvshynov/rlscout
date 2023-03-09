#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "othello_dumb7.h"


uint64_t flip_v_6x6(uint64_t v) {
  return
  ((v & 0b111111000000000000000000000000000000ull) >> 30) |
  ((v & 0b111111000000000000000000000000ull) >> 18) |
  ((v & 0b111111000000000000000000ull) >> 6) |
  ((v & 0b111111000000000000ull) << 6) |
  ((v & 0b111111000000ull) << 18) |
  ((v & 0b111111ull) << 30);
}




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

  // Hash128to64 function from Google's cityhash (under the MIT License).
  uint64_t hash() const {
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    uint64_t a = (board[0] ^ board[1]) * kMul;
    a ^= (a >> 47);
    uint64_t b = (board[1] ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
  }

  bool empty() const {
    return board[0] == 0ull && board[1] == 0ull;
  }

  // just to test
  uint64_t flip_diag_6x6_slow(uint64_t v) const {
    uint64_t res = 0ull;
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        if (v & mask(index(i, j))) {
          res |= mask(index(j, i));
        }
      }
    }
    return res;
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

  void dflip() {
    board[0] = flip_diag_6x6_slow(board[0]);
    board[1] = flip_diag_6x6_slow(board[1]);
  }

  // TODO: slow 
  void vflip() {
    board[0] = flip_v_6x6(board[0]);
    board[1] = flip_v_6x6(board[1]);
  }

  bool operator<(const Self& other) {
    return (board[0] | board[1]) < (other.board[0] | other.board[1]);
  }

  // TODO: this is too slow
  Self to_canonical() const {
    Self res = *this;
    Self curr = res;
    curr.vflip();
    if (curr < res) {
      res = curr;
    }
    curr.dflip();
    if (curr < res) {
      res = curr;
    }
    curr.vflip();
    if (curr < res) {
      res = curr;
    }
    curr.dflip();
    if (curr < res) {
      res = curr;
    }
    curr.vflip();
    if (curr < res) {
      res = curr;
    }
    curr.dflip();
    if (curr < res) {
      res = curr;
    }
    curr.vflip();
    if (curr < res) {
      res = curr;
    }

    return res;
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

  int32_t score(int32_t player) const {
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
