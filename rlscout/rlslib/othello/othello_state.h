#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

#include "othello_6x6_dumb7.h"
#include "othello_6x6_board_ops.h"

template <uint8_t n>
struct OthelloState {
  using Self = OthelloState<n>;
  static_assert(n == 6 || n == 8, "6x6 or 8x8 board only");

  static constexpr auto M = n;
  static constexpr auto N = n;

  uint64_t board[2] = {0b000000000000001000000100000000000000ull,
                       0b000000000000000100001000000000000000ull};

  static const uint64_t k6x6Full = 0b111111111111111111111111111111111111ull;

  int8_t player = 0;  // 0 or 1
  // int32_t winner = -1;  // -1 -- draw or not finished
  int8_t skipped = 0;  // if it becomes 2 the game is over

  uint64_t mask(uint64_t index) const { return (1ull << index); }

  uint64_t index(uint64_t x, uint64_t y) const { return (x * n + y); }

  bool operator==(const Self& other) const {
    return board[0] == other.board[0] && board[1] == other.board[1] &&
           player == other.player &&  // winner == other.winner &&
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

  bool empty() const { return board[0] == 0ull && board[1] == 0ull; }

  void apply_skip() {
    skipped++;
    player = 1 - player;
  }

  bool apply_move(uint64_t index) {
    auto m = mask(index);

    if (!(m & valid_actions())) {
      apply_skip();
      return false;
    }

    return apply_move_mask(m);
  }

  bool apply_move_mask(uint64_t m) {
    skipped = 0;

    auto to_flip =
        OthelloDumb7Fill6x6::to_flip(m, board[player], board[1 - player]);

    board[player] |= m;
    board[player] |= to_flip;
    board[1 - player] ^= to_flip;

    player = 1 - player;
    return true;
  }

  bool finished() const { return skipped >= 2 || full(); }

  bool full() const { return (board[0] | board[1]) == k6x6Full; }

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
    board[0] = flip_diag_6x6(board[0]);
    board[1] = flip_diag_6x6(board[1]);
  }

  void vflip() {
    board[0] = flip_v_6x6(board[0]);
    board[1] = flip_v_6x6(board[1]);
  }

  bool operator<(const Self& other) const {
    return (board[0] | board[1]) < (other.board[0] | other.board[1]);
  }
  
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

  Self maybe_to_canonical(const Self& old, uint64_t move) {
    uint64_t old_board = old.board[0] | old.board[1];
    if (move > old_board) {
      return to_canonical();
    }
    return *this;
  }

  uint64_t get_random_action() {
    auto actions = valid_actions();
    if (actions == 0ull) {
      return 0ull;
    }
    static std::mt19937 gen{42};
    std::uniform_int_distribution<int> dis(0, std::popcount(actions) - 1);
    uint64_t shifts = dis(gen);

    while (shifts--) {
      actions = (actions & (actions - 1));
    }

    uint64_t other_actions = (actions & (actions - 1));
    return other_actions ^ actions;
  }

  void take_random_action() {
    auto single_move = get_random_action();
    if (single_move == 0ull) {
      apply_skip();
    } else {
      apply_move_mask(single_move);
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

  friend std::ostream& operator<<(std::ostream& out, const Self& state) {
    out << state.board[0] << " " << state.board[1] << " " << state.player << " "
        << state.skipped;
    return out;
  }

  friend std::istream& operator>>(std::istream& in, Self& state) {
    uint64_t b0, b1;
    in >> b0 >> b1 >> state.player >> state.skipped;
    state.board[0] = b0;
    state.board[1] = b1;
    return in;
  }

} __attribute__((packed));