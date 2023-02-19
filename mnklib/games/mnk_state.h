#pragma once

#include <bit>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include "mnk_is_winning.h"

// This is mnk-game specific state
// We need to clearly define interface here
// so that we can make generic batch mcts work with other games
template<uint8_t m, uint8_t n, uint8_t k>
struct MNKState {

  // in fact for now we only support 6x6, 7x7 and 8x8 boards and 5 in a row.
  // as we didn't generate the is_winning code for other variants.
  static_assert(m <= 8 && n <= 8 && k <= n && k <= m, "up to 8x8 board is supported");
  // current player (0 or 1)
  int32_t player = 0;
  // -1 if not finished or draw
  int32_t winner = -1;
  uint64_t board[2] = {0ULL, 0ULL};

  static constexpr auto M = m;
  static constexpr auto N = n;

  static constexpr uint64_t kFullBoard = m * n < 64 ? ((1ULL << (m * n)) - 1ULL) : 0xFFFFFFFFFFFFFFFFULL;

  uint64_t mask(uint64_t index) const {
    return (1LL << index);
  }

  uint64_t index(uint64_t x, uint64_t y) const {
    return (x * n + y);
  }
  
  bool apply_move(uint64_t x, uint64_t y) {
    return apply_move(index(x, y));
  }

  bool apply_move(uint64_t index) {
    if (this->finished()) {
      return false;
    }
    if (index >= n * m) {
      return false;
    }

    uint64_t mm = this->mask(index);

    if ((board[0] & mm) || (board[1] & mm)) {
      return false;
    }

    board[player] |= mm;

    if (winning(index)) {
      winner = player;
    }
    player = 1 - player;
    return true;
  }

  uint64_t valid_actions() const {
    return kFullBoard ^ (board[0] | board[1]);
  }

  uint32_t stones_played() const {
    return std::popcount(board[0] | board[1]); 
  }

  // from POV of current player
  std::pair<std::vector<int>, std::vector<int>> boards() const {
    std::vector<int> curr(m * n, 0), opp(m * n, 0);
    for (uint64_t i = 0; i < m * n; i++) {
      if (board[player] & mask(i)) {
        curr[i] = 1;
      } 
      if (board[1 - player] & mask(i)) {
        opp[i] = 1;
      } 
    }
    return std::make_pair(curr, opp);
  }

  void fill_boards(int32_t* boards) const {
    for (uint64_t i = 0; i < m * n; i++) {
      boards[i] = boards[i + n * m] = 0;
      if (board[player] & mask(i)) {
        boards[i] = 1;
      } 
      if (board[1 - player] & mask(i)) {
        boards[i + n * m] = 1;
      } 
    }
  }

  // TODO: silly
  // Won't need this once we have value model though
  void take_random_action() {
    while (true) {
      uint64_t index = rand() % (m * n);
      if (mask(index) & valid_actions()) {
        apply_move(index);
        return;
      }
    }
  }

  bool finished() const {
    return winner >= 0 || valid_actions() == 0ULL;
  }

  double score(int player) const {
    if (winner < 0) {
      return 0.0;
    }
    return (winner == player) ? 1.0 : -1.0;
  }

  // this is slow
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
    for (uint64_t y = 0; y < m; y++) {
      for (uint64_t x = 0; x < n; x++) {
        int cell = get(x, y);
        if (cell == -1) {
          printf(".");
        } else {
          printf("%c", cell == 0 ? '0' : '+');
        }
      }
      printf("\n");
    }
  }

  bool winning(uint64_t index) const {
    return is_winning<m, n, k>(index, board[player]);
  }
};
