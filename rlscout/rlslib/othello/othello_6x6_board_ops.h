#pragma once

#include <cstdint>

uint64_t flip_v_6x6(uint64_t v) {
  return ((v & 0b111111000000000000000000000000000000ull) >> 30) |
         ((v & 0b111111000000000000000000000000ull) >> 18) |
         ((v & 0b111111000000000000000000ull) >> 6) |
         ((v & 0b111111000000000000ull) << 6) |
         ((v & 0b111111000000ull) << 18) | ((v & 0b111111ull) << 30);
}

// idea from
// https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#Diagonal
// reimplemented for 6x6 board with 3 other masks / delta swaps
uint64_t flip_diag_6x6(uint64_t x) {
  uint64_t t;
  const uint64_t k3 = 0x1c71c0000ull;
  const uint64_t k2 = 0x489012240ull;
  const uint64_t k1 = 0x240009000ull;
  t = k3 & (x ^ (x << 15));
  x ^= t ^ (t >> 15);
  t = k2 & (x ^ (x << 5));
  x ^= t ^ (t >> 5);
  t = k1 & (x ^ (x << 10));
  x ^= t ^ (t >> 10);
  return x;
}

uint64_t board_from_string_6x6(const char* str) {
  uint64_t res = 0ull;
  for (uint64_t i = 0; i < 36; i++) {
    if (str[i] == 'x') {
      res |= (1ull << i);
    }
  }
  return res;
}