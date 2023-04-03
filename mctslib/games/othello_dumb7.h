#pragma once

#include <cstdint>

// based on https://www.chessprogramming.org/Dumb7Fill
struct OthelloDumb7Fill6x6 {
  static const uint64_t full = 0xffffffffffffffffull;
  static const uint64_t not_e = ~(full & 0b000001000001000001000001000001000001ull);
  static const uint64_t not_w = ~(full & 0b100000100000100000100000100000100000ull);
  static const uint64_t full6x6 = (1ull << 36ull) - 1ull;

  static uint64_t s_shift(uint64_t b) { return (b << 6) & full6x6; }
  static uint64_t n_shift(uint64_t b) { return b >> 6; }
  static uint64_t e_shift(uint64_t b) { return (b << 1) & not_e & full6x6; }
  static uint64_t se_shift(uint64_t b) { return (b << 7) & not_e & full6x6; }
  static uint64_t ne_shift(uint64_t b) { return (b >> 5) & not_e; }
  static uint64_t w_shift(uint64_t b) { return (b >> 1) & not_w; }
  static uint64_t sw_shift(uint64_t b) { return (b << 5) & not_w & full6x6; }
  static uint64_t nw_shift(uint64_t b) { return (b >> 7) & not_w; }

  static uint64_t s_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    flood |= gen = (gen << 6) & prop;
    flood |= gen = (gen << 6) & prop;
    flood |= gen = (gen << 6) & prop;
    flood |= (gen << 6) & prop;
    return flood;
  }

  static uint64_t n_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    flood |= gen = (gen >> 6) & prop;
    flood |= gen = (gen >> 6) & prop;
    flood |= gen = (gen >> 6) & prop;
    flood |= (gen >> 6) & prop;
    return flood;
  }

  static uint64_t e_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    prop &= not_e;
    flood |= gen = (gen << 1) & prop;
    flood |= gen = (gen << 1) & prop;
    flood |= gen = (gen << 1) & prop;
    flood |= (gen << 1) & prop;
    return flood & not_e;
  }

  static uint64_t ne_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    prop &= not_e;
    flood |= gen = (gen >> 5) & prop;
    flood |= gen = (gen >> 5) & prop;
    flood |= gen = (gen >> 5) & prop;
    flood |= (gen >> 5) & prop;
    return flood & not_e;
  }

  static uint64_t se_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    prop &= not_e;
    flood |= gen = (gen << 7) & prop;
    flood |= gen = (gen << 7) & prop;
    flood |= gen = (gen << 7) & prop;
    flood |= (gen << 7) & prop;
    return flood & not_e;
  }

  static uint64_t w_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    prop &= not_w;
    flood |= gen = (gen >> 1) & prop;
    flood |= gen = (gen >> 1) & prop;
    flood |= gen = (gen >> 1) & prop;
    flood |= (gen >> 1) & prop;
    return flood & not_w;
  }

  static uint64_t sw_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ull;
    prop &= not_w;
    flood |= gen = (gen << 5) & prop;
    flood |= gen = (gen << 5) & prop;
    flood |= gen = (gen << 5) & prop;
    flood |= (gen << 5) & prop;
    return flood & not_w;
  }

  static uint64_t nw_fill(uint64_t gen, uint64_t prop) {
    uint64_t flood = 0ULL;
    prop &= not_w;
    flood |= gen = (gen >> 7) & prop;
    flood |= gen = (gen >> 7) & prop;
    flood |= gen = (gen >> 7) & prop;
    flood |= (gen >> 7) & prop;
    return flood & not_w;
  }

  static uint64_t valid_moves(uint64_t gen, uint64_t prop) {
    uint64_t flood = s_shift(s_fill(gen, prop));
    flood |= n_shift(n_fill(gen, prop));
    flood |= e_shift(e_fill(gen, prop));
    flood |= se_shift(se_fill(gen, prop));
    flood |= ne_shift(ne_fill(gen, prop));
    flood |= w_shift(w_fill(gen, prop));
    flood |= sw_shift(sw_fill(gen, prop));
    flood |= nw_shift(nw_fill(gen, prop));

    return flood & ~(gen | prop);
  }

  // TODO: create special versions for boundary? where we need only 5 directions? 3 directions?
  // TODO: early stop
  // For each of the 8 directions we can have a mask to check that empty slot is in the right area.
  // For example, for all corners we can only do 3 directions, etc.
  // gen1 - place we put new stone
  // gen2 - self stones
  // prop - opp stones
  static uint64_t to_flip_old(uint64_t gen1, uint64_t gen2, uint64_t prop) {
    uint64_t flood 
           = s_fill(gen1, prop) & n_fill(gen2, prop);
    flood |= n_fill(gen1, prop) & s_fill(gen2, prop);

    flood |= e_fill(gen1, prop) & w_fill(gen2, prop);
    flood |= w_fill(gen1, prop) & e_fill(gen2, prop);

    flood |= se_fill(gen1, prop) & nw_fill(gen2, prop);
    flood |= nw_fill(gen1, prop) & se_fill(gen2, prop);

    flood |= ne_fill(gen1, prop) & sw_fill(gen2, prop);
    flood |= sw_fill(gen1, prop) & ne_fill(gen2, prop);

    return flood;
  }

  // a few conditions are better than the old version, as each _fill 
  // is quite expensive
  static uint64_t to_flip(uint64_t gen1, uint64_t gen2, uint64_t prop) {
    uint64_t flood = 0ull;
    static constexpr uint64_t s_mask = 0b000000000000111111111111111111111111ull;
    static constexpr uint64_t n_mask = 0b111111111111111111111111000000000000ull;
    static constexpr uint64_t w_mask = 0b111100111100111100111100111100111100ull;
    static constexpr uint64_t e_mask = 0b001111001111001111001111001111001111ull;

    const bool w_good = (gen1 & w_mask) == gen1;
    const bool e_good = (gen1 & e_mask) == gen1;

    if ((gen1 & s_mask) == gen1) {
      flood |= s_fill(gen1, prop) & n_fill(gen2, prop);
      if (e_good) {
        flood |= se_fill(gen1, prop) & nw_fill(gen2, prop);
      }
      if (w_good) {
        flood |= sw_fill(gen1, prop) & ne_fill(gen2, prop);
      }
    }

    if ((gen1 & n_mask) == gen1) {
      flood |= n_fill(gen1, prop) & s_fill(gen2, prop);
      if (w_good) {
        flood |= nw_fill(gen1, prop) & se_fill(gen2, prop);
      }

      if (e_good) {
        flood |= ne_fill(gen1, prop) & sw_fill(gen2, prop);
      }
    }

    if (e_good) {
      flood |= e_fill(gen1, prop) & w_fill(gen2, prop);
    }

    if (w_good) {
      flood |= w_fill(gen1, prop) & e_fill(gen2, prop);
    }

    return flood;
  }
};