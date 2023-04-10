#pragma once

#include "local_tt.h"
#include "shared_tt.h"

template <typename State, typename score_t>
struct TT {
  static constexpr uint32_t tt_full_level = 24;
  static constexpr int32_t tt_max_level = 33;

  SharedTT<State, score_t> full_tt = SharedTT<State, score_t>{27};

  using LocalTTable = LocalTT<State, score_t, tt_full_level, tt_max_level>;

  LocalTTable replacement_tt = LocalTTable{};

  TT() { std::cout << "tt!" << std::endl; replacement_tt.init_tt(); }

  template <uint32_t stones>
  bool lookup_and_init(const State& state, size_t& slot, score_t& alpha,
                       score_t& beta, score_t& value) {
    if constexpr (stones < tt_full_level) {
      return full_tt.template lookup_and_init<stones>(state, slot, alpha, beta,
                                                      value);
    } else if constexpr (stones < tt_max_level) {
      return replacement_tt.template lookup_and_init<stones>(state, slot, alpha,
                                                             beta, value);
    }
    return false;
  }

  template <uint32_t stones>
  void update(const State& state, size_t& slot, score_t& alpha, score_t& beta,
              score_t& value) {
    if constexpr (stones < tt_full_level) {
      full_tt.template update<stones>(state, slot, alpha, beta, value);
    } else if constexpr (stones < tt_max_level) {
      replacement_tt.template update<stones>(state, slot, alpha, beta, value);
    }
  }

  void log_stats() const {
    full_tt.print_stats();
    replacement_tt.print_stats();
  }
};