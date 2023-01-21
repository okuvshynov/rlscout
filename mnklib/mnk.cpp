#include "mnk.h"

// supports boards 6x6x5, 7x7x5 and 8x8x5

template<int N>
using SquareState = State<N, N, 5>;
template<int N>
using SquareMCTS = MCTSPlayer<N, N, 5>;

extern "C" void* new_state(int N) {
  switch (N) {
    case 6: return reinterpret_cast<void*>(new SquareState<6>());
    case 7: return reinterpret_cast<void*>(new SquareState<7>());
    case 8: return reinterpret_cast<void*>(new SquareState<8>());
  };
  return nullptr;
}

extern "C" void destroy_state(int N, void* state) {
  switch (N) {
    case 6: delete reinterpret_cast<SquareState<6>*>(state); break;
    case 7: delete reinterpret_cast<SquareState<7>*>(state); break;
    case 8: delete reinterpret_cast<SquareState<8>*>(state); break;
  };
  
}

extern "C" void* new_mcts(int N) {
  switch (N) {
    case 6: return reinterpret_cast<void*>(new SquareMCTS<6>());
    case 7: return reinterpret_cast<void*>(new SquareMCTS<7>());
    case 8: return reinterpret_cast<void*>(new SquareMCTS<8>());
  };
  return nullptr;
}

extern "C" void destroy_mcts(int N, void* mcts) {
  switch (N) {
    case 6: delete reinterpret_cast<SquareMCTS<6>*>(mcts); break;
    case 7: delete reinterpret_cast<SquareMCTS<7>*>(mcts); break;
    case 8: delete reinterpret_cast<SquareMCTS<8>*>(mcts); break;
  };
}

extern "C" bool state_apply(int N, void* st, int32_t x, int32_t y) {
  switch (N) {
    case 6: return (reinterpret_cast<SquareState<6>*>(st))->apply_move(x, y);
    case 7: return (reinterpret_cast<SquareState<7>*>(st))->apply_move(x, y);
    case 8: return (reinterpret_cast<SquareState<8>*>(st))->apply_move(x, y);
  };
  return false;
}

extern "C" bool state_finished(int N, void* st) {
  switch (N) {
    case 6: return reinterpret_cast<SquareState<6>*>(st)->finished();
    case 7: return reinterpret_cast<SquareState<7>*>(st)->finished();
    case 8: return reinterpret_cast<SquareState<8>*>(st)->finished();
  }
  return false;
}

extern "C" int state_winner(int N, void* st) {
  switch (N) {
    case 6: return reinterpret_cast<SquareState<6>*>(st)->winner;
    case 7: return reinterpret_cast<SquareState<7>*>(st)->winner;
    case 8: return reinterpret_cast<SquareState<8>*>(st)->winner;
  }
  return -1;
}

// board is expected to be N * N
// -1 is empty, not 0.
extern "C" void state_get_board(int N, void* st, int* board) {
    std::pair<std::vector<int>, std::vector<int>> boards;
    switch (N) {
      case 6: boards = reinterpret_cast<SquareState<6>*>(st)->boards(); break;
      case 7: boards = reinterpret_cast<SquareState<7>*>(st)->boards(); break;
      case 8: boards = reinterpret_cast<SquareState<8>*>(st)->boards(); break;
      default: return;
    }
    for (size_t i = 0; i < N * N; i++) {
        if (boards.first[i] == 1) {
            board[i] = 1;
        } else if (boards.second[i] == 1) {
            board[i] = 0;
        } else {
            board[i] = -1;
        }
    }
}

// boards is expected to be 2 * N * N
extern "C" void state_get_boards(int N, void* st, int* boards_out) {
    std::pair<std::vector<int>, std::vector<int>> boards;
    switch (N) {
      case 6: boards = reinterpret_cast<SquareState<6>*>(st)->boards(); break;
      case 7: boards = reinterpret_cast<SquareState<7>*>(st)->boards(); break;
      case 8: boards = reinterpret_cast<SquareState<8>*>(st)->boards(); break;
      default: return;
    }
    for (size_t i = 0; i < N * N; i++) {
        boards_out[i] = boards_out[i + N * N] = 0;
        if (boards.first[i] == 1) {
            boards_out[i] = 1;
        }
        if (boards.second[i] == 1) {
            boards_out[i + N * N] = 1;
        }
    }
}

// moves must be N*N elements
// it WILL zero out
extern "C" void mcts_get_moves(int N, void* mcts_, void* state_, double temp, int32_t rollouts, double* moves, void (*eval_cb)()) {
    eval_cb();
    switch (N) {
      case 6: {
        auto state = reinterpret_cast<SquareState<6>*>(state_);
        reinterpret_cast<SquareMCTS<6>*>(mcts_)->get_moves(state, temp, rollouts, moves);
        break;
      }
      case 7: {
        auto state = reinterpret_cast<SquareState<7>*>(state_);
        reinterpret_cast<SquareMCTS<7>*>(mcts_)->get_moves(state, temp, rollouts, moves);
        break;
      }
      case 8: {
        auto state = reinterpret_cast<SquareState<8>*>(state_);
        reinterpret_cast<SquareMCTS<8>*>(mcts_)->get_moves(state, temp, rollouts, moves);
        break;
      }
    }
}
