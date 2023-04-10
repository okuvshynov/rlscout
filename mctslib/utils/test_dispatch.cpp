#include "othello/othello_state.h"
#include "alpha_beta/alpha_beta_runtime.h"

using State = OthelloState<6>;
using score_t = int8_t;

int main() {
  auto AB = AlphaBetaRuntime<State, score_t>();
  
  State s;

    for (int i = 0; i < 15; i++) {
        s.take_random_action();
        s.p();
        std::cout << std::endl;
    }

  auto moves = AB.get_move_scores(s.to_canonical(), -5, -3);
  for (auto p : moves) {
    std::cout << p.first << " " << int(p.second) << std::endl;
  }
  return 0;
}