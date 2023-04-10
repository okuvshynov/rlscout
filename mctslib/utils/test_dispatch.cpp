#include "othello/othello_state.h"
#include "alpha_beta/alpha_beta_runtime.h"

using State = OthelloState<6>;
using score_t = int8_t;

int main() {
  auto AB = AlphaBetaRuntime<State, score_t>();
  

  for (int game = 0; game < 10000; game++) {
    auto start = std::chrono::steady_clock::now();
    State s;

    for (int i = 0; i < 14; i++) {
      s.take_random_action();
    }

    auto moves = AB.get_move_scores(s.to_canonical(), -10, 10);
    for (auto p : moves) {
      std::cout << p.first << " " << int(p.second) << "; ";
    }
    std::cout << std::endl;
    auto curr = std::chrono::steady_clock::now();        
    std::chrono::duration<double> diff = curr - start;
    std::cerr << game << " " << diff.count() << std::endl;

  }
  return 0;
}