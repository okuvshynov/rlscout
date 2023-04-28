#include <cstdint>
#include <iostream>

#include "alpha_beta/alpha_beta.h"
#include "othello/othello_state.h"

using State = OthelloState<6>;
using score_t = int8_t;

int main() {
  auto AB = AlphaBeta<State, score_t>();

  State s;
  std::cout << int(AB.alpha_beta<4, true>(s.to_canonical(), -5, -3))
            << std::endl;
  return 0;
}
