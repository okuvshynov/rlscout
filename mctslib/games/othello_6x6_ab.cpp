#include <iostream>
#include <cstdint>

#include "alpha_beta.h"
#include "othello_state.h"


using State = OthelloState<6>;
using score_t = int32_t;

int main() {
  auto AB = AlphaBeta<State, score_t>();
  
  State s;
  std::cout << AB.alpha_beta<4, true>(s.to_canonical(), -4, -3) << std::endl;
  return 0;
}
