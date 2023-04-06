#include "othello/othello_state.h"
#include <iostream>

int main() {
    OthelloState<6> state;

    while (!state.finished()) {
        state.take_random_action();
        state.p();
        state.to_canonical().p();
        std::cout << std::endl;
    }

    return 0;
}