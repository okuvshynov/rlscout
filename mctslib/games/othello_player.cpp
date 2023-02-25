#include "othello_state.h"
#include <chrono>
#include <thread>

int main() {
    OthelloState<6> state;

    using namespace std::chrono_literals;

    while (!state.finished()) {
        state.take_random_action();
        state.p();
    }

    return 0;
}