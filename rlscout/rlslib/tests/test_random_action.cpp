#include "othello/othello_state.h"

void expect_true(bool val) {
    if (!val) {
        std::cerr << "FAIL" << std::endl;
        exit(1);
    }
}

void test_take_random_action() {
    OthelloState<6> state;
    uint64_t hits[36] = {0ull};
    for (int r = 0; r < 40000; r++) {
        uint64_t v = state.get_random_action();
        hits[__builtin_ffsll(v) - 1]++;
    }
    for (int i = 0; i < 36; i++) {
        if (i == 9 || i == 16 || i == 19 || i == 26) {
            expect_true(hits[i] > 9500 && hits[i] < 10500);
        } else {
            expect_true(hits[i] == 0ull);
        }
    }
}

int main() {
    test_take_random_action();
    std::cerr << "OK" << std::endl;
}