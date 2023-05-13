#include "tests/expect.h"
#include "othello/othello_6x6_board_ops.h"

uint64_t mask_for_board(int x[6][6]) {
    uint64_t res = 0ull;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (x[i][j] == 1) {
                res |= (1ull << (i * 6ull + j));
            }
        }
    }
    return res;
}

void test_swap_diag() {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            int a[6][6] = {{0}};

            a[i][j] = 1;
            uint64_t A = flip_diag_6x6(mask_for_board(a));
            for (int ii = 0; ii < 6; ii++) {
                for (int jj = ii + 1; jj < 6; jj++) {
                    std::swap(a[ii][jj], a[jj][ii]);
                }
            }
            uint64_t B = mask_for_board(a);
            expect_true(A == B);
            // 
        }
    }
}

void test_swap_vertical() {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            int a[6][6] = {{0}};

            a[i][j] = 1;
            uint64_t A = flip_v_6x6(mask_for_board(a));
            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 6; jj++) {
                    std::swap(a[ii][jj], a[5 - ii][jj]);
                }
            }
            uint64_t B = mask_for_board(a);
            expect_true(A == B);
            // 
        }
    }
}

int main() {
    test_swap_diag();
    test_swap_vertical();
    std::cerr << "OK" << std::endl;
}