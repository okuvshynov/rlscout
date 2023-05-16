#include "tests/expect.h"
#include "othello/othello_6x6_dumb7.h"
#include "othello/othello_6x6_board_ops.h"

void test_s_shift() {
    const char* b = 
        "......"
        "......"
        "..xx.."
        "..xx.."
        "......"
        "......";

    const char* b_shifted = 
        "......"
        "......"
        "......"
        "..xx.."
        "..xx.."
        "......";

    expect_true(board_from_string_6x6(b_shifted) == OthelloDumb7Fill6x6::s_shift(board_from_string_6x6(b)));

    const char* bb = 
        "......"
        "......"
        "......"
        "......"
        "..xx.."
        "..xx..";

    const char* bb_shifted = 
        "......"
        "......"
        "......"
        "......"
        "......"
        "..xx..";

    expect_true(board_from_string_6x6(bb_shifted) == OthelloDumb7Fill6x6::s_shift(board_from_string_6x6(bb)));
}

void test_n_shift() {
    const char* b = 
        "......"
        "......"
        "..xx.."
        "..xx.."
        "......"
        "......";

    const char* b_shifted = 
        "......"
        "..xx.."
        "..xx.."
        "......"
        "......"
        "......";

    expect_true(board_from_string_6x6(b_shifted) == OthelloDumb7Fill6x6::n_shift(board_from_string_6x6(b)));

    const char* bb = 
        "..xx.."
        "..xx.."
        "......"
        "......"
        "......"
        "......";

    const char* bb_shifted = 
        "..xx.."
        "......"
        "......"
        "......"
        "......"
        "......";

    expect_true(board_from_string_6x6(bb_shifted) == OthelloDumb7Fill6x6::n_shift(board_from_string_6x6(bb)));
}

void test_constants() {
    const char* full = 
        "xxxxxx"
        "xxxxxx"
        "xxxxxx"
        "xxxxxx"
        "xxxxxx"
        "xxxxxx";

    expect_true(board_from_string_6x6(full) == OthelloDumb7Fill6x6::full6x6);

    const char* not_e = 
        "xxxxx."
        "xxxxx."
        "xxxxx."
        "xxxxx."
        "xxxxx."
        "xxxxx.";
    expect_true(board_from_string_6x6(not_e) == OthelloDumb7Fill6x6::not_w);

    const char* not_w = 
        ".xxxxx"
        ".xxxxx"
        ".xxxxx"
        ".xxxxx"
        ".xxxxx"
        ".xxxxx";
    expect_true(board_from_string_6x6(not_w) == OthelloDumb7Fill6x6::not_e);
}


int main() {
    test_constants();
    test_s_shift();
    test_n_shift();
    std::cerr << "OK" << std::endl;
}