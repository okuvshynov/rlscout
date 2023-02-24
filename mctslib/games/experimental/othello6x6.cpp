// some exprimentation to check if valid move check works well
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

struct State {
    uint64_t self = 0ull, opp = 0ull;

    // string of '.' '0' and 'X'.
    static State from_string(const std::string& board) {
        assert(board.size() == 36);
        State res;
        for (uint64_t i = 0; i < board.size(); ++i) {
            if (board[i] == '0') {
                res.self |= (1ull << i);
            }
            if (board[i] == 'X') {
                res.opp |= (1ull << i);
            }
        }
        return res;
    }

    void p() const {
        for (uint64_t i = 0; i < 36; i++) {
            if (self & (1ull << i)) {
                printf("0");
            } else 
            if (opp & (1ull << i)) {
                printf("X");
            }
            else {
                printf(".");
            }
            if (i % 6 == 5) {
                printf("\n");
            }
        }
    }

    void pp(uint64_t v) const {
        for (uint64_t i = 0; i < 36; i++) {
            if (v & (1ull << i)) {
                printf("0");
            } else {
                printf(".");
            }
            if (i % 6 == 5) {
                printf("\n");
            }
        }
    }

    // just look at 1 direction first.
    // seems like if we implement 8 such function (1 for each direction)
    // we should be able to get all valid moves quickly
    // it is possible this can be vectorized (AVX/NEON) or even run on GPU
    // for many games at a time. This is something we can look into at stage (2)
    uint64_t valid_moves_h0() const {
        uint64_t res = 0ull;
        uint64_t kRight1Mask = ~0b000001000001000001000001000001000001ull;
        uint64_t kRight2Mask = ~0b000011000011000011000011000011000011ull;
        uint64_t kRight3Mask = ~0b000111000111000111000111000111000111ull;
        uint64_t kRight4Mask = ~0b001111001111001111001111001111001111ull;
        uint64_t kRight5Mask = ~0b011111011111011111011111011111011111ull;

        // elements to the left of opp
        uint64_t o1 = (opp & kRight1Mask) >> 1ull;
        uint64_t s1 = (self & kRight2Mask) >> 2ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight2Mask) >> 2ull);
        s1 = (self & kRight3Mask) >> 3ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight3Mask) >> 3ull);
        s1 = (self & kRight4Mask) >> 4ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight4Mask) >> 4ull);
        s1 = (self & kRight5Mask) >> 5ull;
        res |= (o1 & s1);

        // which are empty:
        res = res & (~(opp | self));

        pp(res);
        return res;
    }

    uint64_t valid_moves_h1() const {
        uint64_t res = 0ull;
        uint64_t kRight1Mask  = ~0b100000100000100000100000100000100000ull;
        uint64_t kRight2Mask  = ~0b110000110000110000110000110000110000ull;
        uint64_t kRight3Mask  = ~0b111000111000111000111000111000111000ull;
        uint64_t kRight4Mask  = ~0b111100111100111100111100111100111100ull;
        uint64_t kRight5Mask  = ~0b111110111110111110111110111110111110ull;
        

        // elements to the left of opp
        uint64_t o1 = (opp & kRight1Mask) << 1ull;
        uint64_t s1 = (self & kRight2Mask) << 2ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight2Mask) << 2ull);
        s1 = (self & kRight3Mask) << 3ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight3Mask) << 3ull);
        s1 = (self & kRight4Mask) << 4ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kRight4Mask) << 4ull);
        s1 = (self & kRight5Mask) << 5ull;
        res |= (o1 & s1);

        // which are empty:
        res = res & (~(opp | self));

        pp(res);
        return res;
    }
};

std::string bb = 
".X0..."
".XXXX0"
".XX0.."
".X0X.."
"..XX0."
"0XX...";

int main() {
    auto s = State::from_string(bb);
    s.valid_moves_h0();
    s.valid_moves_h1();
    return 0;
}