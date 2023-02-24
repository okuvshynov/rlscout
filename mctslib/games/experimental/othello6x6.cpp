// some exprimentation to check if valid move check works well
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

// this may look like an overkill for self-play implementation as 
// model inference is likely to dominate the cost anyway

// However, we'll use the same implementation for full search, where speed will become 
// important

template<uint64_t N>
struct OthelloValidMoves {
    static_assert(N == 6 || N == 8, "Only 6x6 and 8x8 boards");

    static const uint64_t kLeft1Mask = ~0b000001000001000001000001000001000001ull;
    static const uint64_t kLeft2Mask = ~0b000011000011000011000011000011000011ull;
    static const uint64_t kLeft3Mask = ~0b000111000111000111000111000111000111ull;
    static const uint64_t kLeft4Mask = ~0b001111001111001111001111001111001111ull;
    static const uint64_t kLeft5Mask = ~0b011111011111011111011111011111011111ull;
    
    static const uint64_t kRight1Mask  = ~0b100000100000100000100000100000100000ull;
    static const uint64_t kRight2Mask  = ~0b110000110000110000110000110000110000ull;
    static const uint64_t kRight3Mask  = ~0b111000111000111000111000111000111000ull;
    static const uint64_t kRight4Mask  = ~0b111100111100111100111100111100111100ull;
    static const uint64_t kRight5Mask  = ~0b111110111110111110111110111110111110ull;  

    static const uint64_t kTop1Mask = ~0b000000000000000000000000000000111111ull;
    static const uint64_t kTop2Mask = ~0b000000000000000000000000111111111111ull;
    static const uint64_t kTop3Mask = ~0b000000000000000000111111111111111111ull;
    static const uint64_t kTop4Mask = ~0b000000000000111111111111111111111111ull;
    static const uint64_t kTop5Mask = ~0b000000111111111111111111111111111111ull;

    static const uint64_t kBottom1Mask = ~0b111111000000000000000000000000000000ull;
    static const uint64_t kBottom2Mask = ~0b111111111111000000000000000000000000ull;
    static const uint64_t kBottom3Mask = ~0b111111111111111111000000000000000000ull;
    static const uint64_t kBottom4Mask = ~0b111111111111111111111111000000000000ull;
    static const uint64_t kBottom5Mask = ~0b111111111111111111111111111111000000ull;

    // moving left-right is equivalent to shifts by 1 right/left
    static uint64_t valid_moves_h(uint64_t self, uint64_t opp) {
        uint64_t res = 0ull;

        // elements to the left of opp
        uint64_t o1 = (opp & kLeft1Mask) >> 1ull;
        uint64_t s1 = (self & kLeft2Mask) >> 2ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kLeft2Mask) >> 2ull);
        s1 = (self & kLeft3Mask) >> 3ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kLeft3Mask) >> 3ull);
        s1 = (self & kLeft4Mask) >> 4ull;
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kLeft4Mask) >> 4ull);
        s1 = (self & kLeft5Mask) >> 5ull;
        res |= (o1 & s1);


        // elements to the left of opp
        o1 = (opp & kRight1Mask) << 1ull;
        s1 = (self & kRight2Mask) << 2ull;
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

        return res;
    }         

    // moving up/down is equivalent to shifts by N
    static uint64_t valid_moves_v(uint64_t self, uint64_t opp) {
        uint64_t res = 0ull;

        // elements to the top of opp
        uint64_t o1 = (opp & kTop1Mask) >> (1ull * 6ull);
        uint64_t s1 = (self & kTop2Mask) >> (2ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kTop2Mask) >> (2ull * 6ull));
        s1 = (self & kTop3Mask) >> (3ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kTop3Mask) >> (3ull * 6ull));
        s1 = (self & kTop4Mask) >> (4ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kTop4Mask) >> (4ull * 6ull));
        s1 = (self & kTop5Mask) >> (5ull * 6ull);
        res |= (o1 & s1);

        // elements to the bottom of opp
        o1 = (opp & kBottom1Mask) << (1ull * 6ull);
        s1 = (self & kBottom2Mask) << (2ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kBottom2Mask) << (2ull * 6ull));
        s1 = (self & kBottom3Mask) << (3ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kBottom3Mask) << (3ull * 6ull));
        s1 = (self & kBottom4Mask) << (4ull * 6ull);
        res |= (o1 & s1);

        // next must be either self or opp
        o1 &= ((opp & kBottom4Mask) << (4ull * 6ull));
        s1 = (self & kBottom5Mask) << (5ull * 6ull);
        res |= (o1 & s1);

        // which are empty:
        res = res & (~(opp | self));

        return res;
    }
};


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

    void p(uint64_t highlight = 0ull) const {
        for (uint64_t i = 0; i < 36; i++) {
            if (highlight & (1ull << i)) {
                printf("\033[31m");
            }
            if (self & (1ull << i)) {
                printf("0");
            } else 
            if (opp & (1ull << i)) {
                printf("X");
            }
            else {
                printf("o");
            }
            if (highlight & (1ull << i)) {
                printf("\033[0m");
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
};

// do the same for verticals (where we need to shift by 6)
// and diagonals (where we shift by 5/7 in 2 directions)
// That will cover all 8 directions.

std::string bb = 
".X0..."
".XXXX0"
".XX0.."
".X0X.."
"..XX0."
"0XX...";

int main() {
    auto s = State::from_string(bb);
    s.p(OthelloValidMoves<6>::valid_moves_v(s.self, s.opp));
    s.p(OthelloValidMoves<6>::valid_moves_h(s.self, s.opp));
    
    return 0;
}