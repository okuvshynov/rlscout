// some exprimentation to check if valid move check works well
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

// this may look like an overkill for self-play implementation as 
// model inference is likely to dominate the cost anyway

// However, we'll use the same implementation for full search, where speed will become 
// important

// TODO: refactor all this so that it's not that long!!! 
// do not unroll everything prematurely!!!
template<uint64_t N>
struct OthelloValidMoves {
    static_assert(N == 6 || N == 8, "Only 6x6 and 8x8 boards");

    static constexpr uint64_t kLeftMasks[N - 1] = {
        ~0b000001000001000001000001000001000001ull,
        ~0b000011000011000011000011000011000011ull,
        ~0b000111000111000111000111000111000111ull,
        ~0b001111001111001111001111001111001111ull,
        ~0b011111011111011111011111011111011111ull       
    };

    static constexpr uint64_t kRightMasks[N - 1] = {
        ~0b100000100000100000100000100000100000ull,
        ~0b110000110000110000110000110000110000ull,
        ~0b111000111000111000111000111000111000ull,
        ~0b111100111100111100111100111100111100ull,
        ~0b111110111110111110111110111110111110ull 
    };

    static constexpr uint64_t kTopMasks[N - 1] = {
        ~0b000000000000000000000000000000111111ull,
        ~0b000000000000000000000000111111111111ull,
        ~0b000000000000000000111111111111111111ull,
        ~0b000000000000111111111111111111111111ull,
        ~0b000000111111111111111111111111111111ull
    };

    static constexpr uint64_t kBottomMasks[N - 1] = {
        ~0b111111000000000000000000000000000000ull,
        ~0b111111111111000000000000000000000000ull,
        ~0b111111111111111111000000000000000000ull,
        ~0b111111111111111111111111000000000000ull,
        ~0b111111111111111111111111111111000000ull
    };

    static uint64_t valid_moves(uint64_t self, uint64_t opp) {
        uint64_t res = 0ull;
        static const uint64_t full = 0xffffffffffffffffull;
        uint64_t o1 = full, o2 = full, o3 = full, o4 = full;
        uint64_t o5 = full, o6 = full, o7 = full, o8 = full;
        uint64_t s;

        for (uint64_t i = 0; i + 2 < N; i++) {
            // horizontal
            o1 &= ((opp & kLeftMasks[i]) >> (i + 1ull));
            s = (self & kLeftMasks[i + 1]) >> (i + 2ull);
            res |= (o1 & s);

            o2 &= ((opp & kRightMasks[i]) << (i + 1ull));
            s = (self & kRightMasks[i + 1]) << (i + 2ull);
            res |= (o2 & s);

            // vertical
            o3 &= ((opp & kTopMasks[i]) >> ((i + 1ull) * 6ull));
            s = (self & kTopMasks[i + 1]) >> ((i + 2ull) * 6ull);
            res |= (o3 & s);

            o4 &= ((opp & kBottomMasks[i]) << ((i + 1ull) * 6ull));
            s = (self & kBottomMasks[i + 1]) << ((i + 2ull) * 6ull);
            res |= (o4 & s);

            // diagonals
            o5 &= ((opp & kLeftMasks[i] & kTopMasks[i]) >> ((i + 1ull) * 7ull));
            s = (self & kLeftMasks[i + 1] & kTopMasks[i + 1]) >> ((i + 2ull) * 7ull);
            res |= (o5 & s);

            o6 &= ((opp & kRightMasks[i] & kTopMasks[i]) >> ((i + 1ull) * 5ull));
            s = (self & kRightMasks[i + 1] & kTopMasks[i + 1]) >> ((i + 2ull) * 5ull);
            res |= (o6 & s);

            o7 &= ((opp & kRightMasks[i] & kBottomMasks[i]) << ((i + 1ull) * 7ull));
            s = (self & kRightMasks[i + 1] & kBottomMasks[i + 1]) << ((i + 2ull) * 7ull);
            res |= (o7 & s);

            o8 &= ((opp & kLeftMasks[i] & kBottomMasks[i]) << ((i + 1ull) * 5ull));
            s = (self & kLeftMasks[i + 1] & kBottomMasks[i + 1]) << ((i + 2ull) * 5ull);
            res |= (o8 & s);
        }

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
".XX0.0"
".X0X.."
"..XX0."
"0XX...";

std::string bw = 
"......"
"....0."
"..XX.."
"..XX.."
"....0."
"......";

std::string bw2 = 
"......"
".0..X."
"..XX.."
"..XX.."
".0..X."
"......";

int main() {
    auto s = State::from_string(bw2);
    s.p(OthelloValidMoves<6>::valid_moves(s.self, s.opp));
    
    //s.p(OthelloValidMoves<6>::valid_moves_v(s.self, s.opp));
    //s.p(OthelloValidMoves<6>::valid_moves_h(s.self, s.opp));
    //s.p(OthelloValidMoves<6>::valid_moves_d1(s.self, s.opp));
    
    return 0;
}