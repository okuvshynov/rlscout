#include "othello_state.h"

#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <array>
#include <chrono>

/*

TODO:
- multithreading
- symmetries
- log stats
- better hash table
- log training data from here?
- get rid of duplicate checks, do CPU profile 
- how can this work on GPU?

*/


/*

Current, non-optimized version: 
- 15 level ~5min 
- no-symmetry is ~100m samples.
- with symmetry - probably ~15m sample
- 1 m hours? 
- 


27835.8 12 tt_size 3 tt_hits 0 tt_rate 0 completions 2
27835.8 13 tt_size 14 tt_hits 0 tt_rate 0 completions 13
27835.8 14 tt_size 48 tt_hits 0 tt_rate 0 completions 47
27835.8 15 tt_size 225 tt_hits 3 tt_rate 1.33333 completions 224
27835.8 16 tt_size 467 tt_hits 36 tt_rate 7.70878 completions 466
27835.8 17 tt_size 2328 tt_hits 118 tt_rate 5.06873 completions 2328
27835.8 18 tt_size 3999 tt_hits 327 tt_rate 8.17704 completions 3999

bfs no symm: 

5 4
6 12
7 54
8 236
9 1256
10 6528
11 36294
12 208212

w/o symmetry, ordering, etc:

at 12 level - 800000 hours

w symmetries should be ~100000 hours
29376.8 10 tt_size 2 tt_hits 0 tt_rate 0 completions 1 cutoffs 0
29376.8 11 tt_size 9 tt_hits 0 tt_rate 0 completions 8 cutoffs 6
29376.8 12 tt_size 20 tt_hits 0 tt_rate 0 completions 19 cutoffs 7
29376.8 13 tt_size 81 tt_hits 2 tt_rate 2.46914 completions 80 cutoffs 51
29376.8 14 tt_size 215 tt_hits 7 tt_rate 3.25581 completions 214 cutoffs 129
29376.8 15 tt_size 768 tt_hits 19 tt_rate 2.47396 completions 767 cutoffs 534
29376.8 16 tt_size 1705 tt_hits 90 tt_rate 5.27859 completions 1704 cutoffs 995
29376.8 17 tt_size 6441 tt_hits 363 tt_rate 5.63577 completions 6440 cutoffs 5035
29376.8 18 tt_size 12637 tt_hits 711 tt_rate 5.62634 completions 12637 cutoffs 6872
29376.8 19 tt_size 46205 tt_hits 3218 tt_rate 6.96461 completions 0 cutoffs 37716
29376.8 20 tt_size 85520 tt_hits 5605 tt_rate 6.55402 completions 0 cutoffs 45519
29376.8 21 tt_size 288794 tt_hits 23863 tt_rate 8.26298 completions 0 cutoffs 237217
29376.8 22 tt_size 522368 tt_hits 42999 tt_rate 8.23155 completions 0 cutoffs 282976
29376.8 23 tt_size 1594969 tt_hits 160476 tt_rate 10.0614 completions 0 cutoffs 1298837
29376.8 24 tt_size 2777743 tt_hits 291285 tt_rate 10.4864 completions 0 cutoffs 1541028


501.801
10 tt_size 1048576 tt_hits 0 tt_rate 0 completions 1 cutoffs 0
11 tt_size 1048576 tt_hits 0 tt_rate 0 completions 7 cutoffs 6
12 tt_size 1048576 tt_hits 0 tt_rate 0 completions 13 cutoffs 3
13 tt_size 1048576 tt_hits 2 tt_rate 0.000190735 completions 50 cutoffs 35
14 tt_size 1048576 tt_hits 0 tt_rate 0 completions 126 cutoffs 73
15 tt_size 1048576 tt_hits 10 tt_rate 0.000953674 completions 423 cutoffs 298
16 tt_size 1048576 tt_hits 47 tt_rate 0.00448227 completions 877 cutoffs 502
17 tt_size 1048576 tt_hits 195 tt_rate 0.0185966 completions 3176 cutoffs 2474
18 tt_size 1048576 tt_hits 359 tt_rate 0.0342369 completions 5921 cutoffs 3136
19 tt_size 1048576 tt_hits 1454 tt_rate 0.138664 completions 21320 cutoffs 17550
20 tt_size 1048576 tt_hits 2561 tt_rate 0.244236 completions 37438 cutoffs 19188
21 tt_size 1048576 tt_hits 10259 tt_rate 0.978374 completions 130022 cutoffs 108821
22 tt_size 1048576 tt_hits 18762 tt_rate 1.78928 completions 219991 cutoffs 111634
23 tt_size 1048576 tt_hits 68205 tt_rate 6.50454 completions 707902 cutoffs 593158


w/o symmetry, multithreading, ordereing -- probably ~2500 hours.

*/


using State = OthelloState<6>;
using score_t = int32_t;
const auto min_score = std::numeric_limits<score_t>::min();
const auto max_score = std::numeric_limits<score_t>::max();

constexpr size_t kLevels = 37;

uint64_t leaves = 0ull;
uint64_t tt_hits[kLevels] = {0ull};
uint64_t completions[kLevels] = {0ull};
uint64_t cutoffs[kLevels] = {0ull};

int32_t tt_max_level = 25;
int32_t log_max_level = 15;

constexpr size_t tt_size = 1 << 20; 
std::array<std::pair<State, score_t>, tt_size> transposition_table[kLevels];

auto start = std::chrono::steady_clock::now();

void init_tt() {
    for (size_t i = 0; i < kLevels; i++) {
        for (auto& p: transposition_table[i]) {
            p.first.board[0] = 0ull;
            p.first.board[1] = 0ull;
        }
    }
}

score_t alpha_beta(const State& state, score_t alpha, score_t beta, bool do_max) {
    if (state.finished()) {
        leaves++;
        return state.score(0);
    }
    auto depth = state.stones_played();

    size_t slot = 0;

    if (depth < tt_max_level) {
        slot = state.hash() % tt_size;

        if (transposition_table[depth][slot].first == state) {
            tt_hits[depth]++;
            return transposition_table[depth][slot].second;
        }
    }
    auto moves = state.valid_actions();
    score_t value;
    if (do_max) {
        value = min_score;

        if (moves == 0ull) {
            State new_state = state;
            new_state.apply_skip();
            value = std::max(value, alpha_beta(new_state, alpha, beta, false));
        } else {
            for (uint64_t k = 0; k < State::M * State::N; k++) {
                if ((1ull << k) & moves) {
                    State new_state = state;
                    new_state.apply_move_no_check(k);
                    value = std::max(value, alpha_beta(new_state, alpha, beta, false));
                    alpha = std::max(alpha, value);
                    if (value >= beta) {
                        cutoffs[depth]++;
                        break;
                    }
                }
            }
        }

    } else {
        value = max_score;

        if (moves == 0ull) {
            State new_state = state;
            new_state.apply_skip();
            value = std::min(value, alpha_beta(new_state, alpha, beta, true));
        } else {
            for (uint64_t k = 0; k < State::M * State::N; k++) {
                if ((1ull << k) & moves) {
                    State new_state = state;
                    new_state.apply_move_no_check(k);
                    value = std::min(value, alpha_beta(new_state, alpha, beta, true));
                    beta = std::min(beta, value);
                    if (value <= alpha) {
                        cutoffs[depth]++;
                        break;
                    }
                }
            }
        }
    }
    completions[depth]++;
    if (depth < tt_max_level) {
        transposition_table[depth][slot].first = state;
        transposition_table[depth][slot].second = value;
        if (depth < log_max_level) {
            auto curr = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = curr - start;
            std::cout << diff.count() << std::endl;
            for (size_t d = 0; d < kLevels; d++) {
                //const auto& tt = transposition_table[d];
                if (completions[d] == 0ull && tt_hits[d] == 0ull) {
                    continue;
                }
                std::cout << d << " tt_size "  << tt_size 
                    << " tt_hits " << tt_hits[d] 
                    << " tt_rate " << 100.0 * tt_hits[d] / tt_size
                    << " completions " << completions[d]
                    << " cutoffs " << cutoffs[d]
                    << std::endl;
            }
        }
    }
    return value;
}

int main() {
    init_tt();
    State s;
    std::cout << alpha_beta(s, min_score, max_score, true) << std::endl;
    return 0;
}