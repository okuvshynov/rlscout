#include "othello_state.h"

#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <array>
#include <chrono>

/*

* Faster symmetries
* how to log training data?
* store everything for first N layers
* multithreading
* profile/try running on cloud machine.
* different template based implementation for last levels  
- multithreading
- better hashing
- log stats
- log training data from here?
- get rid of duplicate checks, do CPU profile 
- how can this work on GPU?
- good logging 

bfs symm
5 1
6 3
7 14 ~ 3-5h each? 4 likely
8 60 ~1.5h each?
9 314 ~ 20-25 min each
10 1635
11 9075 ~ 2-3 min each
12 51988
13 293004
14 1706701

196150
6 tt_hits 0 tt_rate 0 completions 2 cutoffs 0 evictions 0
7 tt_hits 0 tt_rate 0 completions 11 cutoffs 5 evictions 0
8 tt_hits 1 tt_rate 1.19209e-05 completions 36 cutoffs 16 evictions 0
9 tt_hits 0 tt_rate 0 completions 140 cutoffs 95 evictions 0
10 tt_hits 6 tt_rate 7.15256e-05 completions 396 cutoffs 247 evictions 0
11 tt_hits 45 tt_rate 0.000536442 completions 1301 cutoffs 912 evictions 0
12 tt_hits 171 tt_rate 0.00203848 completions 3756 cutoffs 2529 evictions 0
13 tt_hits 495 tt_rate 0.00590086 completions 11551 cutoffs 8116 evictions 6
14 tt_hits 1611 tt_rate 0.0192046 completions 33764 cutoffs 23682 evictions 69

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
uint64_t evictions[kLevels] = {0ull};

int32_t tt_max_level = 32;
int32_t log_max_level = 14;
int32_t canonical_max_level = 15;

constexpr size_t tt_size = 1 << 23; 


struct TTEntry {
    State state;
    score_t low, high;
};

std::vector<TTEntry> transposition_table[kLevels];

auto start = std::chrono::steady_clock::now();

void init_tt() {
    for (size_t i = 0; i < kLevels; i++) {
        transposition_table[i].resize(tt_size);
        for (auto& p: transposition_table[i]) {
            p.state.board[0] = 0ull;
            p.state.board[1] = 0ull;
        }
    }
}

score_t alpha_beta(State state, score_t alpha, score_t beta, bool do_max) {
    if (state.finished() || state.full()) {
        leaves++;
        return state.score(0);
    }

    // TODO: we should make this template param 
    auto depth = state.stones_played();

    size_t slot = 0;

    // TODO this becomes if constexpr 
    if (depth < tt_max_level) {
        slot = state.hash() % tt_size;

        if (transposition_table[depth][slot].state == state) {
            if (transposition_table[depth][slot].low >= beta) {
                tt_hits[depth]++;
                return transposition_table[depth][slot].low;
            }
            if (transposition_table[depth][slot].high <= alpha) {
                tt_hits[depth]++;
                return transposition_table[depth][slot].high;
            }

            alpha = std::max(alpha, transposition_table[depth][slot].low);
            beta = std::min(beta, transposition_table[depth][slot].high);
        } else {
            if (!transposition_table[depth][slot].state.empty()) {
                evictions[depth]++;
            } 
            // override / init slot
            transposition_table[depth][slot].low = min_score;
            transposition_table[depth][slot].high = max_score;
        }
    }
    auto alpha0 = alpha;
    auto beta0 = beta;
    // TODO: last level can avoid this whole thing
    auto moves = state.valid_actions();
    score_t value;
    if (do_max) {
        value = min_score;

        if (moves == 0ull) {
            State new_state = state;
            new_state.apply_skip();
            value = std::max(value, alpha_beta(new_state, alpha, beta, false));
       } else {
            // we have one space only
            // TODO: static if
            if (depth + 1 == State::M * State::N) {
                State new_state = state;
                new_state.apply_move_mask(moves);
                return new_state.score(0);
            } else {
                while (moves) {
                    uint64_t other_moves = (moves & (moves - 1));
                    uint64_t move = moves ^ other_moves;
                    State new_state = state;
                    new_state.apply_move_mask(move);
                    if (depth + 1 < canonical_max_level) {
                        new_state = new_state.to_canonical();
                    }
                    
                    value = std::max(value, alpha_beta(new_state, alpha, beta, false));
                    alpha = std::max(alpha, value);
                    if (value >= beta) {
                        cutoffs[depth]++;
                        break;
                    }
                    moves = other_moves;
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
            if (depth + 1 == State::M * State::N) {
                State new_state = state;
                new_state.apply_move_mask(moves);
                return new_state.score(0);
            } else {
                while (moves) {
                    uint64_t other_moves = (moves & (moves - 1));
                    uint64_t move = moves ^ other_moves;
                    State new_state = state;
                    new_state.apply_move_mask(move);
                    if (depth + 1 < canonical_max_level) {
                        new_state = new_state.to_canonical();
                    }
                    
                    value = std::min(value, alpha_beta(new_state, alpha, beta, true));
                    beta = std::min(beta, value);
                    if (value <= alpha) {
                        cutoffs[depth]++;
                        break;
                    }
                    moves = other_moves;
                }
            } 
        }
    }
    completions[depth]++;
    if (depth < tt_max_level) {
        transposition_table[depth][slot].state = state;
        
        if (value <= alpha0) {
            transposition_table[depth][slot].high = value;
        }
        if (value >= beta0){
            transposition_table[depth][slot].low = value;
        } 
        if (value > alpha0 && value < beta0) {
            transposition_table[depth][slot].high = value;
            transposition_table[depth][slot].low = value;
        }
        if (depth < log_max_level) {
            auto curr = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = curr - start;
            std::cout << diff.count() << std::endl;
            for (size_t d = 0; d < kLevels; d++) {
                //const auto& tt = transposition_table[d];
                if (completions[d] == 0ull && tt_hits[d] == 0ull) {
                    continue;
                }
                std::cout << d
                    << " tt_hits " << tt_hits[d] 
                    << " tt_rate " << 100.0 * tt_hits[d] / tt_size
                    << " completions " << completions[d]
                    << " cutoffs " << cutoffs[d]
                    << " evictions " << evictions[d]
                    << std::endl;
            }
        }
    }
    return value;
}

struct StateHash {
    std::size_t operator()(const State& s) const {
        return s.hash();
    }
};

void bfs() {
    std::unordered_set<State, StateHash> curr, next;
    State s;
    curr.insert(s);
    while (true) {
        next.clear();
        for (auto state: curr) {
            auto moves = state.valid_actions();
            for (uint64_t k = 0; k < State::M * State::N; k++) {
                if ((1ull << k) & moves) {
                    State new_state = state;
                    new_state.apply_move_no_check(k);
                    next.insert(new_state.to_canonical());
                }
            }
        }
        std::cout << next.begin()->stones_played() << " " << next.size() << std::endl;
        std::swap(next, curr);
    }
}

int main() {
    //bfs();
    init_tt();
    State s;
    std::cout << alpha_beta(s.to_canonical(), min_score, max_score, true) << std::endl;
    return 0;
}