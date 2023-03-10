#include "othello_state.h"

#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <array>
#include <chrono>

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

constexpr int32_t tt_max_level = 32;
constexpr int32_t log_max_level = 14;
constexpr int32_t canonical_max_level = 15;

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

template<uint32_t stones, bool do_max>
score_t alpha_beta(State state, score_t alpha, score_t beta) {
    if (state.finished() || state.full()) {
        leaves++;
        return state.score(0);
    }

    size_t slot = 0;

    // TODO this becomes if constexpr 
    if constexpr(stones < tt_max_level) {
        slot = state.hash() % tt_size;

        if (transposition_table[stones][slot].state == state) {
            if (transposition_table[stones][slot].low >= beta) {
                tt_hits[stones]++;
                return transposition_table[stones][slot].low;
            }
            if (transposition_table[stones][slot].high <= alpha) {
                tt_hits[stones]++;
                return transposition_table[stones][slot].high;
            }

            alpha = std::max(alpha, transposition_table[stones][slot].low);
            beta = std::min(beta, transposition_table[stones][slot].high);
        } else {
            if (!transposition_table[stones][slot].state.empty()) {
                evictions[stones]++;
            } 
            // override / init slot
            transposition_table[stones][slot].low = min_score;
            transposition_table[stones][slot].high = max_score;
        }
    }
    auto alpha0 = alpha;
    auto beta0 = beta;
    auto moves = state.valid_actions();
    score_t value;
    if constexpr(do_max) {
        value = min_score;

        if (moves == 0ull) {
            State new_state = state;
            new_state.apply_skip();
            value = std::max(value, alpha_beta<stones, false>(new_state, alpha, beta));
        } else {
            if constexpr(stones + 1 == State::M * State::N) {
                State new_state = state;
                new_state.apply_move_mask(moves);
                return new_state.score(0);
            } else {
                while (moves) {
                    uint64_t other_moves = (moves & (moves - 1));
                    uint64_t move = moves ^ other_moves;
                    State new_state = state;
                    new_state.apply_move_mask(move);
                    if constexpr(stones + 1 < canonical_max_level) {
                        new_state = new_state.to_canonical();
                    }
                    
                    value = std::max(value, alpha_beta<stones + 1, false>(new_state, alpha, beta));
                    alpha = std::max(alpha, value);
                    if (value >= beta) {
                        cutoffs[stones]++;
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
            value = std::min(value, alpha_beta<stones, true>(new_state, alpha, beta));
        } else {
            if constexpr(stones + 1 == State::M * State::N) {
                State new_state = state;
                new_state.apply_move_mask(moves);
                return new_state.score(0);
            } else {
                while (moves) {
                    uint64_t other_moves = (moves & (moves - 1));
                    uint64_t move = moves ^ other_moves;
                    State new_state = state;
                    new_state.apply_move_mask(move);
                    if constexpr(stones + 1 < canonical_max_level) {
                        new_state = new_state.to_canonical();
                    }
                    
                    value = std::min(value, alpha_beta<stones + 1, true>(new_state, alpha, beta));
                    beta = std::min(beta, value);
                    if (value <= alpha) {
                        cutoffs[stones]++;
                        break;
                    }
                    moves = other_moves;
                }
            } 
        }
    }
    completions[stones]++;
    if constexpr(stones < tt_max_level) {
        transposition_table[stones][slot].state = state;
        
        if (value <= alpha0) {
            transposition_table[stones][slot].high = value;
        }
        if (value >= beta0){
            transposition_table[stones][slot].low = value;
        } 
        if (value > alpha0 && value < beta0) {
            transposition_table[stones][slot].high = value;
            transposition_table[stones][slot].low = value;
        }
        if (stones < log_max_level) {
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
    init_tt();
    State s;
    std::cout << alpha_beta<4, true>(s.to_canonical(), min_score, max_score) << std::endl;
    return 0;
}