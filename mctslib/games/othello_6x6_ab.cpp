#include "othello_state.h"

#include <unordered_map>
#include <vector>
#include <unordered_set>

#include <chrono>


/*

TODO:
- multithreading
- symmetries
- log stats
- better hash table
- log training data from here?
- get rid of duplicate checks, do CPU profile 

*/

using State = OthelloState<6>;

const auto min_score = std::numeric_limits<double>::min();
const auto max_score = std::numeric_limits<double>::max();

uint64_t leaves = 0ull;
uint64_t tt_hits = 0ull;
int32_t tt_max_level = 25;

struct StateHash {
    std::size_t operator()(const State& s) const {
        return s.hash();
    }
};
std::unordered_map<State, double, StateHash> transposition_table;

auto start = std::chrono::steady_clock::now();


double alpha_beta(const State& state, double alpha, double beta, bool do_max) {
    if (state.finished()) {
        leaves++;
        if (0xfffffff == (leaves & 0xfffffff)) {
            std::cout << "visited " << leaves << " leaves" << std::endl;
            std::cout << "tt size " << transposition_table.size() << " hits " << tt_hits << " rate " << 100.0 * tt_hits / transposition_table.size() << std::endl;
        }
        return state.score(0);
    }
    auto depth = state.stones_played();
    if (depth < tt_max_level) {

        auto it = transposition_table.find(state);
        if (it != transposition_table.end()) {
            tt_hits++;
            return it->second;
        }
    }
    auto moves = state.valid_actions();
    double value = 0.0;
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
                    if (value > beta) {
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
                    if (value < alpha) {
                        break;
                    }
                }
            }
        }
    }
    if (depth < tt_max_level) {
        transposition_table[state] = value;
        if (depth < 18) {
            auto curr = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = curr - start;
            std::cout << "done at d = " << depth << " " << diff.count() << std::endl;
        }
    }
    return value;
}


int main() {
    State s;
    std::cout << alpha_beta(s, min_score, max_score, true) << std::endl;
    return 0;
}