#pragma once

template<typename State, typename score_t>
struct TT {
    TT(size_t log_size) {
        data.resize(1ull << log_size);
    }

    struct Entry {
        State state;
        score_t low, high;
        bool free = true;
    };

    size_t find_slot(State& state) const {
        size_t slot = state.hash() % data.size();
        while (true) {
            if (data[slot].free) {
                return slot;
            }
            if (data[slot].state == state) {
                return slot;
            }
            slot = (slot + 1) % data.size();
        }
    }

    std::vector<Entry> data;
};