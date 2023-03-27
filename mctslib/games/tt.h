#pragma once

#include <cstdint>
#include <vector>

template<typename State, typename score_t>
struct TT {
    TT(size_t log_size) {
        data.resize(1ull << log_size);
        std::cout << "how large " << sizeof(Entry) << std::endl;
    }

    // for 0 window search we can further reduce size
    // for interval [a; a + 1] entry can be in 3 states:
    // [a; a] [a; a + 1], [a + 1, a + 1], so we need 2 bits for that
    // board itself is much larger space consumer though
    // naive way to save 4 bits would be to have bitboards be like this:
    // 1 - empty or not, with 4 center squares never being empty, so 32 bits
    // 2 - white 
    struct CompactEntry {
        uint64_t board0: 36;
        uint64_t board1: 36;
        uint32_t player: 4;
        uint32_t skipped: 4;
        int8_t low;
        int8_t high;
    } __attribute__((packed));

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