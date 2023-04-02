#pragma once

#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>

template<typename State, typename score_t>
struct TT {
    using Self = TT<State, score_t>;
    TT(size_t log_size): log_size_(log_size) {
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

    void save_to(const std::string& filename) const {
        auto out = std::ofstream(filename);

        out << log_size_ << std::endl;
        
        for (size_t i = 0; i < data.size(); i++) {
            const auto& entry = data[i];  
            if (!entry.free) {
                out << i << " " << entry.state << " " << int64_t(entry.low) << " " << int64_t(entry.high) << std::endl;
            }
        }
    }

    static Self load_from(const std::string& filename) {
        auto in = std::ifstream(filename);
        size_t log_size, i;
        in >> log_size;
        Self res(log_size);
        std::cout << log_size << std::endl;
        uint64_t low, high;
        while (in >> i) {
            auto& entry = res.data[i];
            entry.free = false;
            in >> entry.state >> low >> high;
            entry.low = low;
            entry.high = high;
            if (entry.state.stones_played() < 8) {
                entry.state.p();
                std::cout << entry.low << " " << entry.high << std::endl;
            }
        }
        return res;
    }

    const size_t log_size_;

    std::vector<Entry> data;
};