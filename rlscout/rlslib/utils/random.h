#pragma once

#include <cstdint>
#include <random>

// single random generator with a seed to make it reproducible
struct RandomGen {
    static RandomGen& instance() {
        static RandomGen instance_;
        return instance_;
    }

    static void init(int32_t seed) {
        RandomGen::instance().gen_ = std::mt19937(seed);
    }

    static std::mt19937& gen() {
        return RandomGen::instance().gen_;
    }

    private:
        RandomGen() : gen_(1991) {        }
        std::mt19937 gen_;
};