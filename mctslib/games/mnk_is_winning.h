#pragma once

#include <cstdint>

template<uint8_t m, uint8_t n, uint8_t k>
bool is_winning(uint64_t index, uint64_t board);

#include "../_gen/mnk_is_winning6x6x5.h"
#include "../_gen/mnk_is_winning7x7x5.h"
#include "../_gen/mnk_is_winning8x8x5.h"
