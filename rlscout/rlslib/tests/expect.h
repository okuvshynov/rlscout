#pragma once

#include <iostream>

void expect_true(bool val) {
    if (!val) {
        std::cerr << "FAIL" << std::endl;
        exit(1);
    }
}