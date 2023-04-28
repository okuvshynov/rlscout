#pragma once

#include <cstdint>

using EvalFn = void (*)(int32_t, bool);

// Not thread safe evaluation API for model evaluation.
// memory is shared. 
// Input: fill in boards_buffer
// Call: execute eval_cb
// Get results in probs_buffer
struct ModelEvaluator {
    int32_t *boards_buffer;
    float *probs_buffer;
    float *scores_buffer; // unused at the moment
    EvalFn run;
};