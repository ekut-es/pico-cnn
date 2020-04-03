#include "test_utility.h"

/**
 * @brief generates uniform random number in range
 * @param min (lower bound)
 * @param max (upper bound)
 */
static inline fp_t urand(fp_t min, fp_t max) {
    return (fp_t) ((((fp_t)rand()/(fp_t)(RAND_MAX)) * 1.0f) * (max - min) + min);
}