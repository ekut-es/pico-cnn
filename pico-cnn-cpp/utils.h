//
// Created by junga on 03.04.20.
//
#ifndef UTILS_H
#define UTILS_H

#include "parameters.h"
#include <cmath>

inline bool fp_t_eq(fp_t A, fp_t B, fp_t epsilon = EPSILON)
{
    return (std::fabs(A - B) < epsilon);
}

static inline fp_t urand(fp_t min, fp_t max) {
    return (fp_t) ((((fp_t)rand()/(fp_t)(RAND_MAX)) * 1.0f) * (max - min) + min);
}

#endif // UTILS_H