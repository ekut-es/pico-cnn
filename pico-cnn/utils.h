/**
 * @brief Utility functions.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef UTILS_H
#define UTILS_H

#include "parameters.h"
#include <cmath>

inline bool fp_t_eq(fp_t A, fp_t B, fp_t epsilon = EPSILON)
{
    return (std::fabs(A - B) < epsilon);
}

#endif // UTILS_H