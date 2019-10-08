/**
 * @brief functions for fixed16 numbers
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef SRC_FIXED16_H_
#define SRC_FIXED16_H_

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <limits.h>

// how many bits are used to store a fixed point number
#define FIXED_LENGTH 16

// how many bits are used to store the values after the comma
#define DECIMAL_PLACES 10

// minimal fraction of what can be represented by the fixed point number
#define PRECISION (1.0 / (1 << DECIMAL_PLACES))

// 1 as fixed point number
#define FIXED_ONE (0 | (1 << DECIMAL_PLACES))

// 0 as fixed point number
#define FIXED_ZERO 0x0000

// after comma mask
#define AFTER_COMMA_MASK (0xFFFF >> (FIXED_LENGTH-DECIMAL_PLACES))

// type in which fixed point numbers will be stored
typedef int16_t fixed16_t;

/**
 * @brief converts a floating point number into a fixed point number
 *
 * @param float value
 * @return fixed16 value
 */
fixed16_t float_to_fixed16(fp_t float_value);

/**
 * @brief converts a fixed point number into a floating point number
 *
 * @param fixed16 value
 * @return float value
 */
fp_t fixed16_to_float(fixed16_t fixed);

/**
 * @brief converts an integer to a fixed point number
 *
 * @param int16_t value
 * @return fixed16 value
 */
fixed16_t int_to_fixed16(int16_t a);

// converts a fixed point number to an integer
int16_t fixed16_to_int16(fixed16_t a);

/**
 * @brief adds two fixed point values
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a+b
 */
fixed16_t add_fixed16(fixed16_t a, fixed16_t b);

/**
 * @brief subtracts two fixed point numbers
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a-b
 */
fixed16_t sub_fixed16(fixed16_t a, fixed16_t b);

/**
 * @brief multiplies two fixed point values
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a*b
 */
fixed16_t mul_fixed16(fixed16_t a, fixed16_t b);

/**
 * @brief divides two fixed point numbers
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a/b
 */
fixed16_t div_fixed16(fixed16_t a, fixed16_t b);

/**
 * @brief calculates exp(x) of a fixed point number x
 *
 * @param x
 * @return exp(x)
 */
fixed16_t exp_fixed16(fixed16_t x);

/**
 * @brief calculates exp(x) of a int32 number x
 *
 * @param x
 * @return exp(x)
 */
int32_t exp_int32(int32_t x);

/**
 * @brief calculates exp(x) of a int16 number x
 *
 * @param x
 * @return exp(x)
 */
int16_t exp_int16(int16_t x);

#endif /* SRC_FIXED16_H_ */
