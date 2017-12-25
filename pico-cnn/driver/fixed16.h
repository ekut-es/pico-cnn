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

#ifdef __aarch64__
#include "arm_neon.h"
#endif

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

#ifdef __aarch64__
typedef int16x4_t fixed16x4_t;
typedef int16x8_t fixed16x8_t;
#endif

/**
 * @brief converts a floating point number into a fixed point number
 *
 * @param float value
 * @return fixed16 value
 */
inline fixed16_t float_to_fixed16(fp_t float_value) {
    fixed16_t temp;

    // negative
    if(float_value < 0.0) {
    	float_value = -1*float_value;
    	temp = (fixed16_t) ((fp_t) float_value * (fp_t) FIXED_ONE);
    	temp = ~temp + 1;
    } else {
    	temp = (fixed16_t) ((fp_t) float_value * (fp_t) FIXED_ONE);
    }

    return temp;
}

/**
 * @brief converts a fixed point number into a floating point number
 *
 * @param fixed16 value
 * @return float value
 */
inline fp_t fixed16_to_float(fixed16_t fixed) {
    fp_t temp;

    // negative
    if((fixed & (1 << (FIXED_LENGTH-1))) >> (FIXED_LENGTH-1) == 1) {
        fixed = ~fixed + 1;
        uint16_t before_comma = fixed >> DECIMAL_PLACES;
        uint16_t after_comma = fixed & AFTER_COMMA_MASK;
        temp = -1*(before_comma + (after_comma * PRECISION));
    } else {
        uint16_t before_comma = fixed >> DECIMAL_PLACES;
        uint16_t after_comma = fixed & AFTER_COMMA_MASK;
        temp = (before_comma + (after_comma * PRECISION));
    }

    return temp;
}

/**
 * @brief converts an integer to a fixed point number
 *
 * @param int16_t value
 * @return fixed16 value
 */
inline fixed16_t int_to_fixed16(const int16_t a) {
    return a * FIXED_ONE; 
}
/**
 * @brief converts a fixed point number to an integer number
 *
 * @param int16_t value
 * @return fixed16 value
 */

inline int16_t fixed16_to_int16(const fixed16_t a) {
    return a >> DECIMAL_PLACES;
}

/**
 * @brief adds two fixed point values
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a+b
 */
inline fixed16_t add_fixed16(const fixed16_t a, const fixed16_t b) {
	/*
    uint16_t _a = a;
    uint16_t _b = b;
    uint16_t sum = _a + _b;
    return sum;
	*/
	return a + b;
}

/**
 * @brief subtracts two fixed point numbers
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a-b
 */
inline fixed16_t sub_fixed16(const fixed16_t a, const fixed16_t b) {
	/*
    uint16_t _a = a;
    uint16_t _b = b;
    uint16_t diff = _a - _b;
    return diff;
	*/
	return a - b;
}

/**
 * @brief multiplies two fixed point values
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a*b
 */
inline fixed16_t mul_fixed16(const fixed16_t a, const fixed16_t b) {
	int32_t prod = (a * b) >> DECIMAL_PLACES;
	return (int16_t) prod;
}

/**
 * @brief divides two fixed point numbers
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a/b
 */
inline fixed16_t div_fixed16(const fixed16_t a, const fixed16_t b) {
	int32_t quot = ((int32_t)a * (1 << DECIMAL_PLACES)) / b;
	return (int16_t) quot;
}

/**
 * @brief calculates exp(x) of a fixed point number x
 *
 * @param x
 * @return exp(x)
 */
fixed16_t exp_fixed16(fixed16_t x) {

    if(x == 0) return FIXED_ONE; // exp(0) = 1
    if(x == FIXED_ONE) return 0x0adf; // exp(1) = e
    if(x >= 3548) return 0x7fff; // exp(x > 3.4657) = 31.999023
    if(x <= -3553) return 0x0000; // exp(x < -3.4657) = 0

	uint8_t neg = (x < 0);

	if(neg) {
		x = -x;
	}

	fixed16_t result = x + FIXED_ONE;
	fixed16_t term = x;

	uint_fast8_t i;
	for(i = 2; i < 30; i++) {
		term = mul_fixed16(term, div_fixed16(x, int_to_fixed16(i)));
		result += term;

		if((term < 500) && ((i > 15) || (term < 20))) {
			break;
		}
	}

	if(neg) {
		result = div_fixed16(FIXED_ONE, result);
	}

	return result;
}

/**
 * @brief calculates exp(x) of a int32 number x
 *
 * @param x
 * @return exp(x)
 */
inline int32_t exp_int32(const int32_t x) {
    if(x < 0) return 0;
    if(x > 21) return INT_MAX; 

    int32_t values[] = {
        1,          // 0
        3,          // 1
        7,          // 2
        20,         // 3
        54,         // 4
        148,        // 5
        403,        // 6
        1097,       // 7
        2981,       // 8
        8103,       // 9
        22026,      // 10
        59874,      // 11
        162755,     // 12
        442413,     // 13
        1202604,    // 14
        3269017,    // 15
        8886110,    // 16
        24154952,   // 17
        65659969,   // 18
        178482300,  // 19
        485165195,  // 20
        1318815734  // 21
    };

    return values[x];
}

/**
 * @brief calculates exp(x) of a int16 number x
 *
 * @param x
 * @return exp(x)
 */
inline int16_t exp_int16(const int16_t x) {
    if(x < 0) return 0;
    if(x > 10) return SHRT_MAX; 

    int32_t values[] = {
        1,          // 0
        3,          // 1
        7,          // 2
        20,         // 3
        54,         // 4
        148,        // 5
        403,        // 6
        1097,       // 7
        2981,       // 8
        8103,       // 9
        22026       // 10
    };

    return values[x];
}

#ifdef __aarch64__
/**
 * @brief fixed16x4_t vector multiplication
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @return a*b
 */
fixed16x4_t vmul_fixed16(const fixed16x4_t a, const fixed16x4_t b) {
	int32x4_t prod;
   	prod = vmull_s16(a, b);
	prod = vshrq_n_s32(prod, DECIMAL_PLACES);
	return vmovn_s32(prod);
}

/**
 * @brief fixed16x4_t vector multiply-accumulate
 *
 * @assert no overflow handling
 * @param a
 * @param b
 * @param c
 * @return b*c+a
 */
fixed16x4_t vmla_fixed16(const fixed16x4_t a, const fixed16x4_t b, const fixed16x4_t c) {
	int16x4_t prod = vmul_fixed16(b, c);
	return vadd_s16(prod, a);
}
#endif

#endif /* SRC_FIXED16_H_ */
