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

// how many bits are used to store a fixed point number
#define FIXED_LENGTH 16

// how many bits are used to store the values after the comma
#define DECIMAL_PLACES 10

// minimal fraction of what can be represented by the fixed point number
#define PRECISION (1.0 / (1 << DECIMAL_PLACES))

// 1 as fixed point number
#define FIXED_ONE (0 | (1 << DECIMAL_PLACES))

// type in which fixed point numbers will be stored
typedef int16_t fixed16_t;


// converts a floating point number into a fixed point number
fixed16_t float_to_fixed16(fp_t float_value) {
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

// converts a fixed point number into a floating point number
fp_t fixed16_to_float(fixed16_t fixed) {
    fp_t temp;

    // negative
    if((fixed & (1 << (FIXED_LENGTH-1))) >> (FIXED_LENGTH-1) == 1) {
        fixed = ~fixed + 1;
        uint16_t before_comma = fixed >> DECIMAL_PLACES;
        uint16_t after_comma = fixed & 0x3FF;
        temp = -1*(before_comma + (after_comma * PRECISION));
    } else {
        uint16_t before_comma = fixed >> DECIMAL_PLACES;
        uint16_t after_comma = fixed & 0x3FF;
        temp = (before_comma + (after_comma * PRECISION));
    }

    return temp;
}

// converts an integer to a fixed point number
fixed16_t int_to_fixed16(int a) {
    return a * FIXED_ONE; 
}

/**
 * no overflow handling
 */
fixed16_t add_fixed16(fixed16_t a, fixed16_t b) {
    uint16_t _a = a;
    uint16_t _b = b;
    uint16_t sum = _a + _b;
    return sum;
}

/**
 * no overflow handling
 */
fixed16_t sub_fixed16(fixed16_t a, fixed16_t b) {
    uint16_t _a = a;
    uint16_t _b = b;
    uint16_t diff = _a - _b;
    return diff;
}

/**
 * no overflow handling
 */
fixed16_t mul_fixed16(fixed16_t a, fixed16_t b) {
	int32_t prod = ((a * b) & ~(1 << 31)) >> DECIMAL_PLACES;
	return (int16_t) prod;
}

/**
 * no overflow handling
 */
fixed16_t div_fixed16(fixed16_t a, fixed16_t b) {
	int32_t quot = ((int32_t)a * (1 << DECIMAL_PLACES)) / b;
	return (int16_t) quot;
}

fixed16_t exp_fixed16(fixed16_t x) {

    if(x == 0) return FIXED_ONE; // exp(0) = 1
    if(x == FIXED_ONE) return 0x0adf; // exp(1) = e
    if(x >= 3548) return 0x7fff; // exp(x > 3.4657) = 31.999023
    if(x <= -7680) return 0x0000; // exp(x < -32.0) = 0

	uint8_t neg = (x < 0);

	if(neg) {
		x = x*-1;
	}

	fixed16_t result = x + FIXED_ONE;
	fixed16_t term = x;

	uint_fast8_t i;
	for(i = 2; i < 30; i++) {
		term = mul_fixed16(term, div_fixed16(x, int_to_fixed16(i)));
		result = add_fixed16(result, term);

		if((term < 500) && ((i > 15) || (term < 20))) {
			break;
		}
	}

	if(neg) {
		result = div_fixed16(FIXED_ONE, result);
	}

	return result;
}

#endif /* SRC_FIXED16_H_ */
