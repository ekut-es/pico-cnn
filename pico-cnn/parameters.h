/** 
 * @brief provides global parameters for pico-cnn 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

typedef float fp_t;
extern fp_t max_float;
extern fp_t min_float;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#ifdef __aarch64__
// L1 cache line size 64 Byte
#define BLOCK_SIZE 64
#endif

#endif // PARAMETERS_H
