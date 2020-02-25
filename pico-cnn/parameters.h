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

#define FLOAT_MIN -100000

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#endif // PARAMETERS_H
