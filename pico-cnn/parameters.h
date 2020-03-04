/** 
 * @brief provides global parameters for pico-cnn 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define VERSION 1.0

typedef float fp_t;
extern fp_t max_float;
extern fp_t min_float;

#ifdef DEBUG
#define DEBUG_MSG(...) fprintf( stderr, __VA_ARGS__ );
#else
#define DEBUG_MSG(...) do{ } while ( 0 );
#endif

#ifdef INFO
#define INFO_MSG(...) fprintf( stdout, __VA_ARGS__ );
#else
#define INFO_MSG(...) do{ } while ( 0 );
#endif

#define ERROR_MSG(...) fprintf( stderr, __VA_ARGS__ );

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define FLOAT_MIN -100000

#endif // PARAMETERS_H
