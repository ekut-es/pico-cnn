/** 
 * @brief provides global parameters for pico-cnn 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define VERSION 1.0

#define EPSILON 0.0001

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef INFO
#define INFO 1
#endif

typedef float fp_t;
extern fp_t max_float;
extern fp_t min_float;

#define PRINT_DEBUG(x) do { if(DEBUG) { std::cout << "[DEBUG] " << __FILE__ << ":" << __LINE__ << ":" << __func__ << "() " << x << std::endl; } } while (0);

#define PRINT_ERROR(x) do { std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << ":" << __func__ << "() " << x << std::endl; } while (0);

#define PRINT_ERROR_AND_DIE(x) do { std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << ":" << __func__ << "() " << x << std::endl; std::exit(1); } while (0);

#define PRINT_INFO(x) do { if(INFO) { std::cout << "[INFO] " <<  x << std::endl; } } while (0);

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define FLOAT_MIN -100000

namespace pico_cnn {
    enum class DataType{
        F16,
        F32,
        F64
    };
}

#endif // PARAMETERS_H
