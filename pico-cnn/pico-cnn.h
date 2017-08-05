/** 
 * @brief provides all includes to construct a CNN
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include "parameters.h"

#include "layers/activation_function.h"
#include "layers/convolution.h"
#include "layers/pooling.h"
#include "layers/fully_connected.h"

#include "io/read_mnist.h"
#include "io/write_pgm.h"
#include "io/write_float.h"
