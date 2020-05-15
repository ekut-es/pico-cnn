/** 
 * @brief provides all includes to construct a CNN
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_H
#define PICO_CNN_H

#include "parameters.h"
#include "utils.h"

#include "layers/activation_functions/activation_function.h"
#include "layers/activation_functions/clip.h"
#include "layers/activation_functions/lrn.h"
#include "layers/activation_functions/relu.h"
#include "layers/activation_functions/sigmoid.h"
#include "layers/activation_functions/softmax.h"
#include "layers/activation_functions/tan_h.h"

#include "layers/convolution.h"
#include "layers/pooling/pooling.h"
#include "layers/pooling/max_pooling.h"
#include "layers/pooling/average_pooling.h"
#include "layers/pooling/global_max_pooling.h"
#include "layers/pooling/global_average_pooling.h"
#include "layers/fully_connected.h"
#include "layers/batch_normalization.h"
//#include "layers/concatenate.h"

#include "io/read_binary_weights.h"
#include "io/read_binary_reference_data.h"
//#include "io/read_pgm.h"

#ifdef MNIST
#include "io/read_mnist.h"
#ifdef DEBUG
#include "io/write_pgm.h"
#include "io/write_float.h"
#endif
#endif

#ifdef JPEG
#include "io/read_jpeg.h"
#endif

#ifdef IMAGENET
#include "io/read_means.h"
#include "io/read_imagenet_labels.h"
#include "io/read_imagenet_validation_labels.h"
#endif

#ifdef DEBUG
#include <stdio.h>
#endif

#endif // PICO_CNN_H
