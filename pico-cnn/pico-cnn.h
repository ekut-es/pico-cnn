/** 
 * @brief provides all includes to construct a CNN
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_H
#define PICO_CNN_H

#include "parameters.h"

#include "layers/activation_function.h"
#include "layers/convolution.h"
#include "layers/pooling.h"
#include "layers/fully_connected.h"
#include "layers/local_response_normalization.h"

#include "io/read_weights.h"
#include "io/read_pgm.h"

#ifdef MNIST
#include "io/read_mnist.h"
#endif

#ifdef JPEG
#include "io/read_jpeg.h"
#endif

#ifdef IMAGENET
#include "io/read_means.h"
#include "io/read_imagenet_labels.h"
#endif

#ifdef DEBUG
#include "io/write_pgm.h"
#include "io/write_float.h"
#endif

#endif // PICO_CNN_H
