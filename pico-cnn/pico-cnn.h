/** 
 * @brief provides all includes to construct a CNN
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_H
#define PICO_CNN_H

#include "parameters.h"

#include "layers/activation_function.h"
#include "layers/convolution.h"
#include "layers/pooling.h"
#include "layers/fully_connected.h"
#include "layers/batch_normalization.h"
#include "layers/concatenate.h"

#include "io/read_binary_weights.h"
#include "io/read_binary_reference_data.h"
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
#include "io/read_imagenet_validation_labels.h"
#endif

#ifdef DEBUG
#include <stdio.h>
#endif

#endif // PICO_CNN_H
