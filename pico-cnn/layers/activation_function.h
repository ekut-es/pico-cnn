/**
 * @brief contains all activation functions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif

#ifdef ARM_NEON
#include "arm_neon.h"
#include "../driver/neon_mathfun.h"
#endif

/**
 * @brief applies tanh(x) to all pixel of input_channel and stores it in
 * output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void tanh_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);


/**
 * @brief applies relu(x) to all pixel of input_channel and stores it in
 * output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);


/**
  * @brief applies a special sigmoid function, the logistic function 
  *  to all values of input_channel and stores it in
  *  output_channel
  *
  * @param input_channel (height x width)
  * @param height
  * @param width
  * @param output_channel (height x width)
*/
void sigmoid_naive(const fp* input_channel, const uint16_t height, const uint16_t width,  fp_t* output_channel);

/**
 * @brief applies softmax to all pixel of input_channel and stores it in
 * output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void softmax_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);

/**
 * @brief performs a local response normalization (across channels) on
 * input_channel and stores the result in output_channel
 *
 * Formula (Paper):
 * https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn/252343#252343
 * Formula (Implemented):
 * http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param depth
 * @param output_channel
 * @param alpha
 * @param beta
 * @param n
 */
void local_response_normalization_naive(fp_t** input_channels, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** output_channels, const fp_t alpha, const fp_t beta, const uint16_t n);


#ifdef FIXED16
/**
 * @brief applies relu(x) to all pixel of input_channel and stores it in
 * output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void relu_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel);

/**
 * @brief applies softmax to all pixel of input_channel and stores it in
 * output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void softmax_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel);

#endif // FIXED-16

#ifdef ARM_NEON
/**
 * @brief applies relu(x) to all pixel of input_channel and stores it in
 * output_channel optimzed of CPU
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void relu_cpu(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);

/**
 * @brief applies softmax to all pixel of input_channel and stores it in
 * output_channel optimzed of CPU
 * Only single core optimization since softmax is usually performed on a small
 * dataset and a multi core solution would impose a large overhead
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 */
void softmax_cpu_single(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);

/**
 * @brief performs a local response normalization (across channels) on
 * input_channel and stores the result in output_channel optimized for single
 * CPU
 *
 * Formula (Paper):
 * https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn/252343#252343
 * Formula (Implemented):
 * http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param depth
 * @param output_channel
 * @param alpha
 * @param beta
 * @param n
 */
void local_response_normalization_cpu_single(fp_t** input_channel, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** output_channel, const fp_t alpha, const fp_t beta, const uint16_t n);

#endif //ARM-NEON


#endif // ACTIVATION_FUNCTION_H
