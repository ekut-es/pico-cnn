/**
 * @brief contains all activation functions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "../parameters.h"
#include <stdint.h>
#include <math.h>

/**
 * @brief Performs clip operation on the input_channel. All values smaller than min will be set to min.
 *        All values bigger than max will be set to max. All other values stay the same.
 *
 * @param input_channel
 * @param height
 * @param width
 * @param min
 * @param max
 * @param output_channel
 */
void clip_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                const fp_t min, const fp_t max, fp_t* output_channel);

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
 * @brief applies leaky ReLU to all pixel of input channel and stores it in
 * output_channel,
 * with leakyReLU(x) = x * leak   if x <  0
 *                     x          if x >= 0
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param leak
 */
void leaky_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t leak);


/**
 * @brief applies parametrized ReLU to all pixel of input channel and stores it in
 * output_channel. There is a (learnable) parameter for each input channel.
 * PRReLU(x_i) = x_i * a_i   if x_i <  0
 *               x_i         if x_i >= 0
 *  for x_i,the input channel and a_i, the parameter
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param kernel parameters
 */
void parametrized_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, fp_t* kernel);


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
void sigmoid_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,  fp_t* output_channel);

/**
 * @brief applies leaky ReLU to all pixel of input channel and stores it in
 * output_channel,
 * with leakyReLU(x) = x * leak   if x <  0
 *                     x          if x >= 0
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param leak
 */
void leaky_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t leak);


/**
 * @brief applies parametrized ReLU to all pixel of input channel and stores it in
 * output_channel. There is a (learnable) parameter for each input channel.
 * PRReLU(x_i) = x_i * a_i   if x_i <  0
 *               x_i         if x_i >= 0
 *  for x_i,the input channel and a_i, the parameter
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param kernel parameters
 */
void parametrized_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, fp_t* kernel);


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
void sigmoid_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,  fp_t* output_channel);

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

#endif // ACTIVATION_FUNCTION_H
