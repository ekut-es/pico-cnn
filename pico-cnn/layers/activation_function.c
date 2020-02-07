#include "activation_function.h"

void tanh_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    for(i = 0; i < height*width; i++) {
        output_channel[i] = tanhf(input_channel[i]);
    }
}

void relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    for(i = 0; i < height*width; i++) {
        output_channel[i] = (input_channel[i] < 0.0) ? 0.0 : input_channel[i];
    }
}

void leaky_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t leak) {

  #ifdef BIG_LOOPS
 uint64_t i;
  # else
  uint32_t i;
  #endif

  for(i = 0; i < height*width; i++) {
    output_channel[i] = (input_channel[i] < 0.0) ? (leak*input_channel[i]) : input_channel[i];
  }
}


// TODO: check whether kernel size (multiple parameters) makes sense
// TODO: rename kernel?
void parametrized_relu_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, fp_t* kernel) {

  #ifdef BIG_LOOPS
 uint64_t i;
  # else
  uint32_t i;
  #endif

  for(i = 0; i < height*width; i++) {
    output_channel[i] = (input_channel[i] < 0.0) ? (kernel[i] * input_channel[i]) : input_channel[i];
  }

}


// TODO: check whether correct
void sigmoid_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,  fp_t* output_channel) {

  #ifdef BIG_LOOPS
  uint64_t i;
  #else
  uint32_t i;
  #endif

  for(i = 0; i < height*width; i++) {
    output_channel[i] = 1 / (1+expf(- input_channel[i]));
    // alternative formula:
    //  output_channel[i] = 0.5 * (1 + tanhf(input_channel[i] / 2));
  }
}

void softmax_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    double_t denominator = 0.0;

    for(i = 0; i < height*width; i++) {
        denominator += exp((double_t)input_channel[i]);
    }

    for(i = 0; i < height*width; i++) {
        output_channel[i] = (fp_t)(exp((double_t)input_channel[i]) / denominator);
    }
}

void local_response_normalization_naive(fp_t** input_channels, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** output_channels, const fp_t alpha, const fp_t beta, const uint16_t n) {

    int32_t channel, row, column, i;
    int32_t from;
    int32_t to;

    fp_t sum;

    for(channel = 0; channel < depth; channel++) {
        from = MAX(0,channel-(n/2));
        to = MIN(depth-1,channel+(n/2));

        for(row = 0; row < height; row++) {
            for(column = 0; column < width; column++) {

                sum = 0.0;

                for(i = from; i <= to; i++) {
                    sum += powf(input_channels[i][row*width+column], 2);
                }

                output_channels[channel][row*width+column] = input_channels[channel][row*width+column] / powf((1+(alpha/n)*sum),beta);
            }
        }
    }
}

#ifdef FIXED16

void relu_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    for(i = 0; i < height*width; i++) {
        output_channel[i] = ((input_channel[i] & 0x8000) == 0x8000) ? 0 : input_channel[i];
    }
}

void relu_cpu_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel) {
    relu_naive_fixed16(input_channel, height, width, output_channel);
}

void softmax_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    fixed16_t denominator = FIXED_ZERO;

    for(i = 0; i < height*width; i++) {
        denominator += exp_int32(fixed16_to_int16(input_channel[i]));
    }

    for(i = 0; i < height*width; i++) {
        output_channel[i] = div_fixed16(exp_int32(fixed16_to_int16(input_channel[i])), denominator);
    }
}
#endif // FIXED-16

#ifdef ARM_NEON

void relu_cpu(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    float32x4_t input_channel_0;
    float32x4_t input_channel_1;
    float32x4_t input_channel_2;
    float32x4_t input_channel_3;

    float32x4_t output_channel_0;
    float32x4_t output_channel_1;
    float32x4_t output_channel_2;
    float32x4_t output_channel_3;

    float32x4_t zero = {0.0, 0.0, 0.0, 0.0};

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {

        // load input_channel into vectors
        input_channel_0 = vld1q_f32(input_channel+i);
        input_channel_1 = vld1q_f32(input_channel+i+4);
        input_channel_2 = vld1q_f32(input_channel+i+8);
        input_channel_3 = vld1q_f32(input_channel+i+12);

        output_channel_0 = vmaxq_f32(input_channel_0, zero);
        output_channel_1 = vmaxq_f32(input_channel_1, zero);
        output_channel_2 = vmaxq_f32(input_channel_2, zero);
        output_channel_3 = vmaxq_f32(input_channel_3, zero);

        vst1q_f32(output_channel+i, output_channel_0);
        vst1q_f32(output_channel+i+4, output_channel_1);
        vst1q_f32(output_channel+i+8, output_channel_2);
        vst1q_f32(output_channel+i+12, output_channel_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        output_channel[i] = (input_channel[i] < 0.0) ? 0.0 : input_channel[i];
    }
}

void softmax_cpu_single(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    fp_t denominator = 0.0;

    float32x4_t input_channel_0;
    float32x4_t input_channel_1;
    float32x4_t input_channel_2;
    float32x4_t input_channel_3;

    // calculate denominator
    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {
        // load input channel into vectors
        input_channel_0 = vld1q_f32(input_channel+i);
        input_channel_1 = vld1q_f32(input_channel+i+4);
        input_channel_2 = vld1q_f32(input_channel+i+8);
        input_channel_3 = vld1q_f32(input_channel+i+12);

        // apply exponential function to vectors
        input_channel_0 = exp_ps(input_channel_0);
        input_channel_1 = exp_ps(input_channel_1);
        input_channel_2 = exp_ps(input_channel_2);
        input_channel_3 = exp_ps(input_channel_3);

        // add vectors together
        input_channel_1 = vaddq_f32(input_channel_1, input_channel_0);
        input_channel_2 = vaddq_f32(input_channel_2, input_channel_1);
        input_channel_3 = vaddq_f32(input_channel_3, input_channel_2);

        // sum up whole vector
        denominator += vgetq_lane_f32(input_channel_3, 0) + vgetq_lane_f32(input_channel_3, 1) + vgetq_lane_f32(input_channel_3, 2) + vgetq_lane_f32(input_channel_3, 3); //vaddvq_f32(input_channel_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        denominator += expf(input_channel[i]);
    }

    const fp_t inv_denominator = 1.0/denominator;
    // apply softmax
    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {
        // load input channel into vectors
        input_channel_0 = vld1q_f32(input_channel+i);
        input_channel_1 = vld1q_f32(input_channel+i+4);
        input_channel_2 = vld1q_f32(input_channel+i+8);
        input_channel_3 = vld1q_f32(input_channel+i+12);

        // apply exponential function to vectors
        input_channel_0 = exp_ps(input_channel_0);
        input_channel_1 = exp_ps(input_channel_1);
        input_channel_2 = exp_ps(input_channel_2);
        input_channel_3 = exp_ps(input_channel_3);

        // multiply vectors scalar with inverted denominator
        input_channel_0 = vmulq_n_f32(input_channel_0, inv_denominator);
        input_channel_1 = vmulq_n_f32(input_channel_1, inv_denominator);
        input_channel_2 = vmulq_n_f32(input_channel_2, inv_denominator);
        input_channel_3 = vmulq_n_f32(input_channel_3, inv_denominator);

        // store vectors in output channel
        vst1q_f32(output_channel+i, input_channel_0);
        vst1q_f32(output_channel+i+4, input_channel_1);
        vst1q_f32(output_channel+i+8, input_channel_2);
        vst1q_f32(output_channel+i+12, input_channel_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        output_channel[i] = expf(input_channel[i]) * inv_denominator;
    }
}

void local_response_normalization_cpu_single(fp_t** input_channel, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** output_channel, const fp_t alpha, const fp_t beta, const uint16_t n) {

    #ifdef BIG_LOOPS
    uint64_t i;
    #else
    uint32_t i;
    #endif

    int32_t channel, row, column;
    int32_t from;
    int32_t to;

    fp_t sum_0;
    fp_t sum_1;
    fp_t sum_2;
    fp_t sum_3;

    float32x4_t denominator_0;
    float32x4_t sums_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t input_channel_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t output_channel_0;

    float32x4_t one = {1.0, 1.0, 1.0, 1.0};

    fp_t denominator_temp[4];
    fp_t output_channel_temp[4];

    for(channel = 0; channel < depth; channel++) {
        from = MAX(0,channel-(n/2));
        to = MIN(depth-1,channel+(n/2));

        for(row = 0; row < height; row++) {
            for(column = 0; column < width-4; column+=4) {

                sum_0 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_0 += input_channel[i][row*width+column]*input_channel[i][row*width+column];
                }

                sum_1 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_1 += input_channel[i][row*width+column+1]*input_channel[i][row*width+column+1];
                }

                sum_2 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_2 += input_channel[i][row*width+column+2]*input_channel[i][row*width+column+2];
                }

                sum_3 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_3 += input_channel[i][row*width+column+3]*input_channel[i][row*width+column+3];
                }


                const fp_t alpha_n = alpha/n;

                // load vector with sums
                sums_0 = vsetq_lane_f32(sum_0, sums_0, 0);
                sums_0 = vsetq_lane_f32(sum_1, sums_0, 1);
                sums_0 = vsetq_lane_f32(sum_2, sums_0, 2);
                sums_0 = vsetq_lane_f32(sum_3, sums_0, 3);

                // sums multiply with alpha/n
                denominator_0 = vmulq_n_f32(sums_0, alpha_n);
                // add 1
                denominator_0 = vaddq_f32(denominator_0, one);
                // store denominator vector in array
                vst1q_f32(denominator_temp, denominator_0);

                denominator_temp[0] = powf(denominator_temp[0],beta);
                denominator_temp[1] = powf(denominator_temp[1],beta);
                denominator_temp[2] = powf(denominator_temp[2],beta);
                denominator_temp[3] = powf(denominator_temp[3],beta);

                // load array back into vector
                denominator_0 = vld1q_f32(denominator_temp);

                // denominator = 1/denomniator
                denominator_0 = vrecpeq_f32(denominator_0);

                // store denominator vector in array
                vst1q_f32(denominator_temp, denominator_0);

                // load original input channel into vector
                input_channel_0 = vsetq_lane_f32(input_channel[channel][row*width+column],   input_channel_0, 0);
                input_channel_0 = vsetq_lane_f32(input_channel[channel][row*width+column+1], input_channel_0, 1);
                input_channel_0 = vsetq_lane_f32(input_channel[channel][row*width+column+2], input_channel_0, 2);
                input_channel_0 = vsetq_lane_f32(input_channel[channel][row*width+column+3], input_channel_0, 3);

                // output_channel = input_channel * (1/denominator)
                output_channel_0 = vmulq_f32(input_channel_0, denominator_0);

                // store output_channel vector into array
                vst1q_f32(output_channel_temp, output_channel_0);

                output_channel[channel][row*width+column] =   output_channel_temp[0];
                output_channel[channel][row*width+column+1] = output_channel_temp[1];
                output_channel[channel][row*width+column+2] = output_channel_temp[2];
                output_channel[channel][row*width+column+3] = output_channel_temp[3];
            }

            // residual columns
            for(column = column; column < width; column++) {
                sum_0 = 0.0;

                for(i = from; i <= to; i++) {
                    sum_0 += input_channel[i][row*width+column]*input_channel[i][row*width+column];
                }

                output_channel[channel][row*width+column] = input_channel[channel][row*width+column] / powf((1+(alpha/n)*sum_0),beta);
            }
        }
    }
}

#endif //ARM-NEON
