#include <assert.h>
#include "convolution.h"

void convolution1d_naive(const fp_t* input_channel, const int input_size, fp_t* output_channel, const fp_t* kernel,
                         const int kernel_size, const int stride, const int padding, const fp_t bias) {

    int32_t input_channel_idx;
    int32_t kernel_idx;
    int32_t crop = kernel_size/2;
    int32_t output_channel_idx, output_channel_size;

    fp_t pixel;

    output_channel_idx = 0;

    // padding valid
    if(padding == 0) {
        output_channel_size = ((input_size-kernel_size)/stride)+1;

        for(input_channel_idx = crop; input_channel_idx < input_size-crop; input_channel_idx+=stride) {
            pixel = 0.0;

            for(kernel_idx = 0; kernel_idx < kernel_size; kernel_idx++) {
                pixel += kernel[kernel_idx] * input_channel[input_channel_idx-crop+kernel_idx];
            }

            pixel += bias;

            output_channel[output_channel_idx] = pixel;
            output_channel_idx++;
        }
//        assert(output_channel_idx == output_channel_size);
    }

    // padding same
    else if(padding == kernel_size/2) {
        output_channel_size = (((input_size+2*(kernel_size/2))-kernel_size)/stride)+1;

        for(input_channel_idx = 0; input_channel_idx < input_size; input_channel_idx+=stride) {
            pixel = 0.0;
            for(kernel_idx = -padding; kernel_idx <= padding; kernel_idx++) {
                if((input_channel_idx+kernel_idx) < 0 || (input_channel_idx+kernel_idx) > input_size-1) {
                    pixel += 0.0;
                } else {
                    pixel += kernel[kernel_idx+padding] * input_channel[input_channel_idx+kernel_idx];
                }
            }
            pixel += bias;

            output_channel[output_channel_idx] = pixel;
            output_channel_idx++;
        }
//        assert(output_channel_idx == output_channel_size);
    }
}

void convolution2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                         const fp_t* kernel, const uint16_t kernel_size, const uint16_t stride,
                         const uint16_t padding, const fp_t bias) {

    int32_t channel_row, channel_column;
    int32_t kernel_row, kernel_column;
    int32_t crop = kernel_size/2;

    int32_t output_channel_row, output_channel_column, output_channel_width;

    fp_t pixel;

    output_channel_row = 0;
    output_channel_column = 0;

    // padding valid
    if(padding == 0) {
        output_channel_width = (width-kernel_size)/stride+1;

        for(channel_row = crop; channel_row < height-crop; channel_row+=stride) {
            for(channel_column = crop; channel_column < width-crop; channel_column+=stride) {
                pixel = 0.0;

                for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                        pixel += kernel[kernel_row*kernel_size+kernel_column] * input_channel[(channel_row-crop+kernel_row)*width+(channel_column-crop+kernel_column)];
                    }
                }

                pixel += bias;

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {
        output_channel_width = (width+2*(kernel_size/2)-kernel_size)/stride+1;

        for(channel_row = 0; channel_row < height; channel_row+=stride) {
            for(channel_column = 0; channel_column < width; channel_column+=stride) {
                pixel = 0.0;

                for(kernel_row = -padding; kernel_row <= padding; kernel_row++) {
                    for(kernel_column = -padding; kernel_column <= padding; kernel_column++) {
                        if((channel_row+kernel_row) < 0 || (channel_row+kernel_row) > height-1 || (channel_column+kernel_column) < 0 || (channel_column+kernel_column) > width-1) {
                            pixel += 0.0;
                        } else {
                            pixel += kernel[(kernel_row+padding)*kernel_size+(kernel_column+padding)] * input_channel[(channel_row+kernel_row)*width+(channel_column+kernel_column)];
                        }
                    }
                }

                pixel += bias;

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }
}

void add_channel2d_naive(fp_t* channel_a, const fp_t* channel_b, const uint16_t height, const uint16_t width) {
    uint32_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            channel_a[row*width+column] = (channel_a[row*width+column] + channel_b[row*width+column]);
        }
    }
}

#ifdef ARM_NEON
void convolution2d_cpu_3x3_s1_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias) {

    /*
     3 float_32x4 neon registers for kernel (last value will be ignored)
     kernel_0 = [0,0][0,1][0,2][x]
     kernel_1 = [1,0][1,1][1,2][x]
     kernel_2 = [2,0][2,1][2,2][x]

     5x 3 float_32x4 neon registers for channel (last value will be ignored)
     channel_*_0 = [0,0][0,1][0,2][x]  [0,1][0,2][0,3][x]  [0,2][0,3][0,4][x]  [0,3][0,4][0,5][x]  [0,4][0,5][0,6][x]
     channel_*_1 = [1,0][1,1][1,2][x]  [1,1][1,2][1,3][x]  [1,2][1,3][1,4][x]  [1,3][1,4][1,5][x]  [1,4][1,5][1,6][x]
     channel_*_2 = [2,0][2,1][2,2][x]  [2,1][2,2][2,3][x]  [2,2][2,3][2,4][x]  [2,3][2,4][2,5][x]  [2,4][2,5][2,6][x]

     channel_*_0 = channel_*_0 * kernel_0
     channel_*_1 = channel_*_0 + channel_*_1 * kernel_0
     channel_*_2 = channel_*_1 + channel_*_2 * kernel_0
     */

    uint16_t channel_row, channel_column;
    const uint8_t padding = 1;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+3);
    kernel_2 = vld1q_f32(kernel+6);

    // vectors for channel
    float32x4_t channel_0_0;
    float32x4_t channel_1_0;
    float32x4_t channel_2_0;

    float32x4_t channel_0_1;
    float32x4_t channel_1_1;
    float32x4_t channel_2_1;

    float32x4_t channel_0_2;
    float32x4_t channel_1_2;
    float32x4_t channel_2_2;

    float32x4_t channel_0_3;
    float32x4_t channel_1_3;
    float32x4_t channel_2_3;

    float32x4_t channel_0_4;
    float32x4_t channel_1_4;
    float32x4_t channel_2_4;

    for(channel_row = 0; channel_row < height-2; channel_row++) {
        for(channel_column = 0; channel_column < width-padding-5; channel_column+=5) {

            // load channel into vectors
            const uint32_t source_0 = (channel_row+0)*width+channel_column;
            const uint32_t source_1 = (channel_row+1)*width+channel_column;
            const uint32_t source_2 = (channel_row+2)*width+channel_column;

            channel_0_0 = vld1q_f32(input_channel+source_0);
            channel_1_0 = vld1q_f32(input_channel+source_1);
            channel_2_0 = vld1q_f32(input_channel+source_2);

            channel_0_1 = vld1q_f32(input_channel+source_0+1);
            channel_1_1 = vld1q_f32(input_channel+source_1+1);
            channel_2_1 = vld1q_f32(input_channel+source_2+1);

            channel_0_2 = vld1q_f32(input_channel+source_0+2);
            channel_1_2 = vld1q_f32(input_channel+source_1+2);
            channel_2_2 = vld1q_f32(input_channel+source_2+2);

            channel_0_3 = vld1q_f32(input_channel+source_0+3);
            channel_1_3 = vld1q_f32(input_channel+source_1+3);
            channel_2_3 = vld1q_f32(input_channel+source_2+3);

            channel_0_4 = vld1q_f32(input_channel+source_0+4);
            channel_1_4 = vld1q_f32(input_channel+source_1+4);
            channel_2_4 = vld1q_f32(input_channel+source_2+4);

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);
            channel_2_0 = vmlaq_f32(channel_1_0, channel_2_0, kernel_2);

            channel_0_1 = vmulq_f32(channel_0_1, kernel_0);
            channel_1_1 = vmlaq_f32(channel_0_1, channel_1_1, kernel_1);
            channel_2_1 = vmlaq_f32(channel_1_1, channel_2_1, kernel_2);

            channel_0_2 = vmulq_f32(channel_0_2, kernel_0);
            channel_1_2 = vmlaq_f32(channel_0_2, channel_1_2, kernel_1);
            channel_2_2 = vmlaq_f32(channel_1_2, channel_2_2, kernel_2);

            channel_0_3 = vmulq_f32(channel_0_3, kernel_0);
            channel_1_3 = vmlaq_f32(channel_0_3, channel_1_3, kernel_1);
            channel_2_3 = vmlaq_f32(channel_1_3, channel_2_3, kernel_2);

            channel_0_4 = vmulq_f32(channel_0_4, kernel_0);
            channel_1_4 = vmlaq_f32(channel_0_4, channel_1_4, kernel_1);
            channel_2_4 = vmlaq_f32(channel_1_4, channel_2_4, kernel_2);

            // store new channel
            const uint32_t target = channel_row*(width-2*padding)+channel_column;

            // sum up whole vector (with bias)
            output_channel[target] =   vgetq_lane_f32(channel_2_0, 0) + vgetq_lane_f32(channel_2_0, 1) + vgetq_lane_f32(channel_2_0, 2) + bias;
            output_channel[target+1] = vgetq_lane_f32(channel_2_1, 0) + vgetq_lane_f32(channel_2_1, 1) + vgetq_lane_f32(channel_2_1, 2) + bias;
            output_channel[target+2] = vgetq_lane_f32(channel_2_2, 0) + vgetq_lane_f32(channel_2_2, 1) + vgetq_lane_f32(channel_2_2, 2) + bias;
            output_channel[target+3] = vgetq_lane_f32(channel_2_3, 0) + vgetq_lane_f32(channel_2_3, 1) + vgetq_lane_f32(channel_2_3, 2) + bias;
            output_channel[target+4] = vgetq_lane_f32(channel_2_4, 0) + vgetq_lane_f32(channel_2_4, 1) + vgetq_lane_f32(channel_2_4, 2) + bias;

        }

        // residual columns
        for(channel_column = channel_column; channel_column < width-2; channel_column++) {

            // load channel into vectors
            const uint32_t source_0 = (channel_row+0)*width+channel_column;
            const uint32_t source_1 = (channel_row+1)*width+channel_column;
            const uint32_t source_2 = (channel_row+2)*width+channel_column;

            channel_0_0 = vld1q_f32(input_channel+source_0);
            channel_1_0 = vld1q_f32(input_channel+source_1);
            channel_2_0 = vld1q_f32(input_channel+source_2);

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);
            channel_2_0 = vmlaq_f32(channel_1_0, channel_2_0, kernel_2);

            // store new channel
            const uint32_t target = channel_row*(width-2*padding)+channel_column;

            // sum up whole vector (with bias)
            output_channel[target] = vgetq_lane_f32(channel_2_0, 0) + vgetq_lane_f32(channel_2_0, 1) + vgetq_lane_f32(channel_2_0, 2) + bias;
        }
    }
}

void convolution2d_cpu_3x3_s1_same(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias) {

    fp_t* bigger_input_channel = (fp_t*) malloc((height+2)*(width+2)*sizeof(fp_t));

    // fill first line with 0.0
    int32_t i;
    for(i = 0; i < width+2; i++) {
        bigger_input_channel[i] = 0.0;
    }

    int32_t channel_row;
    for(channel_row = 0; channel_row < height; channel_row++) {
        // set first pixel in row to 0.0
        bigger_input_channel[(channel_row+1)*(width+2)] = 0.0;

        memcpy(bigger_input_channel+(channel_row+1)*(width+2)+1, input_channel+channel_row*width, width*sizeof(fp_t));

        // set last pixel in row to 0.0
        bigger_input_channel[(channel_row+1)*(width+2)+(width+2)-1] = 0.0;
    }

    // fill last line with 0.0
    for(i = 0; i < width+2; i++) {
        bigger_input_channel[(height+2-1)*(width+2)+i] = 0.0;
    }

    convolution2d_cpu_3x3_s1_valid(bigger_input_channel, height+2, width+2, output_channel, kernel, bias);
    free(bigger_input_channel);
}

void convolution2d_cpu_5x5_s1_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias) {

    /*
    3 float_32x4 neon registers for kernel
                                    kernel_5 =
    kernel_0 = [0,0][0,1][0,2][0,3]  [0,4]
    kernel_1 = [1,0][1,1][1,2][1,3]  [1,4]
    kernel_2 = [2,0][2,1][2,2][2,3]  [2,4]
    kernel_3 = [3,0][3,1][3,2][3,3]  [3,4]
    kernel_4 = [4,0][4,1][4,2][4,3]  [4,3] = kernel_6
    */

    uint16_t channel_row, channel_column;
    const uint8_t padding = 2;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;
    float32x4_t kernel_3;
    float32x4_t kernel_4;
    float32x4_t kernel_5;
    fp_t kernel_6;

    fp_t kernel_temp[4];

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+5);
    kernel_2 = vld1q_f32(kernel+10);
    kernel_3 = vld1q_f32(kernel+15);
    kernel_4 = vld1q_f32(kernel+20);

    kernel_temp[0] = kernel[4];
    kernel_temp[1] = kernel[9];
    kernel_temp[2] = kernel[14];
    kernel_temp[3] = kernel[19];
    kernel_5 = vld1q_f32(kernel_temp);

    kernel_6 = kernel[24];

    // vectors for channel
    float32x4_t channel_0_0;
    float32x4_t channel_1_0;
    float32x4_t channel_2_0;
    float32x4_t channel_3_0;
    float32x4_t channel_4_0;
    float32x4_t channel_5_0;
    fp_t channel_6_0;

    float32x4_t channel_0_1;
    float32x4_t channel_1_1;
    float32x4_t channel_2_1;
    float32x4_t channel_3_1;
    float32x4_t channel_4_1;
    float32x4_t channel_5_1;
    fp_t channel_6_1;

    float32x4_t channel_0_2;
    float32x4_t channel_1_2;
    float32x4_t channel_2_2;
    float32x4_t channel_3_2;
    float32x4_t channel_4_2;
    float32x4_t channel_5_2;
    fp_t channel_6_2;

    float32x4_t channel_0_3;
    float32x4_t channel_1_3;
    float32x4_t channel_2_3;
    float32x4_t channel_3_3;
    float32x4_t channel_4_3;
    float32x4_t channel_5_3;
    fp_t channel_6_3;

    fp_t channel_temp[4];

    for(channel_row = 0; channel_row < height-4; channel_row++) {
        for(channel_column = 0; channel_column < width-4-4; channel_column+=4) {

            // load channel into vectors
            const uint32_t source_0 = (channel_row+0)*width+channel_column;
            const uint32_t source_1 = (channel_row+1)*width+channel_column;
            const uint32_t source_2 = (channel_row+2)*width+channel_column;
            const uint32_t source_3 = (channel_row+3)*width+channel_column;
            const uint32_t source_4 = (channel_row+4)*width+channel_column;

            channel_0_0 = vld1q_f32(input_channel+source_0);
            channel_1_0 = vld1q_f32(input_channel+source_1);
            channel_2_0 = vld1q_f32(input_channel+source_2);
            channel_3_0 = vld1q_f32(input_channel+source_3);
            channel_4_0 = vld1q_f32(input_channel+source_4);

            channel_temp[0] = input_channel[source_0+4];
            channel_temp[1] = input_channel[source_1+4];
            channel_temp[2] = input_channel[source_2+4];
            channel_temp[3] = input_channel[source_3+4];
            channel_5_0 = vld1q_f32(channel_temp);

            channel_6_0 = input_channel[source_4+4];


            channel_0_1 = vld1q_f32(input_channel+source_0+1);
            channel_1_1 = vld1q_f32(input_channel+source_1+1);
            channel_2_1 = vld1q_f32(input_channel+source_2+1);
            channel_3_1 = vld1q_f32(input_channel+source_3+1);
            channel_4_1 = vld1q_f32(input_channel+source_4+1);

            channel_temp[0] = input_channel[source_0+4+1];
            channel_temp[1] = input_channel[source_1+4+1];
            channel_temp[2] = input_channel[source_2+4+1];
            channel_temp[3] = input_channel[source_3+4+1];
            channel_5_1 = vld1q_f32(channel_temp);

            channel_6_1 = input_channel[source_4+4+1];


            channel_0_2 = vld1q_f32(input_channel+source_0+2);
            channel_1_2 = vld1q_f32(input_channel+source_1+2);
            channel_2_2 = vld1q_f32(input_channel+source_2+2);
            channel_3_2 = vld1q_f32(input_channel+source_3+2);
            channel_4_2 = vld1q_f32(input_channel+source_4+2);

            channel_temp[0] = input_channel[source_0+4+2];
            channel_temp[1] = input_channel[source_1+4+2];
            channel_temp[2] = input_channel[source_2+4+2];
            channel_temp[3] = input_channel[source_3+4+2];
            channel_5_2 = vld1q_f32(channel_temp);

            channel_6_2 = input_channel[source_4+4+2];


            channel_0_3 = vld1q_f32(input_channel+source_0+3);
            channel_1_3 = vld1q_f32(input_channel+source_1+3);
            channel_2_3 = vld1q_f32(input_channel+source_2+3);
            channel_3_3 = vld1q_f32(input_channel+source_3+3);
            channel_4_3 = vld1q_f32(input_channel+source_4+3);

            channel_temp[0] = input_channel[source_0+4+3];
            channel_temp[1] = input_channel[source_1+4+3];
            channel_temp[2] = input_channel[source_2+4+3];
            channel_temp[3] = input_channel[source_3+4+3];
            channel_5_3 = vld1q_f32(channel_temp);

            channel_6_3 = input_channel[source_4+4+3];


            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);
            channel_2_0 = vmlaq_f32(channel_1_0, channel_2_0, kernel_2);
            channel_3_0 = vmlaq_f32(channel_2_0, channel_3_0, kernel_3);
            channel_4_0 = vmlaq_f32(channel_3_0, channel_4_0, kernel_4);
            channel_5_0 = vmlaq_f32(channel_4_0, channel_5_0, kernel_5);

            channel_0_1 = vmulq_f32(channel_0_1, kernel_0);
            channel_1_1 = vmlaq_f32(channel_0_1, channel_1_1, kernel_1);
            channel_2_1 = vmlaq_f32(channel_1_1, channel_2_1, kernel_2);
            channel_3_1 = vmlaq_f32(channel_2_1, channel_3_1, kernel_3);
            channel_4_1 = vmlaq_f32(channel_3_1, channel_4_1, kernel_4);
            channel_5_1 = vmlaq_f32(channel_4_1, channel_5_1, kernel_5);

            channel_0_2 = vmulq_f32(channel_0_2, kernel_0);
            channel_1_2 = vmlaq_f32(channel_0_2, channel_1_2, kernel_1);
            channel_2_2 = vmlaq_f32(channel_1_2, channel_2_2, kernel_2);
            channel_3_2 = vmlaq_f32(channel_2_2, channel_3_2, kernel_3);
            channel_4_2 = vmlaq_f32(channel_3_2, channel_4_2, kernel_4);
            channel_5_2 = vmlaq_f32(channel_4_2, channel_5_2, kernel_5);

            channel_0_3 = vmulq_f32(channel_0_3, kernel_0);
            channel_1_3 = vmlaq_f32(channel_0_3, channel_1_3, kernel_1);
            channel_2_3 = vmlaq_f32(channel_1_3, channel_2_3, kernel_2);
            channel_3_3 = vmlaq_f32(channel_2_3, channel_3_3, kernel_3);
            channel_4_3 = vmlaq_f32(channel_3_3, channel_4_3, kernel_4);
            channel_5_3 = vmlaq_f32(channel_4_3, channel_5_3, kernel_5);

            // store new channel
            const uint32_t target = channel_row*(width-2*padding)+channel_column;

            output_channel[target] =   vgetq_lane_f32(channel_5_0, 0) + vgetq_lane_f32(channel_5_0, 1) + vgetq_lane_f32(channel_5_0, 2) + vgetq_lane_f32(channel_5_0, 3) + (channel_6_0*kernel_6) + bias;
            output_channel[target+1] = vgetq_lane_f32(channel_5_1, 0) + vgetq_lane_f32(channel_5_1, 1) + vgetq_lane_f32(channel_5_1, 2) + vgetq_lane_f32(channel_5_1, 3) + (channel_6_1*kernel_6) + bias;
            output_channel[target+2] = vgetq_lane_f32(channel_5_2, 0) + vgetq_lane_f32(channel_5_2, 1) + vgetq_lane_f32(channel_5_2, 2) + vgetq_lane_f32(channel_5_2, 3) + (channel_6_2*kernel_6) + bias;
            output_channel[target+3] = vgetq_lane_f32(channel_5_3, 0) + vgetq_lane_f32(channel_5_3, 1) + vgetq_lane_f32(channel_5_3, 2) + vgetq_lane_f32(channel_5_3, 3) + (channel_6_3*kernel_6) + bias;

        }

        // residual columns
        for(channel_column = channel_column; channel_column < width-4; channel_column++) {

            // load channel into vectors
            const uint32_t source_0 = (channel_row+0)*width+channel_column;
            const uint32_t source_1 = (channel_row+1)*width+channel_column;
            const uint32_t source_2 = (channel_row+2)*width+channel_column;
            const uint32_t source_3 = (channel_row+3)*width+channel_column;
            const uint32_t source_4 = (channel_row+4)*width+channel_column;

            channel_0_0 = vld1q_f32(input_channel+source_0);
            channel_1_0 = vld1q_f32(input_channel+source_1);
            channel_2_0 = vld1q_f32(input_channel+source_2);
            channel_3_0 = vld1q_f32(input_channel+source_3);
            channel_4_0 = vld1q_f32(input_channel+source_4);

            channel_temp[0] = input_channel[source_0+4];
            channel_temp[1] = input_channel[source_1+4];
            channel_temp[2] = input_channel[source_2+4];
            channel_temp[3] = input_channel[source_3+4];
            channel_5_0 = vld1q_f32(channel_temp);

            channel_6_0 = input_channel[source_4+4];

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);
            channel_2_0 = vmlaq_f32(channel_1_0, channel_2_0, kernel_2);
            channel_3_0 = vmlaq_f32(channel_2_0, channel_3_0, kernel_3);
            channel_4_0 = vmlaq_f32(channel_3_0, channel_4_0, kernel_4);
            channel_5_0 = vmlaq_f32(channel_4_0, channel_5_0, kernel_5);

            // store new channel
            const uint32_t target = channel_row*(width-2*padding)+channel_column;

            output_channel[target] =   vgetq_lane_f32(channel_5_0, 0) + vgetq_lane_f32(channel_5_0, 1) + vgetq_lane_f32(channel_5_0, 2) + vgetq_lane_f32(channel_5_0, 3) + (channel_6_0*kernel_6) + bias;
        }
    }
}

void convolution2d_cpu_5x5_s1_same(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias) {

    fp_t* bigger_input_channel = (fp_t*) malloc((height+4)*(width+4)*sizeof(fp_t));

    // fill first two lines with 0.0
    int32_t i;
    for(i = 0; i < (width+4)*2; i++) {
        bigger_input_channel[i] = 0.0;
    }

    int32_t channel_row;
    for(channel_row = 0; channel_row < height; channel_row++) {
        // set first two pixels in row to 0.0
        bigger_input_channel[(channel_row+2)*(width+4)]   = 0.0;
        bigger_input_channel[(channel_row+2)*(width+4)+1] = 0.0;

        memcpy(bigger_input_channel+(channel_row+2)*(width+4)+2, input_channel+channel_row*width, width*sizeof(fp_t));

        // set last two pixels in row to 0.0
        bigger_input_channel[(channel_row+2)*(width+4)+(width+4)-2] = 0.0;
        bigger_input_channel[(channel_row+2)*(width+4)+(width+4)-1] = 0.0;
    }

    // fill last two lines with 0.0
    for(i = 0; i < width+4; i++) {
        bigger_input_channel[(height+4-2)*(width+4)+i] = 0.0;
    }

    convolution2d_cpu_5x5_s1_valid(bigger_input_channel, height+4, width+4, output_channel, kernel, bias);
    free(bigger_input_channel);
}

void convolution2d_cpu_11x11_s4_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias) {

    /*
    3 float_32x4 neon registers for kernel (last value will be ignored)
    kernel_0 =               kernel_1 =               kernel_2 =
    [0,0][0,1][0,2][0,3]/    [0,4][0,5][0,6][0,7]/    [0,8][0,9][0,10][x]/
    [5,0][5,1][5,2][5,3]/    [5,4][5,5][5,6][5,7]/    [5,8][5,9][5,10][x]/
    [10,0][10,1][10,2][10,3] [10,4][10,5][10,6][10,7] [10,8][10,9][10,10][x]

    kernel_3 =            kernel_4 =            kernel_5 =
    [1,0][1,1][1,2][1,3]/ [1,4][1,5][1,6][1,7]/ [1,8][1,9][1,10][x]/
    [6,0][6,1][6,2][6,3]  [6,4][6,5][6,6][6,7]  [6,8][6,9][6,10][x]
    kernel_6 =            kernel_7 =            kernel_8 =
    [2,0][2,1][2,2][2,3]/ [2,4][2,5][2,6][2,7]/ [2,8][2,9][2,10][x]/
    [7,0][7,1][7,2][7,3]  [7,4][7,5][7,6][7,7]  [7,8][7,9][7,10][x]
    kernel_9 =            kernel_10 =           kernel_11 =
    [3,0][3,1][3,2][3,3]  [3,4][3,5][3,6][3,7]  [3,8][3,9][3,10][x]
    [8,0][8,1][8,2][8,3]  [8,4][8,5][8,6][8,7]  [8,8][8,9][8,10][x]
    kernel_12 =           kernel_13 =           kernel_14 =
    [4,0][4,1][4,2][4,3]  [4,4][4,5][4,6][4,7]  [4,4][4,4][4,10][x]
    [9,0][9,1][9,2][9,3]  [9,4][9,5][9,6][9,7]  [9,8][9,9][9,10][x]
    */

	uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_width = ((width-2*5)/4)+1;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;
    float32x4_t kernel_3;
    float32x4_t kernel_4;
    float32x4_t kernel_5;
    float32x4_t kernel_6;
    float32x4_t kernel_7;
    float32x4_t kernel_8;
    float32x4_t kernel_9;
    float32x4_t kernel_10;
    float32x4_t kernel_11;
    float32x4_t kernel_12;
    float32x4_t kernel_13;
    float32x4_t kernel_14;

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+5);
    kernel_2 = vld1q_f32(kernel+10);
    kernel_3 = vld1q_f32(kernel+15);
    kernel_4 = vld1q_f32(kernel+20);

    // vectors for channel
    float32x4_t channel_0_0;
    float32x4_t channel_1_0;
    float32x4_t channel_2_0;
    float32x4_t channel_3_0;
    float32x4_t channel_4_0;
    float32x4_t channel_5_0;
    float32x4_t channel_6_0;
    float32x4_t channel_7_0;
    float32x4_t channel_8_0;
    float32x4_t channel_9_0;
    float32x4_t channel_10_0;
    float32x4_t channel_11_0;
    float32x4_t channel_12_0;
    float32x4_t channel_13_0;
    float32x4_t channel_14_0;

    fp_t pixel;

    output_channel_row = 0;
    output_channel_column = 0;

    for(channel_row = 0; channel_row < height-10; channel_row+=4) {
        for(channel_column = 0; channel_column < width-10; channel_column+=4) {

            // load first part of kernel
            kernel_0 = vld1q_f32(kernel);
            kernel_1 = vld1q_f32(kernel+4);
            kernel_2 = vld1q_f32(kernel+8);

            kernel_3 = vld1q_f32(kernel+11);
            kernel_4 = vld1q_f32(kernel+15);
            kernel_5 = vld1q_f32(kernel+19);

            kernel_6 = vld1q_f32(kernel+22);
            kernel_7 = vld1q_f32(kernel+26);
            kernel_8 = vld1q_f32(kernel+30);

            kernel_9 = vld1q_f32(kernel+33);
            kernel_10 = vld1q_f32(kernel+37);
            kernel_11 = vld1q_f32(kernel+41);

            kernel_12 = vld1q_f32(kernel+44);
            kernel_13 = vld1q_f32(kernel+48);
            kernel_14 = vld1q_f32(kernel+52);

            // load first part of channel
            channel_0_0 = vld1q_f32(input_channel+(channel_row+0)*width+channel_column);
            channel_1_0 = vld1q_f32(input_channel+(channel_row+0)*width+channel_column+4);
            channel_2_0 = vld1q_f32(input_channel+(channel_row+0)*width+channel_column+8);

            channel_3_0 = vld1q_f32(input_channel+(channel_row+1)*width+channel_column);
            channel_4_0 = vld1q_f32(input_channel+(channel_row+1)*width+channel_column+4);
            channel_5_0 = vld1q_f32(input_channel+(channel_row+1)*width+channel_column+8);

            channel_6_0 = vld1q_f32(input_channel+(channel_row+2)*width+channel_column);
            channel_7_0 = vld1q_f32(input_channel+(channel_row+2)*width+channel_column+4);
            channel_8_0 = vld1q_f32(input_channel+(channel_row+2)*width+channel_column+8);

            channel_9_0  = vld1q_f32(input_channel+(channel_row+3)*width+channel_column);
            channel_10_0 = vld1q_f32(input_channel+(channel_row+3)*width+channel_column+4);
            channel_11_0 = vld1q_f32(input_channel+(channel_row+3)*width+channel_column+8);

            channel_12_0 = vld1q_f32(input_channel+(channel_row+4)*width+channel_column);
            channel_13_0 = vld1q_f32(input_channel+(channel_row+4)*width+channel_column+4);
            channel_14_0 = vld1q_f32(input_channel+(channel_row+4)*width+channel_column+8);

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);

            channel_3_0 = vmlaq_f32(channel_1_0, channel_3_0, kernel_3);
            channel_4_0 = vmlaq_f32(channel_3_0, channel_4_0, kernel_4);

            channel_6_0 = vmlaq_f32(channel_4_0, channel_6_0, kernel_6);
            channel_7_0 = vmlaq_f32(channel_6_0, channel_7_0, kernel_7);

            channel_9_0 = vmlaq_f32(channel_7_0, channel_9_0, kernel_9);
            channel_10_0 = vmlaq_f32(channel_9_0, channel_10_0, kernel_10);

            channel_12_0 = vmlaq_f32(channel_10_0, channel_12_0, kernel_12);
            channel_13_0 = vmlaq_f32(channel_12_0, channel_13_0, kernel_13);

            // sum up whole vector
            pixel = vgetq_lane_f32(channel_13_0, 0) + vgetq_lane_f32(channel_13_0, 1) + vgetq_lane_f32(channel_13_0, 2) + vgetq_lane_f32(channel_13_0, 3);


            channel_2_0 = vmulq_f32(channel_2_0, kernel_2);
            channel_5_0 = vmlaq_f32(channel_2_0, channel_5_0, kernel_5);
            channel_8_0 = vmlaq_f32(channel_5_0, channel_8_0, kernel_8);
            channel_11_0 = vmlaq_f32(channel_8_0, channel_11_0, kernel_11);
            channel_14_0 = vmlaq_f32(channel_11_0, channel_14_0, kernel_14);

            // sum up whole vector
            pixel += vgetq_lane_f32(channel_14_0, 0) + vgetq_lane_f32(channel_14_0, 1) + vgetq_lane_f32(channel_14_0, 2);


            // load second part of kernel
            kernel_0 = vld1q_f32(kernel+55);
            kernel_1 = vld1q_f32(kernel+59);
            kernel_2 = vld1q_f32(kernel+63);

            kernel_3 = vld1q_f32(kernel+66);
            kernel_4 = vld1q_f32(kernel+70);
            kernel_5 = vld1q_f32(kernel+74);

            kernel_6 = vld1q_f32(kernel+77);
            kernel_7 = vld1q_f32(kernel+81);
            kernel_8 = vld1q_f32(kernel+85);

            kernel_9 = vld1q_f32(kernel+88);
            kernel_10 = vld1q_f32(kernel+92);
            kernel_11 = vld1q_f32(kernel+96);

            kernel_12 = vld1q_f32(kernel+99);
            kernel_13 = vld1q_f32(kernel+103);
            kernel_14 = vld1q_f32(kernel+107);

            // load second part of channel
            channel_0_0 = vld1q_f32(input_channel+(channel_row+5)*width+channel_column);
            channel_1_0 = vld1q_f32(input_channel+(channel_row+5)*width+channel_column+4);
            channel_2_0 = vld1q_f32(input_channel+(channel_row+5)*width+channel_column+8);

            channel_3_0 = vld1q_f32(input_channel+(channel_row+6)*width+channel_column);
            channel_4_0 = vld1q_f32(input_channel+(channel_row+6)*width+channel_column+4);
            channel_5_0 = vld1q_f32(input_channel+(channel_row+6)*width+channel_column+8);

            channel_6_0 = vld1q_f32(input_channel+(channel_row+7)*width+channel_column);
            channel_7_0 = vld1q_f32(input_channel+(channel_row+7)*width+channel_column+4);
            channel_8_0 = vld1q_f32(input_channel+(channel_row+7)*width+channel_column+8);

            channel_9_0  = vld1q_f32(input_channel+(channel_row+8)*width+channel_column);
            channel_10_0 = vld1q_f32(input_channel+(channel_row+8)*width+channel_column+4);
            channel_11_0 = vld1q_f32(input_channel+(channel_row+8)*width+channel_column+8);

            channel_12_0 = vld1q_f32(input_channel+(channel_row+9)*width+channel_column);
            channel_13_0 = vld1q_f32(input_channel+(channel_row+9)*width+channel_column+4);
            channel_14_0 = vld1q_f32(input_channel+(channel_row+9)*width+channel_column+8);

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);

            channel_3_0 = vmlaq_f32(channel_1_0, channel_3_0, kernel_3);
            channel_4_0 = vmlaq_f32(channel_3_0, channel_4_0, kernel_4);

            channel_6_0 = vmlaq_f32(channel_4_0, channel_6_0, kernel_6);
            channel_7_0 = vmlaq_f32(channel_6_0, channel_7_0, kernel_7);

            channel_9_0 = vmlaq_f32(channel_7_0, channel_9_0, kernel_9);
            channel_10_0 = vmlaq_f32(channel_9_0, channel_10_0, kernel_10);

            channel_12_0 = vmlaq_f32(channel_10_0, channel_12_0, kernel_12);
            channel_13_0 = vmlaq_f32(channel_12_0, channel_13_0, kernel_13);

            // sum up whole vector
            pixel += vgetq_lane_f32(channel_13_0, 0) + vgetq_lane_f32(channel_13_0, 1) + vgetq_lane_f32(channel_13_0, 2) + vgetq_lane_f32(channel_13_0, 3);


            channel_2_0 = vmulq_f32(channel_2_0, kernel_2);
            channel_5_0 = vmlaq_f32(channel_2_0, channel_5_0, kernel_5);
            channel_8_0 = vmlaq_f32(channel_5_0, channel_8_0, kernel_8);
            channel_11_0 = vmlaq_f32(channel_8_0, channel_11_0, kernel_11);
            channel_14_0 = vmlaq_f32(channel_11_0, channel_14_0, kernel_14);

            // sum up whole vector
            pixel += vgetq_lane_f32(channel_14_0, 0) + vgetq_lane_f32(channel_14_0, 1) + vgetq_lane_f32(channel_14_0, 2);


            // load third part of kernel
            kernel_0 = vld1q_f32(kernel+110);
            kernel_1 = vld1q_f32(kernel+114);
            kernel_2 = vld1q_f32(kernel+118);

            // load third part of channel
            channel_0_0 = vld1q_f32(input_channel+(channel_row+10)*width+channel_column);
            channel_1_0 = vld1q_f32(input_channel+(channel_row+10)*width+channel_column+4);
            channel_2_0 = vld1q_f32(input_channel+(channel_row+10)*width+channel_column+8);

            // apply kernel
            channel_0_0 = vmulq_f32(channel_0_0, kernel_0);
            channel_1_0 = vmlaq_f32(channel_0_0, channel_1_0, kernel_1);

            // sum up whole vector
            pixel += vgetq_lane_f32(channel_1_0, 0) + vgetq_lane_f32(channel_1_0, 1) + vgetq_lane_f32(channel_1_0, 2) + vgetq_lane_f32(channel_1_0, 3);

            channel_2_0 = vmulq_f32(channel_2_0, kernel_2);

            // sum up whole vector
            pixel += vgetq_lane_f32(channel_2_0, 0) + vgetq_lane_f32(channel_2_0, 1) + vgetq_lane_f32(channel_2_0, 2) + bias;


            // store new channel
            output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
            output_channel_column++;
        }
        output_channel_row++;
        output_channel_column = 0;
    }
}

void add_channel2d_cpu(fp_t* channel_a, const fp_t* channel_b, const uint16_t height, const uint16_t width) {
    uint32_t i;

    float32x4_t channel_a_0;
    float32x4_t channel_a_1;
    float32x4_t channel_a_2;
    float32x4_t channel_a_3;

    float32x4_t channel_b_0;
    float32x4_t channel_b_1;
    float32x4_t channel_b_2;
    float32x4_t channel_b_3;

    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {
        // load channels into vector
        channel_a_0 = vld1q_f32(channel_a+i);
        channel_a_1 = vld1q_f32(channel_a+i+4);
        channel_a_2 = vld1q_f32(channel_a+i+8);
        channel_a_3 = vld1q_f32(channel_a+i+12);

        channel_b_0 = vld1q_f32(channel_b+i);
        channel_b_1 = vld1q_f32(channel_b+i+4);
        channel_b_2 = vld1q_f32(channel_b+i+8);
        channel_b_3 = vld1q_f32(channel_b+i+12);

        // add vectors
        channel_a_0 = vaddq_f32(channel_a_0, channel_b_0);
        channel_a_1 = vaddq_f32(channel_a_1, channel_b_1);
        channel_a_2 = vaddq_f32(channel_a_2, channel_b_2);
        channel_a_3 = vaddq_f32(channel_a_3, channel_b_3);

        // store vectors in channel a
        vst1q_f32(channel_a+i, channel_a_0);
        vst1q_f32(channel_a+i+4, channel_a_1);
        vst1q_f32(channel_a+i+8, channel_a_2);
        vst1q_f32(channel_a+i+12, channel_a_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        channel_a[i] = channel_a[i] + channel_b[i];
    }
}
#endif //ARM_NEON

#ifdef FIXED16
void convolution2d_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel, const fixed16_t* kernel, const uint16_t kernel_size, const uint16_t stride, const uint16_t padding, const fixed16_t bias) {
    int32_t channel_row, channel_column;
    int32_t kernel_row, kernel_column;
    int32_t crop = kernel_size/2;

    int32_t output_channel_row, output_channel_column, output_channel_width;

    fixed16_t pixel;

    output_channel_row = 0;
    output_channel_column = 0;

    // padding valid
    if(padding == 0) {
        if(stride == 1) {
            output_channel_width = ((width-2*crop)/stride);
        } else {
            output_channel_width = ((width-2*crop)/stride)+1;
        }

        for(channel_row = crop; channel_row < height-crop; channel_row+=stride) {
            for(channel_column = crop; channel_column < width-crop; channel_column+=stride) {
                pixel = 0;

                for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                        pixel = add_fixed16(pixel, mul_fixed16(kernel[kernel_row*kernel_size+kernel_column], input_channel[(channel_row-crop+kernel_row)*width+(channel_column-crop+kernel_column)]));
                    }
                }

                pixel = add_fixed16(pixel, bias);

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {
        output_channel_width = width;

        for(channel_row = 0; channel_row < height; channel_row+=stride) {
            for(channel_column = 0; channel_column < width; channel_column+=stride) {
                pixel = 0;

                for(kernel_row = -padding; kernel_row <= padding; kernel_row++) {
                    for(kernel_column = -padding; kernel_column <= padding; kernel_column++) {
                        if((channel_row+kernel_row) < 0 || (channel_row+kernel_row) > height-1 || (channel_column+kernel_column) < 0 || (channel_column+kernel_column) > width-1) {
                            pixel = add_fixed16(0, pixel);
                        } else {
                            pixel = add_fixed16(pixel, mul_fixed16(kernel[(kernel_row+padding)*kernel_size+(kernel_column+padding)], input_channel[(channel_row+kernel_row)*width+(channel_column+kernel_column)]));
                        }
                    }
                }

                pixel = add_fixed16(pixel, bias);

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }
}

void convolution2d_cpu_5x5_s1_valid_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel, const fixed16_t* kernel, const fixed16_t bias) {
	convolution2d_naive_fixed16(input_channel, height, width, output_channel, kernel, 5, 1, 0, bias);
}

void add_channel2d_naive_fixed16(fixed16_t* channel_a, const fixed16_t* channel_b, const uint16_t height, const uint16_t width) {
    uint32_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            channel_a[row*width+column] = add_fixed16(channel_a[row*width+column], channel_b[row*width+column]);
        }
    }
}

void add_channel2d_cpu_fixed16(fixed16_t* channel_a, const fixed16_t* channel_b, const uint16_t height, const uint16_t width) {
	add_channel2d_naive_fixed16(channel_a, channel_b, height, width);
}
#endif // FIXED16
