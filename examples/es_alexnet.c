/** 
 * @brief AlexNet implementation as provided in the BVLC model zoo:
 * https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define IMAGENET
#define DEBUG

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>

void usage() {
    printf("./es_alexnet \\\n"); 
    printf("PATH_TO_ES_WEIGHTS.weights \\\n");
    printf("PATH_TO_MEANS_FILE.means \\\n");
    printf("PATH_TO_IMAGE_NET_LABELS.txt \\\n");
    printf("PATH_TO_IMAGE.jpg\n");
}

int main(int argc, char** argv) {

    if(argc != 5) {
        fprintf(stderr, "too few or to many arguments!\n");
        usage();
        return 1;
    }

    char weights_path[1024];
    char means_path[1024];
    char labels_path[1024];
    char jpeg_path[1024];

    strcpy(weights_path, argv[1]);
    strcpy(means_path, argv[2]);
    strcpy(labels_path, argv[3]);
    strcpy(jpeg_path, argv[4]);

    unsigned int i, j, k;


    // read kernels and biasses
    fp_t*** kernels;
    fp_t** biasses;

    printf("reading weights from '%s'\n", weights_path);

    if(read_weights(weights_path, &kernels, &biasses) != 0) {
        fprintf(stderr, "could not read weights from '%s'\n", weights_path);
        return 1;
    }

    // read means
    printf("reading means from '%s'\n", means_path);

    fp_t* means = (fp_t*) malloc(3*sizeof(fp_t));

    if(read_means(means_path, means) != 0) {
        fprintf(stderr, "could not read means file '%s'!\n", means_path);
        return 1;
    }

    // TODO read labels


    // read input image
    printf("reading input image '%s'\n", jpeg_path);

    fp_t** pre_mean_input;

    uint16_t height;
    uint16_t width;

    read_jpeg(&pre_mean_input, jpeg_path, 0.0, 255.0, &height, &width);

    // make pgm of pre mean input image
    #ifdef DEBUG
    float* pre_mean_input_file_content = (fp_t*) malloc(227*227*3*sizeof(fp_t));
    for(j = 0; j < 3; j++) {
        memcpy(&pre_mean_input_file_content[j*227*227], pre_mean_input[j], 227*227*sizeof(fp_t));
    }
    
    write_pgm(pre_mean_input_file_content, 3*227, 227, "pre_mean_input.pgm");
    write_float(pre_mean_input_file_content, 3*227, 227, "pre_mean_input.float");
    free(pre_mean_input_file_content);
    #endif

    // substract mean from each channel
    fp_t** input = (fp_t**) malloc(3*sizeof(fp_t*));
    input[0] = (fp_t*) malloc(227*227*sizeof(fp_t));
    input[1] = (fp_t*) malloc(227*227*sizeof(fp_t));
    input[2] = (fp_t*) malloc(227*227*sizeof(fp_t));
        
    uint16_t row;
    uint16_t column;

    for(i = 0; i < 3; i++) {
        for(row = 0; row < height; row++) {
            for(column = 0; column < height; column++) {
                input[i][row*width+column] = pre_mean_input[i][row*width+column] - means[i];
            }
        }
    }

    // free pre mean input image
    /*
    for(i = 0; i < 3; i++) {
        free(pre_mean_input[i]);
    }
    free(pre_mean_input);
    */

    // make pgm of input image
    #ifdef DEBUG
    float* input_file_content = (fp_t*) malloc(227*227*3*sizeof(fp_t));
    for(j = 0; j < 3; j++) {
        memcpy(&input_file_content[j*227*227], input[j], 227*227*sizeof(fp_t));
    }
    
    write_pgm(input_file_content, 3*227, 227, "input.pgm");
    write_float(input_file_content, 3*227, 227, "input.float");
    free(input_file_content);
    #endif


    printf("starting CNN\n");

    // conv1 input 227x227x3 -> output 55x55x96
    fp_t** conv1_output;
    conv1_output = (fp_t**) malloc(96*sizeof(fp_t*));

    for(i = 0; i < 96; i++) {
        conv1_output[i] = (fp_t*) malloc(55*55*sizeof(fp_t));
    }

    fp_t* conv1_intermediate = (fp_t*) malloc(55*55*sizeof(fp_t));

    fp_t** conv1_kernels = kernels[0];
    fp_t* conv1_bias = biasses[0];

 
    uint16_t kernel_number = 0;

    for(i = 0; i < 96; i++) {
        convolution2d_naive(input[0], 227, 227, conv1_output[i], conv1_kernels[kernel_number], 11, 4, 0.0);
        kernel_number++;

        convolution2d_naive(input[1], 227, 227, conv1_intermediate, conv1_kernels[kernel_number], 11, 4, 0.0);
        add_image2d_naive(conv1_output[i], conv1_intermediate, 55, 55);
        kernel_number++;

        convolution2d_naive(input[2], 227, 227, conv1_intermediate, conv1_kernels[kernel_number], 11, 4, conv1_bias[i]);
        add_image2d_naive(conv1_output[i], conv1_intermediate, 55, 55);
        kernel_number++;
    }

    // make pgm of input image
    #ifdef DEBUG
    fp_t* conv1_file_content = (fp_t*) malloc(55*55*96*sizeof(fp_t));
    for(i = 0; i < 96; i++) {
        memcpy(&conv1_file_content[i*55*55], conv1_output[i], 55*55*sizeof(fp_t));
    }
    
    write_pgm(conv1_file_content, 96*55, 55, "conv1_output.pgm");
    write_float(conv1_file_content, 96*55, 55, "conv1_output.float");
    free(conv1_file_content);
    #endif

    // free input
    free(input[0]);
    free(input[1]);
    free(input[2]);
    free(input);

    // relu1
    for(i = 0; i < 96; i++) {
        relu_naive(conv1_output[i], 55, 55, conv1_output[i]);
    }

    // make pgm of input image
    #ifdef DEBUG
    fp_t* relu1_file_content = (fp_t*) malloc(55*55*96*sizeof(fp_t));
    for(i = 0; i < 96; i++) {
        memcpy(&relu1_file_content[i*55*55], conv1_output[i], 55*55*sizeof(fp_t));
    }
    write_pgm(relu1_file_content, 96*55, 55, "relu1_output.pgm");
    write_float(relu1_file_content, 96*55, 55, "relu1_output.float");
    free(relu1_file_content);
    #endif

    // TODO lrn 

    // free conv1 output
    for(i = 0; i < 96; i++) {
        free(conv1_output[i]);
    }
    free(conv1_output);


    for(i = 0; i < 288; i++) {
        free(kernels[0][i]);
    }

    /*
    for(i = 0; i < 24576; i++) {
        free(kernels[1][i]);
    }
    for(i = 0; i < 98304; i++) {
        free(kernels[2][i]);
    }
    for(i = 0; i < 147456; i++) {
        free(kernels[3][i]);
    }
    for(i = 0; i < 98304; i++) {
        free(kernels[4][i]);
    }
    free(kernels[5][0]);
    free(kernels[6][0]);
    free(kernels[7][0]);
    */
    free(kernels);

    for(i = 0; i < 1; i++) {
        free(biasses[i]);
    }
    free(biasses);

    return 0;
}



