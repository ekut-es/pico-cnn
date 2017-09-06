/** 
 * @brief AlexNet implementation as provided in the BVLC model zoo:
 * https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define DEBUG

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>

void usage() {
    printf("./bvlc_alexnet \\\n"); 
    printf("PATH_TO_ALEXNET_WEIGHTS.weights \\\n");
    printf("PATH_TO_IMAGE_NET_MEAN.binaryproto \\\n");
    printf("PATH_TO_IMAGE_NET_LABELS.txt \\\n");
    printf("PATH_TO_IMAGE.jpg\n");
}

int main(int argc, char** argv) {

    if(argc != 5) {
        fprintf(stderr, "too few or to many arguments!\n");
        usage();
        return 1;
    }

    unsigned int i, j, k;

    float_t** input;

    uint16_t height;
    uint16_t width;

    read_jpeg(&input, argv[4], 0.0, 1.0, &height, &width);

    // make pgm of original image
    #ifdef DEBUG
    float* input_file_content = (float_t*) malloc(227*227*3*sizeof(float_t));
    for(j = 0; j < 3; j++) {
        memcpy(&input_file_content[j*227*227], input[j], 227*227*sizeof(float_t));
    }
    
    write_pgm(input_file_content, 3*227, 227, "input.pgm");
    write_float(input_file_content, 3*227, 227, "input.float");
    free(input_file_content);
    #endif

    return 0;
}



