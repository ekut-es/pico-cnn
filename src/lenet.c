#include "parameters.h"
#include <stdio.h>
#include "read_mnist.h"
#include "write_pgm.h"
#include "lenet_kernels.h"
#include "convolution.h"
#include "activation_function.h"
#include "pooling.h"

#define NUM 1
#define DEBUG
#define INDEX 0

int main(int argc, char** argv) {

    if(argc != 2) {
        fprintf(stderr, "no path to mnist dataset provided!\n");
        return 1;
    }

    int i, j;

    // read mnist t10k images
    char t10k_images_path[strlen(argv[1]) + 20];
    t10k_images_path[0] = '\0';
    strcat(t10k_images_path, argv[1]);
    strcat(t10k_images_path, "/t10k-images.idx3-ubyte");

    float_t** t10k_images;
    int num_images;
    int padding = 2;
    
    printf("reading images from '%s'\n", t10k_images_path);

    num_images = read_mnist_images(t10k_images_path, &t10k_images, NUM, padding);

    if(num_images < 1) {
        fprintf(stderr, "could not read mnist images from '%s'\n", t10k_images_path);
        return 1;
    }

    // read t10k labels
    char t10k_labels_path[strlen(argv[1]) + 20];
    t10k_labels_path[0] = '\0';
    strcat(t10k_labels_path, argv[1]);
    strcat(t10k_labels_path, "/t10k-labels.idx1-ubyte");

    uint8_t* t10k_labels;
    int num_labels;

    printf("reading labels from '%s'\n", t10k_labels_path);

    num_labels = read_mnist_labels(t10k_labels_path, &t10k_labels, NUM);

    // make pgm of original image
    #ifdef DEBUG
    write_pgm(32, 32, t10k_images[INDEX], "input.pgm");
    #endif

    // TODO read kernels and bias from file

    for(i = 0; i < NUM; i++) {

        // C1
        float_t** c1_output;
        c1_output = (float_t**) malloc(6*sizeof(float_t*));

        for(j = 0; j < 6; j++) {
            c1_output[j] = (float_t*) malloc(28*28*sizeof(float_t));
            convolution2d_naive(t10k_images[i], 32, 32, c1_output[j], c1_kernels[j], 5, c1_bias[j]);
            tanh_naive(c1_output[j], 28, 28, c1_output[j]);
        }

        // make pgm C1
        #ifdef DEBUG
        if(i == INDEX) {
            float* c1_pgm = (float_t*) malloc(28*28*6*sizeof(float_t));
            memcpy(&c1_pgm[0*28*28], c1_output[0], 28*28*sizeof(float_t));
            memcpy(&c1_pgm[1*28*28], c1_output[1], 28*28*sizeof(float_t));
            memcpy(&c1_pgm[2*28*28], c1_output[2], 28*28*sizeof(float_t));
            memcpy(&c1_pgm[3*28*28], c1_output[3], 28*28*sizeof(float_t));
            memcpy(&c1_pgm[4*28*28], c1_output[4], 28*28*sizeof(float_t));
            memcpy(&c1_pgm[5*28*28], c1_output[5], 28*28*sizeof(float_t));
            write_pgm(6*28, 28, c1_pgm, "c1_output.pgm");
            free(c1_pgm);
        }
        #endif

        // S1
        float_t** s1_output;
        s1_output = (float_t**) malloc(6*sizeof(float_t*));

        for(j = 0; j < 6; j++) {
            s1_output[j] = (float_t*) malloc(14*14*sizeof(float_t));
            max_pooling2d_naive(c1_output[j], 28, 28, s1_output[j], 2);
            tanh_naive(s1_output[j], 14, 14, s1_output[j]);
        }

        // make pgm S1
        #ifdef DEBUG
        if(i == INDEX) {
            float* s1_pgm = (float_t*) malloc(14*14*6*sizeof(float_t));
            memcpy(&s1_pgm[0*14*14], s1_output[0], 14*14*sizeof(float_t));
            memcpy(&s1_pgm[1*14*14], s1_output[1], 14*14*sizeof(float_t));
            memcpy(&s1_pgm[2*14*14], s1_output[2], 14*14*sizeof(float_t));
            memcpy(&s1_pgm[3*14*14], s1_output[3], 14*14*sizeof(float_t));
            memcpy(&s1_pgm[4*14*14], s1_output[4], 14*14*sizeof(float_t));
            memcpy(&s1_pgm[5*14*14], s1_output[5], 14*14*sizeof(float_t));
            write_pgm(6*14, 14, s1_pgm, "s1_output.pgm");
            free(s1_pgm);
        }
        #endif


        // make pgm of S1

        // C1 free memory
        for(j = 0; j < 6; j++) {
            free(c1_output[j]);
        }

        free(c1_output);

        // C2

        // S1 free memory
        for(j = 0; j < 6; j++) {
            free(s1_output[j]);
        }

        free(s1_output);

    }

        
    

    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    return 0;
}
