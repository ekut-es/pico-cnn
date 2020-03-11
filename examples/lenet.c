#define MNIST
#define NUM 1000
#define INDEX 0

#include "network.h"
#include "network_initialization.h"
#include "network_cleanup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "pico-cnn/pico-cnn.h"

void usage() {
    printf("./lenet PATH_TO_DATASET PATH_TO_BINARY_WEIGHTS_FILE\n");
}

void sort_prediction(fp_t* prediction, uint8_t* labels, const uint16_t length) {
    // simple bubble sort
    uint16_t i,j;

    fp_t temp_prediction;
    uint8_t temp_label;

    for(i = 0; i < length-1; i++) {
        for(j = 0; j < length-1-i; j++) {
            if(prediction[j] < prediction[j+1]) {
                // swap
                temp_prediction = prediction[j];
                prediction[j] = prediction[j+1];
                prediction[j+1] = temp_prediction;

                temp_label = labels[j];
                labels[j] = labels[j+1];
                labels[j+1] = temp_label;
            }
        }
    }
}

int32_t main(int32_t argc, char** argv) {

    if(argc == 1) {
        ERROR_MSG("No path to dataset and weights provided!\n");
        usage();
        return 1;
    }

    if(argc == 2) {
        ERROR_MSG("No path to weights provided!\n");
        usage();
        return 1;
    }

    char mnist_path[1024];
    char weights_path[1024];

    strcpy(mnist_path, argv[1]);
    strcpy(weights_path, argv[2]);

    uint32_t i, j;

    // read mnist t10k images
    char t10k_images_path[strlen(mnist_path) + 20];
    t10k_images_path[0] = '\0';
    strcat(t10k_images_path, mnist_path);
    strcat(t10k_images_path, "/t10k-images.idx3-ubyte");

    fp_t** t10k_images;
    uint32_t num_t10k_images;

    INFO_MSG("Reading images from '%s'\n", t10k_images_path);

    num_t10k_images = read_mnist_images(t10k_images_path, &t10k_images, NUM, 0, 0.0, 1.0);

    if(num_t10k_images < 1) {
        ERROR_MSG("Could not read mnist images from '%s'\n", t10k_images_path);
        return 1;
    }

    // read t10k labels
    char t10k_labels_path[strlen(mnist_path) + 20];
    t10k_labels_path[0] = '\0';
    strcat(t10k_labels_path, mnist_path);
    strcat(t10k_labels_path, "/t10k-labels.idx1-ubyte");

    uint8_t* t10k_labels;
    uint32_t num_t10k_labels;

    INFO_MSG("Reading labels from '%s'\n", t10k_labels_path);

    num_t10k_labels = read_mnist_labels(t10k_labels_path, &t10k_labels, NUM);

    if(num_t10k_images != num_t10k_labels) {
        ERROR_MSG("%d images != %d labels\n", num_t10k_images, num_t10k_labels);
        return 1;
    }

    // make pgm of original image
    #ifdef DEBUG
    write_pgm(t10k_images[INDEX], 28, 28, "input.pgm");
    write_float(t10k_images[INDEX], 28, 28, "input.float");
    #endif

    initialize_network();

    INFO_MSG("Reading weights from '%s'\n", weights_path);

    if(read_binary_weights(weights_path, &kernels, &biases) != 0){
        ERROR_MSG("Could not read weights from '%s'\n", weights_path);
        return 1;
    }

    uint32_t correct_predictions = 0;

    uint32_t confusion_matrix[10][10] = {
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0}
    };

    INFO_MSG("Starting CNN\n");

    float *output  = (float*) malloc(10*sizeof(float));


    for(i = 0; i < NUM; i++) {

        network(&t10k_images[i], output);

        float max = output[0];
        uint32_t label = 0;
        for(uint32_t idx = 1; idx < 10; idx++) {
            if(output[idx] > max) {
                max = output[idx];
                label = idx;
            }

        }

        if(t10k_labels[i] == label) {
            correct_predictions++;
        }

        #ifdef DEBUG
        if(i == 0) {
            for(uint32_t idx = 0; idx < 10; idx++){
                DEBUG_MSG("%f ", output[idx]);
            }
            DEBUG_MSG("\n");
            for(uint32_t h = 0; h < 28*28; h++){
//                if(t10k_images[0][h] != 0.0){
//                    DEBUG_MSG("1 ");
//                } else {
//                    DEBUG_MSG("0 ");
//                }
                DEBUG_MSG("%.3f ", t10k_images[0][h]);
                if(h % 28 == 27)
                    DEBUG_MSG("\n");
            }
            DEBUG_MSG("\n");

            DEBUG_MSG("Prediction: %d\n", label);
        }
        #endif

        confusion_matrix[label][t10k_labels[i]]++;

    }

    cleanup_network();

    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    free(output);

    // calculate and print results
    fp_t error_rate = 1.0-((fp_t) correct_predictions/((fp_t) NUM));

    INFO_MSG("Error rate: %f (%d/%d)\n", error_rate, correct_predictions, num_t10k_images);

    INFO_MSG("Columns: actual label\n");
    INFO_MSG("Rows: predicted label\n");
    INFO_MSG("*\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n");

    for(i = 0; i < 10; i++) {
        INFO_MSG("%d\t", i);
        for(j = 0; j < 10; j++) {
            INFO_MSG("%d\t", confusion_matrix[i][j]);
        }
        INFO_MSG("\n");
    }

    return 0;
}
