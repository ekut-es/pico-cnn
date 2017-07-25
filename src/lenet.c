#include "parameters.h"
#include <stdio.h>
#include "read_mnist.h"
#include "write_pgm.h"
#include "convolution.h"

#define NUM 1

int main(int argc, char** argv) {

    if(argc != 2) {
        fprintf(stderr, "no path to mnist dataset provided!\n");
        return 1;
    }

    int i;

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
    for(i = 0; i < NUM; i++) {
        char pgm_path[20];
        sprintf(pgm_path, "%u_%u.pgm", i, t10k_labels[i]);
        write_pgm(28+2*padding, 28+2*padding, t10k_images[i], pgm_path);
    }

    float_t* conv_image;
    conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

    float_t kernel[25] = {1/255.0f, 4 /255.0f, 6 /255.0f, 4 /255.0f, 1/255.0f,
                          4/255.0f, 16/255.0f, 25/255.0f, 16/255.0f, 4/255.0f,
                          6/255.0f, 24/255.0f, 36/255.0f, 24/255.0f, 6/255.0f,
                          4/255.0f, 16/255.0f, 25/255.0f, 16/255.0f, 4/255.0f,
                          1/255.0f, 4 /255.0f, 6 /255.0f, 4 /255.0f, 1/255.0f};

    convolution2d(t10k_images[0], 32, 32, conv_image, kernel, 5, 1);

    write_pgm(28, 28, conv_image, "conv_image.pgm");

    // free memory
    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    return 0;
}
