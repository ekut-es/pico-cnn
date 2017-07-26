#include "parameters.h"
#include <stdio.h>
#include "read_mnist.h"
#include "write_pgm.h"
#include "convolution.h"
#include "activation_function.h"

#define NUM 2

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
    for(i = 0; i < NUM; i++) {
        char pgm_path[20];
        sprintf(pgm_path, "%u_%u.pgm", i, t10k_labels[i]);
        write_pgm(28+2*padding, 28+2*padding, t10k_images[i], pgm_path);
    }

    // TODO read kernels and bias from file
    // TODO loop for first convolution layer
    // TODO max and average pooling

    int index = 0;

    // 0
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {0.0503702, 0.321829, 0.178996, 0.291132, -0.102737,
                            -0.043053, 0.0800959, 0.260101, -0.0331972, -0.0382014,
                            -0.115718, -0.0159754, -0.115343, -0.0055044, -0.00718199,
                            -0.00923267, -0.0708757, 0.0664141, 0.016911, 0.157364,
                            -0.0537186, -0.0435987, 0.0133913, -0.0243876, -0.123523};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, -0.00347255);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv0_image.pgm");

        free(conv_image);
    }

    // 1
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {-0.0770365, 0.0345966, -0.160769, -0.225885, -0.119277,
                            -0.0264168, 0.0616594, -0.131291, -0.0888015, -0.0878033,
                            -0.0321225, -0.053587, 0.157665, -0.00792996, 0.14707,
                            0.118949, 0.164109, 0.179299, 0.2048, 0.02664,
                            -0.112001, 0.14574, 0.0809537, 0.183332, 0.200962};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, 0.0908163);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv1_image.pgm");

        free(conv_image);
    }

    // 2
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {0.16622, 0.0980283, -0.154528, 0.0794689, -0.149562,
                            -0.0489036, -0.143187, -0.166235, 0.106519, 0.077899,
                            -0.167934, -0.0211066, -0.0540232, 0.065682, 0.201788,
                            -0.0860653, -0.023683, -0.104004, 0.126629, 0.00943944,
                            -0.0664681, -0.0823138, 0.110359, 0.111226, 0.132523};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, 0.0125789);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv2_image.pgm");

        free(conv_image);
    }

    // 3
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {-0.0193047, -0.148967, -0.129466, 0.117262, 0.0556525,
                            0.177834, 0.0465613, 0.064182, -0.0560068, -0.0997407,
                            -0.00353711, 0.0869634, -0.150035, -0.136275, -0.159884,
                            0.00542482, -0.0268633, 0.096359, -0.105669, -0.126976,
                            0.144686, -0.0741221, -0.100288, -0.0635448, -0.0388995};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, 0.027025);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv3_image.pgm");

        free(conv_image);
    }

    // 4
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {-0.193615, -0.154315, 0.0477816, 0.174346, -0.0321918,
                            0.0878409, -0.0274826, 0.0644111, 0.086192, 0.0962856,
                            -0.101726, 0.123956, 0.0949929, 0.0642185, -0.0504073,
                            0.248567, 0.0775272, -0.099301, 0.101063, 0.0305015,
                            -0.107393, -0.00542532, -0.0396422, 0.176904, 0.150425};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, 0.0268217);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv4_image.pgm");

        free(conv_image);
    }

    // 5
    {
        float_t* conv_image;
        conv_image = (float_t*) malloc((32-4)*(32-4)*sizeof(float_t));

        float_t kernel[25] = {-0.0159969, -0.0330545, 0.0151124, -0.155135, 0.0267772,
                            0.0603045, -0.119766, 0.0301981, -0.178801, -0.0762496,
                            0.116541, 0.16019, -0.207605, -0.0729984, 0.0412346,
                            0.214281, 0.092025, -0.0666976, -0.0286977, -0.121057,
                            0.182971, 0.154956, 0.0157412, -0.010005, 0.00333844};

        convolution2d_naive(t10k_images[index], 32, 32, conv_image, kernel, 5, -0.0190336);
        tanh_naive(conv_image, 28, 28, conv_image);

        write_pgm(28, 28, conv_image, "conv5_image.pgm");

        free(conv_image);
    }

    
    // free memory
    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    return 0;
}
