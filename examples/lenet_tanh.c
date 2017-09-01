/** 
 * @brief LeNet-5 implementation as proposed by Yann LeCun
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include "pico-cnn/pico-cnn.h"

#include <float.h>

#define NUM 10000
//#define DEBUG
#define INDEX 0

void usage() {
    printf("./lenet_tanh PATH_TO_MNIST_DATASET PATH_TO_WEIGHTS_FILE\n");
}



/**
 * @brief takes the output of the fully-connected layer and converts into
 * values from [0,1]
 *
 * @param image (1 x width)
 * @param width
 * @param new_image (1 x width)
 */
void convert_prediction(const float_t* original_image, const uint16_t width, float_t* new_image) {
    uint16_t column;

    for(column = 0; column < width; column++) {
        new_image[column] = (original_image[column] + 1.0)/2.0;
    }
}

/**
 * @brief sorts the prediction and the labels (in place) of the network such that the label with the
 * highes prediction is at the front of the array (position 0)
 *
 * @param prediction (1 x length)
 * @param labels (1 x length)
 * @param length
 */
void sort_prediction(float_t* prediction, uint8_t* labels, const uint16_t length) {
    // simple bubble sort
    uint16_t i,j;

    float_t temp_prediction;
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

int main(int argc, char** argv) {

    if(argc == 1) {
        fprintf(stderr, "no path to mnist dataset and weights provided!\n");
        usage();
        return 1;
    }

    if(argc == 2) {
        fprintf(stderr, "no path to weights provided!\n");
        usage();
        return 1;
    }

    int i, j, k;

    // read mnist t10k images
    char t10k_images_path[strlen(argv[1]) + 20];
    t10k_images_path[0] = '\0';
    strcat(t10k_images_path, argv[1]);
    strcat(t10k_images_path, "/t10k-images.idx3-ubyte");
    //strcat(t10k_images_path, "/train-images.idx3-ubyte");

    float_t** t10k_images;
    int num_t10k_images;
    int padding = 2;
    
    printf("reading images from '%s'\n", t10k_images_path);

    num_t10k_images = read_mnist_images(t10k_images_path, &t10k_images, NUM, padding, -1.0, 1.0);

    if(num_t10k_images < 1) {
        fprintf(stderr, "could not read mnist images from '%s'\n", t10k_images_path);
        return 1;
    }

    // read t10k labels
    char t10k_labels_path[strlen(argv[1]) + 20];
    t10k_labels_path[0] = '\0';
    strcat(t10k_labels_path, argv[1]);
    strcat(t10k_labels_path, "/t10k-labels.idx1-ubyte");
    //strcat(t10k_labels_path, "/train-labels.idx1-ubyte");

    uint8_t* labels;
    labels = (uint8_t*) malloc(10*sizeof(uint8_t)); 

    uint8_t* t10k_labels;
    int num_t10k_labels;

    printf("reading labels from '%s'\n", t10k_labels_path);

    num_t10k_labels = read_mnist_labels(t10k_labels_path, &t10k_labels, NUM);

    if(num_t10k_images != num_t10k_labels) {
        fprintf(stderr, "%d images != %d labels\n", num_t10k_images, num_t10k_labels);
        return 1;
    }

    // make pgm of original image
    #ifdef DEBUG
    write_pgm(t10k_images[INDEX], 32, 32, "input.pgm");
    write_float(t10k_images[INDEX], 32, 32, "input.float");
    #endif

    // read kernels and biasses
    float_t*** kernels;
    float_t** biasses;

    printf("reading weights from '%s'\n", argv[2]);

    if(read_weights(argv[2], &kernels, &biasses) != 0) {
        fprintf(stderr, "could not read weights from '%s'\n", t10k_images_path);
        return 1;
    }

    int correct_predictions = 0;

    int confusion_matrix[10][10] = {
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
    


    for(i = 0; i < NUM; i++) {

        // C1 input 32x32x1 -> output 28x28x6
        float_t** c1_output;
        c1_output = (float_t**) malloc(6*sizeof(float_t*));
        
        float_t** c1_kernels = kernels[0];
        float_t* c1_bias = biasses[0];

        for(j = 0; j < 6; j++) {
            c1_output[j] = (float_t*) malloc(28*28*sizeof(float_t));
            convolution2d_naive(t10k_images[i], 32, 32, c1_output[j], c1_kernels[j], 5, c1_bias[j]);
            tanh_naive(c1_output[j], 28, 28, c1_output[j]);
        }

        // make pgm C1
        #ifdef DEBUG
        if(i == INDEX) {
            float* c1_file_content = (float_t*) malloc(28*28*6*sizeof(float_t));
            memcpy(&c1_file_content[0*28*28], c1_output[0], 28*28*sizeof(float_t));
            memcpy(&c1_file_content[1*28*28], c1_output[1], 28*28*sizeof(float_t));
            memcpy(&c1_file_content[2*28*28], c1_output[2], 28*28*sizeof(float_t));
            memcpy(&c1_file_content[3*28*28], c1_output[3], 28*28*sizeof(float_t));
            memcpy(&c1_file_content[4*28*28], c1_output[4], 28*28*sizeof(float_t));
            memcpy(&c1_file_content[5*28*28], c1_output[5], 28*28*sizeof(float_t));
            write_pgm(c1_file_content, 6*28, 28, "c1_output.pgm");
            write_float(c1_file_content, 6*28, 28, "c1_output.float");
            free(c1_file_content);
        }
        #endif

        // S2 input 28x28x6 -> output 14x14x6
        float_t** s2_output;
        s2_output = (float_t**) malloc(6*sizeof(float_t*));

        for(j = 0; j < 6; j++) {
            s2_output[j] = (float_t*) malloc(14*14*sizeof(float_t));
            max_pooling2d_naive(c1_output[j], 28, 28, s2_output[j], 2);
            tanh_naive(s2_output[j], 14, 14, s2_output[j]);
        }

        // make pgm S2
        #ifdef DEBUG
        if(i == INDEX) {
            float* s2_file_content = (float_t*) malloc(14*14*6*sizeof(float_t));
            memcpy(&s2_file_content[0*14*14], s2_output[0], 14*14*sizeof(float_t));
            memcpy(&s2_file_content[1*14*14], s2_output[1], 14*14*sizeof(float_t));
            memcpy(&s2_file_content[2*14*14], s2_output[2], 14*14*sizeof(float_t));
            memcpy(&s2_file_content[3*14*14], s2_output[3], 14*14*sizeof(float_t));
            memcpy(&s2_file_content[4*14*14], s2_output[4], 14*14*sizeof(float_t));
            memcpy(&s2_file_content[5*14*14], s2_output[5], 14*14*sizeof(float_t));
            write_pgm(s2_file_content, 6*14, 14, "s2_output.pgm");
            write_float(s2_file_content, 6*14, 14, "s2_output.float");
            free(s2_file_content);
        }
        #endif


        // C1 free memory
        for(j = 0; j < 6; j++) {
            free(c1_output[j]);
        }

        free(c1_output);

        // C3 input 14x14x6 -> output 10x10x16

        //   0  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        // 0 X              X   X   X           X   X   X   X       X   X
        // 1 X  X               X   X   X           X   X   X   X       X
        // 2 X  X   X               X   X   X           X       X   X   X
        // 3    X   X   X           X   X   X   X           X       X   X
        // 4        X   X   X           X   X   X   X       X   X       X
        // 5            X   X   X           X   X   X   X       X   X   X
       
        // kernel map
        //   0  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        // 0 0              24  30  36          54  60  66  72      84  90
        // 1 1  7               31  37  43          61  67  73  79      91
        // 2 2  8   14              38  44  50          68      80  86  92
        // 3    9   15  21          39  45  51  57          75      87  93
        // 4        16  22  28          46  52  58  64      76  82      94
        // 5            23  29  35          53  59  65  71      83  89  95
        
        float_t** c3_output;
        c3_output = (float_t**) malloc(16*sizeof(float_t*));

        float_t** c3_intermediate;
        c3_intermediate = (float_t**) malloc(6*sizeof(float_t*));

        float_t** c3_kernels = kernels[1];
        float_t* c3_bias = biasses[1];

        for(j = 0; j < 6; j++) {
            c3_intermediate[j] = (float_t*) malloc(10*10*sizeof(float_t)); 
        }

        // 0
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[0], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[1], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[2], c3_kernels[2], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[0];
        }

        c3_output[0] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[0], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[0], 10, 10, c3_output[0]);

        // 1
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[0], c3_kernels[7], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[1], c3_kernels[8], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[2], c3_kernels[9], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[1];
        }

        c3_output[1] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[1], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[1], 10, 10, c3_output[1]);

        // 2
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[0], c3_kernels[14], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[1], c3_kernels[15], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[2], c3_kernels[16], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[2];
        }

        c3_output[2] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[2], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[2], 10, 10, c3_output[2]);

        // 3
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[0], c3_kernels[21], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[1], c3_kernels[22], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[2], c3_kernels[23], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[3];
        }

        c3_output[3] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[3], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[3], 10, 10, c3_output[3]);
        
        // 4
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[24], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[1], c3_kernels[28], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[2], c3_kernels[29], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[4];
        }

        c3_output[4] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[4], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[4], 10, 10, c3_output[4]);
        
        // 5
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[30], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[31], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[2], c3_kernels[35], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[5];
        }

        c3_output[5] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[5], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[5], 10, 10, c3_output[5]);
        
        // 6
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[36], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[37], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[2], c3_kernels[38], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[3], c3_kernels[39], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[6];
        }

        c3_output[6] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[6], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[6], 10, 10, c3_output[6]);
        
        // 7
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[0], c3_kernels[43], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[1], c3_kernels[44], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[2], c3_kernels[45], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[3], c3_kernels[46], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[7];
        }

        c3_output[7] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[7], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[7], 10, 10, c3_output[7]);
        
        // 8
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[0], c3_kernels[50], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[1], c3_kernels[51], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[2], c3_kernels[52], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[53], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[8];
        }

        c3_output[8] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[8], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[8], 10, 10, c3_output[8]);
        
        // 9
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[54], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[1], c3_kernels[57], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[2], c3_kernels[58], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[59], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[9];
        }

        c3_output[9] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[9], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[9], 10, 10, c3_output[9]);
        
        // 10
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[60], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[61], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[2], c3_kernels[64], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[65], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[10];
        }

        c3_output[10] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[10], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[10], 10, 10, c3_output[10]);
        
        // 11
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[66], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[67], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[2], c3_kernels[68], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[71], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[11];
        }

        c3_output[11] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[11], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[11], 10, 10, c3_output[11]);
        
        // 12
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[72], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[73], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[2], c3_kernels[75], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[3], c3_kernels[76], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[12];
        }

        c3_output[12] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[12], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[12], 10, 10, c3_output[12]);
        
        // 13
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[0], c3_kernels[79], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[1], c3_kernels[80], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[2], c3_kernels[82], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[83], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[13];
        }

        c3_output[13] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[13], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[13], 10, 10, c3_output[13]);
        
        // 14
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[84], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[1], c3_kernels[86], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[2], c3_kernels[87], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[3], c3_kernels[89], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[14];
        }

        c3_output[14] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[14], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[14], 10, 10, c3_output[14]);
        
        // 15
        convolution2d_naive(s2_output[0], 14, 14, c3_intermediate[0], c3_kernels[90], 5, 0.0);
        convolution2d_naive(s2_output[1], 14, 14, c3_intermediate[1], c3_kernels[91], 5, 0.0);
        convolution2d_naive(s2_output[2], 14, 14, c3_intermediate[2], c3_kernels[92], 5, 0.0);
        convolution2d_naive(s2_output[3], 14, 14, c3_intermediate[3], c3_kernels[93], 5, 0.0);
        convolution2d_naive(s2_output[4], 14, 14, c3_intermediate[4], c3_kernels[94], 5, 0.0);
        convolution2d_naive(s2_output[5], 14, 14, c3_intermediate[5], c3_kernels[95], 5, 0.0);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[1], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[2], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[3], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[4], 10, 10);
        add_image2d_naive(c3_intermediate[0], c3_intermediate[5], 10, 10);

        for(j = 0; j < 10*10; j++) {
            c3_intermediate[0][j] += c3_bias[15];
        }

        c3_output[15] = (float_t*) malloc(10*10*sizeof(float_t));
        memcpy(c3_output[15], c3_intermediate[0], 10*10*sizeof(float_t));
        tanh_naive(c3_output[15], 10, 10, c3_output[15]);

        // free intermediate C3 outputs
        for(j = 0; j < 6; j++) {
            free(c3_intermediate[j]); 
        }

        free(c3_intermediate);

        // make pgm C3
        #ifdef DEBUG
        if(i == INDEX) {
            float* c3_file_content = (float_t*) malloc(10*10*16*sizeof(float_t));

            for(j = 0; j < 16; j++) {
                memcpy(&c3_file_content[j*10*10], c3_output[j], 10*10*sizeof(float_t));
            }

            write_pgm(c3_file_content, 16*10, 10, "c3_output.pgm");
            write_float(c3_file_content, 16*10, 10, "c3_output.float");
            free(c3_file_content);
        }
        #endif


        // S2 free memory
        for(j = 0; j < 6; j++) {
            free(s2_output[j]);
        }

        free(s2_output);

        // S4 input 10x10x16 -> output 5x5x16
        float_t** s4_output;
        s4_output = (float_t**) malloc(16*sizeof(float_t*));

        for(j = 0; j < 16; j++) {
            s4_output[j] = (float_t*) malloc(5*5*sizeof(float_t));
            max_pooling2d_naive(c3_output[j], 10, 10, s4_output[j], 2);
            tanh_naive(s4_output[j], 5, 5, s4_output[j]);
        }

        // make pgm S4
        #ifdef DEBUG
        if(i == INDEX) {
            
            float* s4_file_content = (float_t*) malloc(5*5*16*sizeof(float_t));

            for(j = 0; j < 16; j++) {
                memcpy(&s4_file_content[j*5*5], s4_output[j], 5*5*sizeof(float_t));
            }

            write_pgm(s4_file_content, 16*5, 5, "s4_output.pgm");
            write_float(s4_file_content, 16*5, 5, "s4_output.float");
            free(s4_file_content);
        }
        #endif

        // C3 free memory
        for(j = 0; j < 16; j++) {
            free(c3_output[j]);
        }

        free(c3_output);

        // C5 input 5x5x16 -> output 1x1x120
        float_t* c5_output;
        c5_output = (float_t*) malloc(120*sizeof(float_t));

        float_t** c5_kernels = kernels[2];
        float_t* c5_bias = biasses[2];

        float_t c5_intermediate;
    
        for(j = 0; j < 120; j++) {
            c5_output[j] = 0.0;
           
            
            for(k = 0; k < 16; k++) {
                convolution2d_naive(s4_output[k], 5, 5, &c5_intermediate, c5_kernels[j*16+k], 5, 0.0);
                c5_output[j] += c5_intermediate;
            }

            c5_output[j] += c5_bias[j];
        }
        
        tanh_naive(c5_output, 120, 1, c5_output);

        // make pgm C5
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(c5_output, 120, 1, "c5_output.pgm");
            write_float(c5_output, 120, 1, "c5_output.float");
        }
        #endif


        // S4 free memory
        for(j = 0; j < 16; j++) {
            free(s4_output[j]);
        }

        free(s4_output);

        // F6 input 1x1x120 -> output 1x10x1
        float_t* f6_output;
        f6_output = (float_t*) malloc(10*sizeof(float_t));

        float_t* f6_kernel = kernels[3][0];
        float_t* f6_bias = biasses[3];
        
        fully_connected_naive(c5_output, 120, f6_output, 10, f6_kernel, f6_bias);
        tanh_naive(f6_output, 10, 1, f6_output);

        // make pgm F6
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(f6_output, 1, 10, "f6_output.pgm");
            write_float(f6_output, 1, 10, "f6_output.float");
        }
        #endif

        // C5 free memory
        free(c5_output);


        for(j = 0; j < 10; j++) {
            labels[j] = j;
        }

        convert_prediction(f6_output, 10, f6_output);
        sort_prediction(f6_output, labels, 10);

        #ifdef DEBUG
        if(i == INDEX) {
            printf("%d\n", t10k_labels[i]);
            for(j = 0; j < 10; j++) {
                printf("%d: %f\n", labels[j], f6_output[j]);
            }
        }
        #endif


        if(t10k_labels[i] == labels[0]) {
            correct_predictions++;
        }

        confusion_matrix[labels[0]][t10k_labels[i]]++;

        // F6 free memory
        free(f6_output);
    }

    // freeing memory
    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);
    free(labels);

    for(i = 0; i < 6; i++) {
        free(kernels[0][i]);
    }
    for(i = 0; i < 96; i++) {
        free(kernels[1][i]);
    }
    for(i = 0; i < 1920; i++) {
        free(kernels[2][i]);
    }

    free(kernels[3][0]);

    for(i = 0; i < 4; i++) {
        free(biasses[i]);
    }

    // calculate and print results
    float_t error_rate = 1.0-((float_t) correct_predictions/10000.0);

    printf("error rate: %f (%d/%d)\n", error_rate, correct_predictions, num_t10k_images);

    printf("columns: actual label\n");
    printf("rows: predicted label\n");
    printf("*\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n");

    for(i = 0; i < 10; i++) {
        printf("%d\t", i);
        for(j = 0; j < 10; j++) {
            printf("%d\t", confusion_matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
