/** 
 * @brief LeNet implementation as provided by the Caffe LeNet MNIST example:
 * http://caffe.berkeleyvision.org/gathered/examples/mnist.html
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define MNIST
#define NUM 10000
//#define DEBUG
#define INDEX 0

#include "pico-cnn/pico-cnn.h"
#include <omp.h>

void usage() {
    printf("./caffe_lenet PATH_TO_MNIST_DATASET PATH_TO_WEIGHTS_FILE\n");
}

/**
 * @brief sorts the prediction and the labels (in place) of the network such that the label with the
 * highes prediction is at the front of the array (position 0)
 *
 * @param prediction (1 x length)
 * @param labels (1 x length)
 * @param length
 */
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

    char mnist_path[1024];
    char weights_path[1024];

    strcpy(mnist_path, argv[1]);
    strcpy(weights_path, argv[2]);

    unsigned int i, j, k;

    // read mnist t10k images
    char t10k_images_path[strlen(mnist_path) + 20];
    t10k_images_path[0] = '\0';
    strcat(t10k_images_path, mnist_path);
    strcat(t10k_images_path, "/t10k-images.idx3-ubyte");
    //strcat(t10k_images_path, "/train-images.idx3-ubyte");

    fp_t** t10k_images;
    int num_t10k_images;
    
    printf("reading images from '%s'\n", t10k_images_path);

    num_t10k_images = read_mnist_images(t10k_images_path, &t10k_images, NUM, 0, 0.0, 1.0);

    if(num_t10k_images < 1) {
        fprintf(stderr, "could not read mnist images from '%s'\n", t10k_images_path);
        return 1;
    }

    // read t10k labels
    char t10k_labels_path[strlen(mnist_path) + 20];
    t10k_labels_path[0] = '\0';
    strcat(t10k_labels_path, mnist_path);
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
    write_pgm(t10k_images[INDEX], 28, 28, "input.pgm");
    write_float(t10k_images[INDEX], 28, 28, "input.float");
    #endif

    // read kernels and biasses
    fp_t*** kernels;
    fp_t** biasses;

    printf("reading weights from '%s'\n", weights_path);

    if(read_weights(weights_path, &kernels, &biasses) != 0) {
        fprintf(stderr, "could not read weights from '%s'\n", weights_path);
        return 1;
    }

    restructure_fully_connected_kernel(&kernels[2][0], 800, 500);
    restructure_fully_connected_kernel(&kernels[3][0], 500, 10);

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

    // allocate memory for CNN and set pointers
    // C1
    fp_t** c1_output;
    c1_output = (fp_t**) malloc(20*sizeof(fp_t*));

    for(j = 0; j < 20; j++) {
        c1_output[j] = (fp_t*) malloc(24*24*sizeof(fp_t));
    }

    fp_t** c1_kernels = kernels[0];
    fp_t* c1_bias = biasses[0];

    

    // S2
    fp_t** s2_output;
    s2_output = (fp_t**) malloc(20*sizeof(fp_t*));

    for(j = 0; j < 20; j++) {
        s2_output[j] = (fp_t*) malloc(12*12*sizeof(fp_t));
    }

    // C3
    fp_t** c3_output;
    c3_output = (fp_t**) malloc(50*sizeof(fp_t*));

    for(j = 0; j < 50; j++) {
        c3_output[j] = (fp_t*) malloc(8*8*sizeof(fp_t));
    }

    fp_t** c3_intermediate = malloc(omp_get_max_threads()*sizeof(fp_t*));
    
    for(j = 0; j < omp_get_max_threads(); j++) {
        c3_intermediate[j] = malloc(8*8*sizeof(fp_t*));
    }
    
    fp_t** c3_kernels = kernels[1];
    fp_t* c3_bias = biasses[1];

    // S4 
    fp_t** s4_output;
    s4_output = (fp_t**) malloc(50*sizeof(fp_t*));

    for(j = 0; j < 50; j++) {
        s4_output[j] = (fp_t*) malloc(4*4*sizeof(fp_t));
    }

    // F5
    fp_t* s4_output_merged = (fp_t*) malloc(4*4*50*sizeof(fp_t));

    fp_t* f5_output;
    f5_output = (fp_t*) malloc(500*sizeof(fp_t));

    fp_t* f5_kernel = kernels[2][0];
    fp_t* f5_bias = biasses[2];
   
    // F6
    fp_t* f6_output;
    f6_output = (fp_t*) malloc(10*sizeof(fp_t));

    fp_t* f6_kernel = kernels[3][0];
    fp_t* f6_bias = biasses[3];


    printf("starting CNN\n");

    for(i = 0; i < NUM; i++) {
        // C1 input 28x28x1 -> output 24x24x20
        #pragma omp parallel for private(j)
        for(j = 0; j < 20; j++) {
            convolution2d_cpu_5x5_s1_valid(t10k_images[i], 28, 28, c1_output[j], c1_kernels[j], c1_bias[j]);
        }

        // make pgm C1
        #ifdef DEBUG
        if(i == INDEX) {
            float* c1_file_content = (fp_t*) malloc(24*24*20*sizeof(fp_t));
            for(j = 0; j < 20; j++) {
                memcpy(&c1_file_content[j*24*24], c1_output[j], 24*24*sizeof(fp_t));
            }
            
            write_pgm(c1_file_content, 20*24, 24, "c1_output.pgm");
            write_float(c1_file_content, 20*24, 24, "c1_output.float");
            free(c1_file_content);
        }
        #endif

        // S2 input 24x24x20 -> output 12x12x20
        #pragma omp parallel for private(j)
        for(j = 0; j < 20; j++) {
            max_pooling2d_cpu_2x2_s2(c1_output[j], 24, 24, s2_output[j]);
        }

        // make pgm S2
        #ifdef DEBUG
        if(i == INDEX) {
            float* s2_file_content = (fp_t*) malloc(12*12*20*sizeof(fp_t));
            for(j = 0; j < 20; j++) {
                memcpy(&s2_file_content[j*12*12], s2_output[j], 12*12*sizeof(fp_t));
            }
            write_pgm(s2_file_content, 20*12, 12, "s2_output.pgm");
            write_float(s2_file_content, 20*12, 12, "s2_output.float");
            free(s2_file_content);
        }
        #endif

        // C3 input 12x12x20 -> output 8x8x50
        #pragma omp parallel for private(j,k)
        for(j = 0; j < 50; j++) {
            convolution2d_cpu_5x5_s1_valid(s2_output[0], 12, 12, c3_output[j], c3_kernels[j*20], c3_bias[j]);

            for(k = 1; k < 20; k++) {
                convolution2d_cpu_5x5_s1_valid(s2_output[k], 12, 12, c3_intermediate[omp_get_thread_num()], c3_kernels[j*20+k], 0.0);
                add_image2d_cpu(c3_output[j], c3_intermediate[omp_get_thread_num()], 8, 8);
            }
        }

        // make pgm C3
        #ifdef DEBUG
        if(i == INDEX) {
            float* c3_file_content = (fp_t*) malloc(8*8*50*sizeof(fp_t));
            for(j = 0; j < 50; j++) {
                memcpy(&c3_file_content[j*8*8], c3_output[j], 8*8*sizeof(fp_t));
            }
            write_pgm(c3_file_content, 50*8, 8, "c3_output.pgm");
            write_float(c3_file_content, 50*8, 8, "c3_output.float");
            free(c3_file_content);
        }
        #endif

        // S4 input 8x8x50 -> output 8x8x50 
        #pragma omp parallel for private(j)
        for(j = 0; j < 50; j++) {
            max_pooling2d_cpu_2x2_s2(c3_output[j], 8, 8, s4_output[j]);
        }

        // make pgm S4
        #ifdef DEBUG
        if(i == INDEX) {
            float* s4_file_content = (fp_t*) malloc(4*4*50*sizeof(fp_t));
            for(j = 0; j < 50; j++) {
                memcpy(&s4_file_content[j*4*4], s4_output[j], 4*4*sizeof(fp_t));
            }
            write_pgm(s4_file_content, 50*4, 4, "s4_output.pgm");
            write_float(s4_file_content, 50*4, 4, "s4_output.float");
            free(s4_file_content);
        }
        #endif

        // F5 input 50x4x4 = 1x800 -> output 1x500
        // merge S4 output
        for(j = 0; j < 50; j++) {
            memcpy(&s4_output_merged[j*4*4], s4_output[j], 4*4*sizeof(fp_t));
        }

        #pragma omp parallel for private(j)
        for(j = 0; j < omp_get_max_threads(); j++) {
            fully_connected_cpu(s4_output_merged, 800, f5_output, 500, f5_kernel, f5_bias, (500/omp_get_max_threads())*omp_get_thread_num(), (500/omp_get_max_threads())*(omp_get_thread_num()+1));
        }

        // make pgm F5
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(f5_output, 1, 500, "f5_output.pgm");
            write_float(f5_output, 1, 500, "f5_output.float");
        }
        #endif

        // ReLU
        relu_cpu(f5_output, 1, 500, f5_output);

        // make pgm F5 ReLU
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(f5_output, 1, 500, "f5_relu_output.pgm");
            write_float(f5_output, 1, 500, "f5_relu_output.float");
        }
        #endif

        // F6 input 1x500 -> 1x10
        fully_connected_cpu(f5_output, 500, f6_output, 10, f6_kernel, f6_bias, 0, 10);

        // make pgm F6
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(f6_output, 1, 10, "f6_output.pgm");
            write_float(f6_output, 1, 10, "f6_output.float");
        }
        #endif

        // softmax
        softmax_cpu_single(f6_output, 1, 10, f6_output);

        // make pgm F6 softmax
        #ifdef DEBUG
        if(i == INDEX) {
            write_pgm(f6_output, 1, 10, "f6_softmax_output.pgm");
            write_float(f6_output, 1, 10, "f6_softmax_output.float");
        }
        #endif

        for(j = 0; j < 10; j++) {
            labels[j] = j;
        }

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
    }

    // freeing memory
    // C1
    for(j = 0; j < 20; j++) {
        free(c1_output[j]);
    }
    free(c1_output);

    // S2
    for(j = 0; j < 20; j++) {
        free(s2_output[j]);
    }
    free(s2_output);

    // C3
    for(j = 0; j < 50; j++) {
        free(c3_output[j]);
    }
    free(c3_output);

    for(j = 0; j < omp_get_max_threads(); j++) {
        free(c3_intermediate[j]);
    }
    free(c3_intermediate);

    // S4
    for(j = 0; j < 50; j++) {
        free(s4_output[j]);
    }
    free(s4_output);

    // F5
    free(s4_output_merged);
    free(f5_output);

    // F6
    free(f6_output);


    // data
    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    // weights
    for(i = 0; i < 20; i++) {
        free(kernels[0][i]);
    }
    free(kernels[0]);

    for(i = 0; i < 1000; i++) {
        free(kernels[1][i]);
    }
    free(kernels[1]);

    free(kernels[2][0]);
    free(kernels[2]);
    free(kernels[3][0]);
    free(kernels[3]);
    free(kernels);

    for(i = 0; i < 4; i++) {
        free(biasses[i]);
    }
    free(biasses);

    free(labels);

    // calculate and print results
    fp_t error_rate = 1.0-((fp_t) correct_predictions/((fp_t) NUM));

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
