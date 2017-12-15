/** 
 * @brief AlexNet implementation as provided in the BVLC model zoo:
 * https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define IMAGENET
#define NUM 10
//#define DEBUG

// used to read the validation images
#define IMAGE_PREFIX "ILSVRC2012_val_"
#define NUMBER_LENGTH 8
#define IMAGE_SUFFIX ".JPEG"
#define START_NUMBER 1

// The first image will be read from:
// PATH_TO_IMAGE_NET_VAL_IMAGES/IMAGE_PREFIX_00000001.JPEG

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>
#include <omp.h>

void usage() {
    printf("./es_alexnet \\\n"); 
    printf("PATH_TO_ES_ALEXNET_WEIGHTS.weights \\\n");
    printf("PATH_TO_MEANS_FILE.means \\\n");
    printf("PATH_TO_IMAGE_NET_LABELS.txt \\\n");
    printf("PATH_TO_IMAGE_NET_VAL_LABELS.txt \\\n");
    printf("PATH_TO_IMAGE_NET_VAL_IMAGES\n");
}

/**
 * @brief reads a jpeg image from path_to_input_image substracts the means from
 * all channels and stores it in input_image
 *
 * @param path_to_input_image
 * @param input_image
 */
int read_input_image(const char* path_to_input_image, fp_t*** input_image, fp_t* means) {
    fp_t** pre_mean_input;

    uint16_t height;
    uint16_t width;

    if(read_jpeg(&pre_mean_input, path_to_input_image, 0.0, 255.0, &height, &width) != 0) {
        return 1;
    }

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
    (*input_image) = (fp_t**) malloc(3*sizeof(fp_t*));
    (*input_image)[0] = (fp_t*) malloc(227*227*sizeof(fp_t));
    (*input_image)[1] = (fp_t*) malloc(227*227*sizeof(fp_t));
    (*input_image)[2] = (fp_t*) malloc(227*227*sizeof(fp_t));
        
    uint16_t row;
    uint16_t column;

    int i;

    for(i = 0; i < 3; i++) {
        for(row = 0; row < height; row++) {
            for(column = 0; column < height; column++) {
                (*input_image)[i][row*width+column] = pre_mean_input[i][row*width+column] - means[i];
            }
        }
    }

    // free pre mean input image
    for(i = 0; i < 3; i++) {
        free(pre_mean_input[i]);
    }
    free(pre_mean_input);

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

    return 0;
}

/**
 * @brief sorts the prediction and the labels (in place) of the network such that the label with the
 * highes prediction is at the front of the array (position 0)
 *
 * @param prediction (1 x length)
 * @param labels (1 x length)
 * @param length
 */
void sort_prediction(fp_t* prediction, uint16_t* labels_pos, const uint16_t length) {
    // simple bubble sort
    uint16_t i,j;

    fp_t temp_prediction;
    uint16_t temp_label;

    for(i = 0; i < length-1; i++) {
        for(j = 0; j < length-1-i; j++) {
            if(prediction[j] < prediction[j+1]) {
                // swap
                temp_prediction = prediction[j];
                prediction[j] = prediction[j+1];
                prediction[j+1] = temp_prediction;

                temp_label = labels_pos[j];
                labels_pos[j] = labels_pos[j+1];
                labels_pos[j+1] = temp_label;
            }
        }
    }
}

int main(int argc, char** argv) {

    if(argc != 6) {
        fprintf(stderr, "too few or to many arguments!\n");
        usage();
        return 1;
    }

    char weights_path[1024];
    char means_path[1024];
    char labels_path[1024];
    char labels_val_path[1024];
    char images_val_path[1024];

    strcpy(weights_path, argv[1]);
    strcpy(means_path, argv[2]);
    strcpy(labels_path, argv[3]);
    strcpy(labels_val_path, argv[4]);
    strcpy(images_val_path, argv[5]);

    unsigned int i, j, k;

    // read kernels and biasses
    fp_t*** kernels;
    fp_t** biasses;

    printf("reading weights from '%s'\n", weights_path);

    if(read_weights(weights_path, &kernels, &biasses) != 0) {
        fprintf(stderr, "could not read weights from '%s'\n", weights_path);
        return 1;
    }

    // restructure fully-connected kernels
    restructure_fully_connected_kernel(&kernels[5][0], 9216, 4096);
    restructure_fully_connected_kernel(&kernels[6][0], 4096, 4096);
    restructure_fully_connected_kernel(&kernels[7][0], 4096, 1000);

    // read means
    printf("reading means from '%s'\n", means_path);

    fp_t* means = (fp_t*) malloc(3*sizeof(fp_t));

    if(read_means(means_path, means) != 0) {
        fprintf(stderr, "could not read means file '%s'!\n", means_path);
        return 1;
    }


    // read labels
    printf("reading labels from '%s'\n", labels_path);

    char** labels;
    int num_labels;
    num_labels = read_imagenet_labels(labels_path, &labels, 1000);

    if(num_labels != 1000) {
        fprintf(stderr, "could not read imagenet labels '%s'\n", labels_path);
        return 1;
    }


    // read validation labels
    printf("reading validation labels from '%s'\n", labels_val_path);

    uint32_t* validation_labels;
    int num_validation_labels;
    
    num_validation_labels = read_imagenet_validation_labels(labels_val_path, &validation_labels, 10000);
   
    if(num_validation_labels != 10000) {
        fprintf(stderr, "could not read imagenet validation labels '%s'\n", labels_val_path);
        return 1;
    }

    // allocate memory
    // conv1
    fp_t** conv1_output;
    conv1_output = (fp_t**) malloc(96*sizeof(fp_t*));

    for(j = 0; j < 96; j++) {
        conv1_output[j] = (fp_t*) malloc(55*55*sizeof(fp_t));
    }

    fp_t** conv1_intermediate = (fp_t**) malloc(omp_get_max_threads()*sizeof(fp_t*));
    
    for(j = 0; j < omp_get_max_threads(); j++) {
        conv1_intermediate[j] = (fp_t*) malloc(55*55*sizeof(fp_t));
    }

    fp_t** conv1_kernels = kernels[0];
    fp_t* conv1_bias = biasses[0];

    // norm1
    fp_t** norm1_output;
    norm1_output = (fp_t**) malloc(96*sizeof(fp_t*));
    
    for(j = 0; j < 96; j++) {
        norm1_output[j] = (fp_t*) malloc(55*55*sizeof(fp_t));
    }

    // pool1
    fp_t** pool1_output;
    pool1_output = (fp_t**) malloc(96*sizeof(fp_t*));
    
    for(j = 0; j < 96; j++) {
        pool1_output[j] = (fp_t*) malloc(27*27*sizeof(fp_t));
    }

    // conv2
    fp_t** conv2_output;
    conv2_output = (fp_t**) malloc(256*sizeof(fp_t*));

    for(j = 0; j < 256; j++) {
        conv2_output[j] = (fp_t*) malloc(27*27*sizeof(fp_t));
    }

    fp_t** conv2_intermediate = (fp_t**) malloc(omp_get_max_threads()*sizeof(fp_t*));

    for(j = 0; j < omp_get_max_threads(); j++) {
        conv2_intermediate[j] = (fp_t*) malloc(27*27*sizeof(fp_t));
    }

    fp_t** conv2_kernels = kernels[1];
    fp_t* conv2_bias = biasses[1];

    // norm2
    fp_t** norm2_output;
    norm2_output = (fp_t**) malloc(256*sizeof(fp_t*));
    
    for(j = 0; j < 256; j++) {
        norm2_output[j] = (fp_t*) malloc(27*27*sizeof(fp_t));
    }

    // pool2
    fp_t** pool2_output;
    pool2_output = (fp_t**) malloc(256*sizeof(fp_t*));
    
    for(j = 0; j < 256; j++) {
        pool2_output[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    // conv3
    fp_t** conv3_output;
    conv3_output = (fp_t**) malloc(384*sizeof(fp_t*));

    for(j = 0; j < 384; j++) {
        conv3_output[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv3_intermediate = (fp_t**) malloc(omp_get_max_threads()*sizeof(fp_t*));
    
    for(j = 0; j < omp_get_max_threads(); j++) {
        conv3_intermediate[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv3_kernels = kernels[2];
    fp_t* conv3_bias = biasses[2];

    // conv4
    fp_t** conv4_output;
    conv4_output = (fp_t**) malloc(384*sizeof(fp_t*));

    for(j = 0; j < 384; j++) {
        conv4_output[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv4_intermediate = (fp_t**) malloc(omp_get_max_threads()*sizeof(fp_t*));
    
    for(j = 0; j < omp_get_max_threads(); j++) {
        conv4_intermediate[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv4_kernels = kernels[3];
    fp_t* conv4_bias = biasses[3];

    // conv5
    fp_t** conv5_output;
    conv5_output = (fp_t**) malloc(256*sizeof(fp_t*));

    for(j = 0; j < 256; j++) {
        conv5_output[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv5_intermediate = (fp_t**) malloc(omp_get_max_threads()*sizeof(fp_t*));

    for(j = 0; j < omp_get_max_threads(); j++) {
        conv5_intermediate[j] = (fp_t*) malloc(13*13*sizeof(fp_t));
    }

    fp_t** conv5_kernels = kernels[4];
    fp_t* conv5_bias = biasses[4];

    // pool5
    fp_t** pool5_output;
    pool5_output = (fp_t**) malloc(256*sizeof(fp_t*));
    
    for(j = 0; j < 256; j++) {
        pool5_output[j] = (fp_t*) malloc(6*6*sizeof(fp_t));
    }

    // fc6
    fp_t* fc6_output;
    fc6_output = (fp_t*) malloc(4096*sizeof(fp_t));

    fp_t* fc6_kernel = kernels[5][0];
    fp_t* fc6_bias = biasses[5];



    uint32_t top1_correct_predictions = 0;
    uint32_t top5_correct_predictions = 0;

    printf("starting CNN\n");

    for(i = START_NUMBER; i < START_NUMBER+NUM; i++) {

        // read input images
        fp_t** input;
        char path_to_input_image[1024];

        // NUMBER_LENGTH -------------------|
        //                                  \/
        sprintf(path_to_input_image, "%s/%s%08d%s",  images_val_path, IMAGE_PREFIX, i, IMAGE_SUFFIX);
        printf("%s\n", path_to_input_image);

        if(read_input_image(path_to_input_image, &input, means) != 0) {
            fprintf(stderr, "could not read input image '%s'\n", path_to_input_image);
            return 1;
        }


        // conv1 input 227x227x3 -> output 55x55x96
        #pragma omp parallel for private(j)
        for(j = 0; j < 96; j++) {
            convolution2d_cpu_11x11_s4_valid(input[0], 227, 227, conv1_output[j], conv1_kernels[j*3], 0.0);

            convolution2d_cpu_11x11_s4_valid(input[1], 227, 227, conv1_intermediate[omp_get_thread_num()], conv1_kernels[j*3+1], 0.0);
            add_image2d_cpu(conv1_output[j], conv1_intermediate[omp_get_thread_num()], 55, 55);

            convolution2d_cpu_11x11_s4_valid(input[2], 227, 227, conv1_intermediate[omp_get_thread_num()], conv1_kernels[j*3+2], conv1_bias[j]);
            add_image2d_cpu(conv1_output[j], conv1_intermediate[omp_get_thread_num()], 55, 55);
        }

        // make pgm of input image
        #ifdef DEBUG
        fp_t* conv1_file_content = (fp_t*) malloc(55*55*96*sizeof(fp_t));
        for(j = 0; j < 96; j++) {
            memcpy(&conv1_file_content[j*55*55], conv1_output[j], 55*55*sizeof(fp_t));
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
        #pragma omp parallel for private(j)
        for(j = 0; j < 96; j++) {
            relu_cpu(conv1_output[j], 55, 55, conv1_output[j]);
        }

        // make pgm of relu1 output
        #ifdef DEBUG
        fp_t* relu1_file_content = (fp_t*) malloc(55*55*96*sizeof(fp_t));
        for(j = 0; j < 96; j++) {
            memcpy(&relu1_file_content[j*55*55], conv1_output[j], 55*55*sizeof(fp_t));
        }
        write_pgm(relu1_file_content, 96*55, 55, "relu1_output.pgm");
        write_float(relu1_file_content, 96*55, 55, "relu1_output.float");
        free(relu1_file_content);
        #endif


        // norm1
        local_response_normalization_cpu_single(conv1_output, 55, 55, 96, norm1_output, 0.0001, 0.75, 5);

        // make pgm of norm1 output
        #ifdef DEBUG
        fp_t* norm1_file_content = (fp_t*) malloc(55*55*96*sizeof(fp_t));
        for(j = 0; j < 96; j++) {
            memcpy(&norm1_file_content[j*55*55], norm1_output[j], 55*55*sizeof(fp_t));
        }
        write_pgm(norm1_file_content, 96*55, 55, "norm1_output.pgm");
        write_float(norm1_file_content, 96*55, 55, "norm1_output.float");
        free(norm1_file_content);
        #endif

        // pool1 55x55x96 -> 27x27x96
        #pragma omp parallel for private(j)
        for(j = 0; j < 96; j++) {
            max_pooling2d_cpu_3x3_s2(norm1_output[j], 55, 55, pool1_output[j]);
        }

        // make pgm of pool1 output
        #ifdef DEBUG
        fp_t* pool1_file_content = (fp_t*) malloc(27*27*96*sizeof(fp_t));
        for(j = 0; j < 96; j++) {
            memcpy(&pool1_file_content[j*27*27], pool1_output[j], 27*27*sizeof(fp_t));
        }
        write_pgm(pool1_file_content, 96*27, 27, "pool1_output.pgm");
        write_float(pool1_file_content, 96*27, 27, "pool1_output.float");
        free(pool1_file_content);
        #endif


        // conv2 27x27x96 -> 27x27x256
        #pragma omp parallel for private(j,k)
        for(j = 0; j < 256; j++) {
            convolution2d_cpu_5x5_s1_same(pool1_output[0], 27, 27, conv2_output[j], conv2_kernels[j*96], 0.0);

            for(k = 1; k < 95; k++) {
                convolution2d_cpu_5x5_s1_same(pool1_output[k], 27, 27, conv2_intermediate[omp_get_thread_num()], conv2_kernels[j*96+k], 0.0);
                add_image2d_cpu(conv2_output[j], conv2_intermediate[omp_get_thread_num()], 27, 27);
            }
            convolution2d_cpu_5x5_s1_same(pool1_output[95], 27, 27, conv2_intermediate[omp_get_thread_num()], conv2_kernels[j*96+95], conv2_bias[j]);
            add_image2d_cpu(conv2_output[j], conv2_intermediate[omp_get_thread_num()], 27, 27);
        }

        // make pgm of conv2 output
        #ifdef DEBUG
        fp_t* conv2_file_content = (fp_t*) malloc(27*27*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&conv2_file_content[j*27*27], conv2_output[j], 27*27*sizeof(fp_t));
        }
        write_pgm(conv2_file_content, 256*27, 27, "conv2_output.pgm");
        write_float(conv2_file_content, 256*27, 27, "conv2_output.float");
        free(conv2_file_content);
        #endif


        // relu2
        #pragma omp parallel for private(j)
        for(j = 0; j < 256; j++) {
            relu_cpu(conv2_output[j], 27, 27, conv2_output[j]);
        }

        // make pgm of relu1 output
        #ifdef DEBUG
        fp_t* relu2_file_content = (fp_t*) malloc(27*27*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&relu2_file_content[j*27*27], conv2_output[j], 27*27*sizeof(fp_t));
        }
        write_pgm(relu2_file_content, 256*27, 27, "relu2_output.pgm");
        write_float(relu2_file_content, 256*27, 27, "relu2_output.float");
        free(relu2_file_content);
        #endif


        // norm2
        local_response_normalization_cpu_single(conv2_output, 27, 27, 256, norm2_output, 0.0001, 0.75, 5);

        // make pgm of norm2 output
        #ifdef DEBUG
        fp_t* norm2_file_content = (fp_t*) malloc(27*27*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&norm2_file_content[j*27*27], norm2_output[j], 27*27*sizeof(fp_t));
        }
        write_pgm(norm2_file_content, 256*27, 27, "norm2_output.pgm");
        write_float(norm2_file_content, 256*27, 27, "norm2_output.float");
        free(norm2_file_content);
        #endif

        // pool2 27x27x256 -> 13x13x256
        #pragma omp parallel for private(j)
        for(j = 0; j < 256; j++) {
            max_pooling2d_cpu_3x3_s2(norm2_output[j], 27, 27, pool2_output[j]);
        }

        // make pgm of pool2 output
        #ifdef DEBUG
        fp_t* pool2_file_content = (fp_t*) malloc(13*13*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&pool2_file_content[j*13*13], pool2_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(pool2_file_content, 256*13, 13, "pool2_output.pgm");
        write_float(pool2_file_content, 256*13, 13, "pool2_output.float");
        free(pool2_file_content);
        #endif


        // conv3
        #pragma omp parallel for private(j,k)
        for(j = 0; j < 384; j++) {
            convolution2d_cpu_3x3_s1_same(pool2_output[0], 13, 13, conv3_output[j], conv3_kernels[j*256], 0.0);

            for(k = 1; k < 255; k++) {
                convolution2d_cpu_3x3_s1_same(pool2_output[k], 13, 13, conv3_intermediate[omp_get_thread_num()], conv3_kernels[j*256+k], 0.0);
                add_image2d_cpu(conv3_output[j], conv3_intermediate[omp_get_thread_num()], 13, 13);
            }
            convolution2d_cpu_3x3_s1_same(pool2_output[255], 13, 13, conv3_intermediate[omp_get_thread_num()], conv3_kernels[j*256+255], conv3_bias[j]);
            add_image2d_cpu(conv3_output[j], conv3_intermediate[omp_get_thread_num()], 13, 13);
        }

        // make pgm of conv2 output
        #ifdef DEBUG
        fp_t* conv3_file_content = (fp_t*) malloc(13*13*384*sizeof(fp_t));
        for(j = 0; j < 384; j++) {
            memcpy(&conv3_file_content[j*13*13], conv3_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(conv3_file_content, 256*13, 13, "conv3_output.pgm");
        write_float(conv3_file_content, 256*13, 13, "conv3_output.float");
        free(conv3_file_content);
        #endif


        // relu3
        #pragma omp parallel for private(j)
        for(j = 0; j < 384; j++) {
            relu_cpu(conv3_output[j], 13, 13, conv3_output[j]);
        }

        // make pgm of relu1 output
        #ifdef DEBUG
        fp_t* relu3_file_content = (fp_t*) malloc(13*13*384*sizeof(fp_t));
        for(j = 0; j < 384; j++) {
            memcpy(&relu3_file_content[j*13*13], conv3_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(relu3_file_content, 384*13, 13, "relu3_output.pgm");
        write_float(relu3_file_content, 384*13, 13, "relu3_output.float");
        free(relu3_file_content);
        #endif


        // conv4
        #pragma omp parallel for private(j,k)
        for(j = 0; j < 384; j++) {
            convolution2d_cpu_3x3_s1_same(conv3_output[0], 13, 13, conv4_output[j], conv4_kernels[j*384], 0.0);

            for(k = 1; k < 383; k++) {
                convolution2d_cpu_3x3_s1_same(conv3_output[k], 13, 13, conv4_intermediate[omp_get_thread_num()], conv4_kernels[j*384+k], 0.0);
                add_image2d_cpu(conv4_output[j], conv4_intermediate[omp_get_thread_num()], 13, 13);
            }
            convolution2d_cpu_3x3_s1_same(conv3_output[383], 13, 13, conv4_intermediate[omp_get_thread_num()], conv4_kernels[j*384+383], conv4_bias[j]);
            add_image2d_cpu(conv4_output[j], conv4_intermediate[omp_get_thread_num()], 13, 13);
        }

        // make pgm of conv4 output
        #ifdef DEBUG
        fp_t* conv4_file_content = (fp_t*) malloc(13*13*384*sizeof(fp_t));
        for(j = 0; j < 384; j++) {
            memcpy(&conv4_file_content[j*13*13], conv4_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(conv4_file_content, 384*13, 13, "conv4_output.pgm");
        write_float(conv4_file_content, 384*13, 13, "conv4_output.float");
        free(conv4_file_content);
        #endif


        // relu4
        #pragma omp parallel for private(j)
        for(j = 0; j < 384; j++) {
            relu_cpu(conv4_output[j], 13, 13, conv4_output[j]);
        }

        // make pgm of relu4 output
        #ifdef DEBUG
        fp_t* relu4_file_content = (fp_t*) malloc(13*13*384*sizeof(fp_t));
        for(j = 0; j < 384; j++) {
            memcpy(&relu4_file_content[j*13*13], conv4_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(relu4_file_content, 384*13, 13, "relu4_output.pgm");
        write_float(relu4_file_content, 384*13, 13, "relu4_output.float");
        free(relu4_file_content);
        #endif


        // conv5
        
        #pragma omp parallel for private(j,k)
        for(j = 0; j < 256; j++) {
            convolution2d_cpu_3x3_s1_same(conv4_output[0], 13, 13, conv5_output[j], conv5_kernels[j*384], 0.0);

            for(k = 1; k < 383; k++) {
                convolution2d_cpu_3x3_s1_same(conv4_output[k], 13, 13, conv5_intermediate[omp_get_thread_num()], conv5_kernels[j*384+k], 0.0);
                add_image2d_cpu(conv5_output[j], conv5_intermediate[omp_get_thread_num()], 13, 13);
            }
            convolution2d_cpu_3x3_s1_same(conv4_output[383], 13, 13, conv5_intermediate[omp_get_thread_num()], conv5_kernels[j*384+383], conv5_bias[j]);
            add_image2d_cpu(conv5_output[j], conv5_intermediate[omp_get_thread_num()], 13, 13);
        }

        // make pgm of conv5 output
        #ifdef DEBUG
        fp_t* conv5_file_content = (fp_t*) malloc(13*13*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&conv5_file_content[j*13*13], conv5_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(conv5_file_content, 256*13, 13, "conv5_output.pgm");
        write_float(conv5_file_content, 256*13, 13, "conv5_output.float");
        free(conv5_file_content);
        #endif


        // relu5
        for(j = 0; j < 256; j++) {
            relu_cpu(conv5_output[j], 13, 13, conv5_output[j]);
        }

        // make pgm of relu5 output
        #ifdef DEBUG
        fp_t* relu5_file_content = (fp_t*) malloc(13*13*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&relu5_file_content[j*13*13], conv5_output[j], 13*13*sizeof(fp_t));
        }
        write_pgm(relu5_file_content, 256*13, 13, "relu5_output.pgm");
        write_float(relu5_file_content, 256*13, 13, "relu5_output.float");
        free(relu5_file_content);
        #endif


        // pool5
        #pragma omp parallel for private(j)
        for(j = 0; j < 256; j++) {
            max_pooling2d_cpu_3x3_s2(conv5_output[j], 13, 13, pool5_output[j]);
        }

        // make pgm of pool2 output
        #ifdef DEBUG
        fp_t* pool5_file_content = (fp_t*) malloc(6*6*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&pool5_file_content[j*6*6], pool5_output[j], 6*6*sizeof(fp_t));
        }
        write_pgm(pool5_file_content, 256*6, 6, "pool5_output.pgm");
        write_float(pool5_file_content, 256*6, 6, "pool5_output.float");
        free(pool5_file_content);
        #endif

        // fc6 6x6x256 = 1x9216 -> 1x4096
        // merge S4 output
        fp_t* pool5_output_merged = (fp_t*) malloc(6*6*256*sizeof(fp_t));

        for(j = 0; j < 256; j++) {
            memcpy(&pool5_output_merged[j*6*6], pool5_output[j], 6*6*sizeof(fp_t));
        }
                
        #pragma omp parallel for private(j)
        for(j = 0; j < omp_get_max_threads(); j++) {
            fully_connected_cpu(pool5_output_merged, 9216, fc6_output, 4096, fc6_kernel, fc6_bias, (4096/omp_get_max_threads())*omp_get_thread_num(), (4096/omp_get_max_threads())*(omp_get_thread_num()+1));
        }

        // make pgm fc6 output
        #ifdef DEBUG
        write_pgm(fc6_output, 1, 4096, "fc6_output.pgm");
        write_float(fc6_output, 1, 4096, "fc6_output.float");
        #endif
        
        // free fc6_output
        free(pool5_output_merged);


        // relu6
        relu_cpu(fc6_output, 1, 4096, fc6_output);

        // make pgm of relu5 output
        #ifdef DEBUG
        write_pgm(fc6_output, 1, 4096, "relu6_output.pgm");
        write_float(fc6_output, 1, 4096, "relu6_output.float");
        #endif


        // drop6
        // do nothing


        // fc7 1x4096 -> 1x4096
        fp_t* fc7_output;
        fc7_output = (fp_t*) malloc(4096*sizeof(fp_t));

        fp_t* fc7_kernel = kernels[6][0];
        fp_t* fc7_bias = biasses[6];
        
        #pragma omp parallel for private(j)
        for(j = 0; j < omp_get_max_threads(); j++) {
            fully_connected_cpu(fc6_output, 4096, fc7_output, 4096, fc7_kernel, fc7_bias, (4096/omp_get_max_threads())*omp_get_thread_num(), (4096/omp_get_max_threads())*(omp_get_thread_num()+1));
        }

        // make pgm fc7 output
        #ifdef DEBUG
        write_pgm(fc7_output, 1, 4096, "fc7_output.pgm");
        write_float(fc7_output, 1, 4096, "fc7_output.float");
        #endif

        
        // relu7
        relu_cpu(fc7_output, 1, 4096, fc7_output);

        // make pgm of relu7 output
        #ifdef DEBUG
        write_pgm(fc7_output, 1, 4096, "relu7_output.pgm");
        write_float(fc7_output, 1, 4096, "relu7_output.float");
        #endif

        
        // drop8
        // do nothing


        // fc8 1x4096 -> 1x1000
        fp_t* fc8_output;
        fc8_output = (fp_t*) malloc(1000*sizeof(fp_t));

        fp_t* fc8_kernel = kernels[7][0];
        fp_t* fc8_bias = biasses[7];

        #pragma omp parallel for private(j)
        for(j = 0; j < omp_get_max_threads(); j++) {
            fully_connected_cpu(fc7_output, 4096, fc8_output, 1000, fc8_kernel, fc8_bias, (1000/omp_get_max_threads())*omp_get_thread_num(), (1000/omp_get_max_threads())*(omp_get_thread_num()+1));
        }

        // make pgm fc8 output
        #ifdef DEBUG
        write_pgm(fc8_output, 1, 1000, "fc8_output.pgm");
        write_float(fc8_output, 1, 1000, "fc8_output.float");
        #endif

        // free fc7 output
        free(fc7_output);


        // prob
        softmax_cpu_single(fc8_output, 1, 1000, fc8_output);

        // make pgm prob
        #ifdef DEBUG
        write_pgm(fc8_output, 1, 1000, "prob_output.pgm");
        write_float(fc8_output, 1, 1000, "prob_output.float");
        #endif

        // print prediction
        uint16_t* labels_pos;
        labels_pos = (uint16_t*) malloc(1000*sizeof(uint16_t));

        for(j = 0; j < 1000; j++) {
            labels_pos[j] = j;
        }

        sort_prediction(fc8_output, labels_pos, 1000);

        #ifdef DEBUG
        printf("prediction:\n");

        for(j = 0; j < 5; j++) {
            printf("%d %d %f %s\n", j+1, labels_pos[j], fc8_output[j], labels[labels_pos[j]]);
        }

        printf("actual: %d %s\n\n", validation_labels[i], labels[validation_labels[i]]);
        #endif

        // count top-1/top-5 correct predictions
        if(validation_labels[i] == labels_pos[0]) {
            top1_correct_predictions++;
            top5_correct_predictions++;
        }
        else if(validation_labels[i] == labels_pos[1]) {
            top5_correct_predictions++;
        }
        else if(validation_labels[i] == labels_pos[2]) {
            top5_correct_predictions++;
        }
        else if(validation_labels[i] == labels_pos[3]) {
            top5_correct_predictions++;
        }
        else if(validation_labels[i] == labels_pos[4]) {
            top5_correct_predictions++;
        }
       
        free(labels_pos);
        free(fc8_output);
    }

    // free memory
    // conv1
    for(j = 0; j < 96; j++) {
        free(conv1_output[j]);
    }
    free(conv1_output);

    for(j = 0; j < omp_get_max_threads(); j++) {
        free(conv1_intermediate[j]);
    }
    free(conv1_intermediate);

    // norm1
    for(j = 0; j < 96; j++) {
        free(norm1_output[j]);
    }
    free(norm1_output);

    // pool1
    for(j = 0; j < 96; j++) {
        free(pool1_output[j]);
    }
    free(pool1_output);

    // conv2
    for(j = 0; j < 256; j++) {
        free(conv2_output[j]);
    }
    free(conv2_output);

    for(j = 0; j < omp_get_max_threads(); j++) {
        free(conv2_intermediate[j]);
    }
    free(conv2_intermediate);

    // norm2
    for(j = 0; j < 256; j++) {
        free(norm2_output[j]);
    }
    free(norm2_output);

    // pool2
    for(j = 0; j < 256; j++) {
        free(pool2_output[j]);
    }
    free(pool2_output);

    // conv3
    for(j = 0; j < 384; j++) {
        free(conv3_output[j]);
    }
    free(conv3_output);

    for(j = 0; j < omp_get_max_threads(); j++) {
        free(conv3_intermediate[j]);
    }
    free(conv3_intermediate);

    // conv4
    for(j = 0; j < omp_get_max_threads(); j++) {
        free(conv4_intermediate[j]);
    }
    free(conv4_intermediate);

    for(j = 0; j < 384; j++) {
        free(conv4_output[j]);
    }
    free(conv4_output);

    // conv5
    for(j = 0; j < omp_get_max_threads(); j++) {
        free(conv5_intermediate[j]);
    }
    free(conv5_intermediate);

    for(j = 0; j < 256; j++) {
        free(conv5_output[j]);
    }
    free(conv5_output);

    // pool5
    for(j = 0; j < 256; j++) {
        free(pool5_output[j]);
    }
    free(pool5_output);

    // fc6 output
    free(fc6_output);

    // labels
    for(i = 0; i < 1000; i++) {
        free(labels[i]);
    }
    free(labels);

    // kernels
    for(i = 0; i < 288; i++) {
        free(kernels[0][i]);
    }
    free(kernels[0]);

    for(i = 0; i < 24576; i++) {
        free(kernels[1][i]);
    }
    free(kernels[1]);

    for(i = 0; i < 98304; i++) {
        free(kernels[2][i]);
    }
    free(kernels[2]);

    for(i = 0; i < 147456; i++) {
        free(kernels[3][i]);
    }
    free(kernels[3]);

    for(i = 0; i < 98304; i++) {
        free(kernels[4][i]);
    }
    free(kernels[4]);

    free(kernels[5][0]);
    free(kernels[5]);
    free(kernels[6][0]);
    free(kernels[6]);
    free(kernels[7][0]);
    free(kernels[7]);
    free(kernels);

    for(i = 0; i < 8; i++) {
        free(biasses[i]);
    }
    free(biasses);
 
    
    // free means
    free(means);   

    fp_t top1_error_rate = 1.0-((fp_t) top1_correct_predictions/((fp_t) NUM));
    fp_t top5_error_rate = 1.0-((fp_t) top5_correct_predictions/((fp_t) NUM));

    printf("top-1 error rate: %f (%d/%d)\n", top1_error_rate, top1_correct_predictions, NUM);
    printf("top-5 error rate: %f (%d/%d)\n", top5_error_rate, top5_correct_predictions, NUM);

    return 0;
}


