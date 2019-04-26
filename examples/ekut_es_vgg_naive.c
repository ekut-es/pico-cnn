/** 
 * @brief VGG-16 implementation
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define IMAGENET
#define NUM 100
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

void usage() {
    printf("./ekut_es_vgg_naive \\\n"); 
    printf("PATH_TO_EKUT_ES_VGG_WEIGHTS.weights \\\n");
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
    int i;

    if(read_jpeg(&pre_mean_input, path_to_input_image, 0.0, 255.0, &height, &width) != 0) {
        return 1;
    }

    // make pgm of pre mean input image
    #ifdef DEBUG
    float* pre_mean_input_file_content = (fp_t*) malloc(224*224*3*sizeof(fp_t));
    for(i = 0; i < 3; i++) {
        memcpy(&pre_mean_input_file_content[i*224*224], pre_mean_input[i], 224*224*sizeof(fp_t));
    }
    
    write_pgm(pre_mean_input_file_content, 3*224, 224, "pre_mean_input.pgm");
    write_float(pre_mean_input_file_content, 3*224, 224, "pre_mean_input.float");
    free(pre_mean_input_file_content);
    #endif

    // substract mean from each channel
    (*input_image) = (fp_t**) malloc(3*sizeof(fp_t*));
    (*input_image)[0] = (fp_t*) malloc(224*224*sizeof(fp_t));
    (*input_image)[1] = (fp_t*) malloc(224*224*sizeof(fp_t));
    (*input_image)[2] = (fp_t*) malloc(224*224*sizeof(fp_t));
        
    uint16_t row;
    uint16_t column;


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
    float* input_file_content = (fp_t*) malloc(224*224*3*sizeof(fp_t));
    for(i = 0; i < 3; i++) {
        memcpy(&input_file_content[i*224*224], (*input_image)[i], 224*224*sizeof(fp_t));
    }
    
    write_pgm(input_file_content, 3*224, 224, "input.pgm");
    write_float(input_file_content, 3*224, 224, "input.float");
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

        // conv1_1 input 224x224x3 -> output 224x224x64
        fp_t** conv1_1_output;
        conv1_1_output = (fp_t**) malloc(64*sizeof(fp_t*));

        for(j = 0; j < 64; j++) {
            conv1_1_output[j] = (fp_t*) malloc(224*224*sizeof(fp_t));
        }

        fp_t* conv1_1_intermediate = (fp_t*) malloc(224*224*sizeof(fp_t));

        fp_t** conv1_1_kernels = kernels[0];
        fp_t* conv1_1_bias = biasses[0];

        uint16_t kernel_number = 0;

        for(j = 0; j < 64; j++) {
            convolution2d_naive(input[0], 224, 224, conv1_1_output[j], conv1_1_kernels[kernel_number], 3, 1, 1, 0.0);
            kernel_number++;

            convolution2d_naive(input[1], 224, 224, conv1_1_intermediate, conv1_1_kernels[kernel_number], 3, 1, 1, 0.0);
            add_image2d_naive(conv1_1_output[j], conv1_1_intermediate, 224, 224);
            kernel_number++;

            convolution2d_naive(input[2], 224, 224, conv1_1_intermediate, conv1_1_kernels[kernel_number], 3, 1, 1, conv1_1_bias[j]);
            add_image2d_naive(conv1_1_output[j], conv1_1_intermediate, 224, 224);
            kernel_number++;
        }

        // make pgm of conv1_1 image
        #ifdef DEBUG
        fp_t* conv1_1_file_content = (fp_t*) malloc(224*224*64*sizeof(fp_t));
        for(j = 0; j < 64; j++) {
            memcpy(&conv1_1_file_content[j*224*224], conv1_1_output[j], 224*224*sizeof(fp_t));
        }
        
        write_pgm(conv1_1_file_content, 64*224, 224, "conv1_1_output.pgm");
        write_float(conv1_1_file_content, 64*224, 224, "conv1_1_output.float");
        free(conv1_1_file_content);
        #endif

        // free conv1_intermediate
        free(conv1_1_intermediate);

        // free input
        free(input[0]);
        free(input[1]);
        free(input[2]);
        free(input);

        
        // relu1_1
        for(j = 0; j < 64; j++) {
            relu_naive(conv1_1_output[j], 224, 224, conv1_1_output[j]);
        }

        // make pgm of relu1_1 output
        #ifdef DEBUG
        fp_t* relu1_1_file_content = (fp_t*) malloc(224*224*64*sizeof(fp_t));
        for(j = 0; j < 64; j++) {
            memcpy(&relu1_1_file_content[j*224*224], conv1_1_output[j], 224*224*sizeof(fp_t));
        }
        write_pgm(relu1_1_file_content, 64*224, 224, "relu1_1_output.pgm");
        write_float(relu1_1_file_content, 64*224, 224, "relu1_1_output.float");
        free(relu1_1_file_content);
        #endif

        
        // conv1_2 input 224x224x64 -> output 224x224x64
        fp_t** conv1_2_output;
        conv1_2_output = (fp_t**) malloc(64*sizeof(fp_t*));

        for(j = 0; j < 64; j++) {
            conv1_2_output[j] = (fp_t*) malloc(224*224*sizeof(fp_t));
        }

        fp_t* conv1_2_intermediate = (fp_t*) malloc(224*224*sizeof(fp_t));

        fp_t** conv1_2_kernels = kernels[1];
        fp_t* conv1_2_bias = biasses[1];


        for(j = 0; j < 64; j++) {
            convolution2d_naive(conv1_1_output[0], 224, 224, conv1_2_output[j], conv1_2_kernels[j*64], 3, 1, 1, 0.0);

            for(k = 1; k < 63; k++) {
                convolution2d_naive(conv1_1_output[k], 224, 224, conv1_2_intermediate, conv1_2_kernels[j*64+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv1_2_output[j], conv1_2_intermediate, 224, 224);
            }
            convolution2d_naive(conv1_1_output[63], 224, 224, conv1_2_intermediate, conv1_2_kernels[j*64+63], 3, 1, 1, conv1_2_bias[j]);
            add_image2d_naive(conv1_2_output[j], conv1_2_intermediate, 224, 224);
        }

        // free conv1_1 output
        for(j = 0; j < 64; j++) {
            free(conv1_1_output[j]);
        }
        free(conv1_1_output);

        // make pgm of conv1_2 image
        #ifdef DEBUG
        fp_t* conv1_2_file_content = (fp_t*) malloc(224*224*64*sizeof(fp_t));
        for(j = 0; j < 64; j++) {
            memcpy(&conv1_2_file_content[j*224*224], conv1_2_output[j], 224*224*sizeof(fp_t));
        }
        
        write_pgm(conv1_2_file_content, 64*224, 224, "conv1_2_output.pgm");
        write_float(conv1_2_file_content, 64*224, 224, "conv1_2_output.float");
        free(conv1_2_file_content);
        #endif

        // free conv1_intermediate
        free(conv1_2_intermediate);

        
        // relu1_2
        for(j = 0; j < 64; j++) {
            relu_naive(conv1_2_output[j], 224, 224, conv1_2_output[j]);
        }

        // make pgm of relu1_2 output
        #ifdef DEBUG
        fp_t* relu1_2_file_content = (fp_t*) malloc(224*224*64*sizeof(fp_t));
        for(j = 0; j < 64; j++) {
            memcpy(&relu1_2_file_content[j*224*224], conv1_2_output[j], 224*224*sizeof(fp_t));
        }
        write_pgm(relu1_2_file_content, 64*224, 224, "relu1_2_output.pgm");
        write_float(relu1_2_file_content, 64*224, 224, "relu1_2_output.float");
        free(relu1_2_file_content);
        #endif

        
        // pool1 input 224x224x64 -> output 112x112x64
        fp_t** pool1_output;
        pool1_output = (fp_t**) malloc(64*sizeof(fp_t*));
        
        for(j = 0; j < 64; j++) {
            pool1_output[j] = (fp_t*) malloc(112*112*sizeof(fp_t));
        }

        for(j = 0; j < 64; j++) {
            max_pooling2d_naive(conv1_2_output[j], 224, 224, pool1_output[j], 2, 2);
        }

        // make pgm of pool1 output
        #ifdef DEBUG
        fp_t* pool1_file_content = (fp_t*) malloc(112*112*64*sizeof(fp_t));
        for(j = 0; j < 64; j++) {
            memcpy(&pool1_file_content[j*112*112], pool1_output[j], 112*112*sizeof(fp_t));
        }
        write_pgm(pool1_file_content, 64*112, 112, "pool1_output.pgm");
        write_float(pool1_file_content, 64*112, 112, "pool1_output.float");
        free(pool1_file_content);
        #endif

        // free conv1_2 output
        for(j = 0; j < 64; j++) {
            free(conv1_2_output[j]);
        }
        free(conv1_2_output);

        
        // conv2_1 input 112x112x64 -> output 112x112x128
        fp_t** conv2_1_output;
        conv2_1_output = (fp_t**) malloc(128*sizeof(fp_t*));

        for(j = 0; j < 128; j++) {
            conv2_1_output[j] = (fp_t*) malloc(112*112*sizeof(fp_t));
        }

        fp_t* conv2_1_intermediate = (fp_t*) malloc(112*112*sizeof(fp_t));

        fp_t** conv2_1_kernels = kernels[2];
        fp_t* conv2_1_bias = biasses[2];


        for(j = 0; j < 128; j++) {
            convolution2d_naive(pool1_output[0], 112, 112, conv2_1_output[j], conv2_1_kernels[j*64], 3, 1, 1, 0.0);

            for(k = 1; k < 63; k++) {
                convolution2d_naive(pool1_output[k], 112, 112, conv2_1_intermediate, conv2_1_kernels[j*64+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv2_1_output[j], conv2_1_intermediate, 112, 112);
            }
            convolution2d_naive(pool1_output[63], 112, 112, conv2_1_intermediate, conv2_1_kernels[j*64+63], 3, 1, 1, conv2_1_bias[j]);
            add_image2d_naive(conv2_1_output[j], conv2_1_intermediate, 112, 112);
        }


        // make pgm of conv2_1 image
        #ifdef DEBUG
        fp_t* conv2_1_file_content = (fp_t*) malloc(112*112*128*sizeof(fp_t));
        for(j = 0; j < 128; j++) {
            memcpy(&conv2_1_file_content[j*112*112], conv2_1_output[j], 112*112*sizeof(fp_t));
        }
        
        write_pgm(conv2_1_file_content, 128*112, 112, "conv2_1_output.pgm");
        write_float(conv2_1_file_content, 128*112, 112, "conv2_1_output.float");
        free(conv2_1_file_content);
        #endif

        // free conv2_1_intermediate
        free(conv2_1_intermediate);

        // free pool1 output
        for(j = 0; j < 64; j++) {
            free(pool1_output[j]);
        }
        free(pool1_output);


        // relu2_1
        for(j = 0; j < 128; j++) {
            relu_naive(conv2_1_output[j], 112, 112, conv2_1_output[j]);
        }

        // make pgm of relu2_1 output
        #ifdef DEBUG
        fp_t* relu2_1_file_content = (fp_t*) malloc(112*112*128*sizeof(fp_t));
        for(j = 0; j < 128; j++) {
            memcpy(&relu2_1_file_content[j*112*112], conv2_1_output[j], 112*112*sizeof(fp_t));
        }
        write_pgm(relu2_1_file_content, 128*112, 112, "relu2_1_output.pgm");
        write_float(relu2_1_file_content, 128*112, 112, "relu2_1_output.float");
        free(relu2_1_file_content);
        #endif

        
        // conv2_2 input 112x112x128 -> output 112x112x128
        fp_t** conv2_2_output;
        conv2_2_output = (fp_t**) malloc(128*sizeof(fp_t*));

        for(j = 0; j < 128; j++) {
            conv2_2_output[j] = (fp_t*) malloc(112*112*sizeof(fp_t));
        }

        fp_t* conv2_2_intermediate = (fp_t*) malloc(112*112*sizeof(fp_t));

        fp_t** conv2_2_kernels = kernels[3];
        fp_t* conv2_2_bias = biasses[3];


        for(j = 0; j < 128; j++) {
            convolution2d_naive(conv2_1_output[0], 112, 112, conv2_2_output[j], conv2_2_kernels[j*128], 3, 1, 1, 0.0);

            for(k = 1; k < 127; k++) {
                convolution2d_naive(conv2_1_output[k], 112, 112, conv2_2_intermediate, conv2_2_kernels[j*128+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv2_2_output[j], conv2_2_intermediate, 112, 112);
            }
            convolution2d_naive(conv2_1_output[127], 112, 112, conv2_2_intermediate, conv2_2_kernels[j*128+127], 3, 1, 1, conv2_2_bias[j]);
            add_image2d_naive(conv2_2_output[j], conv2_2_intermediate, 112, 112);
        }

        // make pgm of conv2_2 image
        #ifdef DEBUG
        fp_t* conv2_2_file_content = (fp_t*) malloc(112*112*128*sizeof(fp_t));
        for(j = 0; j < 128; j++) {
            memcpy(&conv2_2_file_content[j*112*112], conv2_2_output[j], 112*112*sizeof(fp_t));
        }
        
        write_pgm(conv2_2_file_content, 128*112, 112, "conv2_2_output.pgm");
        write_float(conv2_2_file_content, 128*112, 112, "conv2_2_output.float");
        free(conv2_2_file_content);
        #endif

        // free conv2_2_intermediate
        free(conv2_2_intermediate);

        // free conv2_1 output
        for(j = 0; j < 128; j++) {
            free(conv2_1_output[j]);
        }
        free(conv2_1_output);


        // relu2_2
        for(j = 0; j < 128; j++) {
            relu_naive(conv2_2_output[j], 112, 112, conv2_2_output[j]);
        }

        // make pgm of relu2_2 output
        #ifdef DEBUG
        fp_t* relu2_2_file_content = (fp_t*) malloc(112*112*128*sizeof(fp_t));
        for(j = 0; j < 128; j++) {
            memcpy(&relu2_2_file_content[j*112*112], conv2_2_output[j], 112*112*sizeof(fp_t));
        }
        write_pgm(relu2_2_file_content, 128*112, 112, "relu2_2_output.pgm");
        write_float(relu2_2_file_content, 128*112, 112, "relu2_2_output.float");
        free(relu2_2_file_content);
        #endif

        
        // pool2 input 112x112x128 -> output 56x56x128
        fp_t** pool2_output;
        pool2_output = (fp_t**) malloc(128*sizeof(fp_t*));
        
        for(j = 0; j < 128; j++) {
            pool2_output[j] = (fp_t*) malloc(56*56*sizeof(fp_t));
        }

        for(j = 0; j < 128; j++) {
            max_pooling2d_naive(conv2_2_output[j], 112, 112, pool2_output[j], 2, 2);
        }

        // make pgm of pool2 output
        #ifdef DEBUG
        fp_t* pool2_file_content = (fp_t*) malloc(56*56*128*sizeof(fp_t));
        for(j = 0; j < 128; j++) {
            memcpy(&pool2_file_content[j*56*56], pool2_output[j], 56*56*sizeof(fp_t));
        }
        write_pgm(pool2_file_content, 128*56, 56, "pool2_output.pgm");
        write_float(pool2_file_content, 128*56, 56, "pool2_output.float");
        free(pool2_file_content);
        #endif

        // free conv2_2 output
        for(j = 0; j < 128; j++) {
            free(conv2_2_output[j]);
        }
        free(conv2_2_output);

        
        // conv3_1 input 56x56x128 -> output 56x56x256
        fp_t** conv3_1_output;
        conv3_1_output = (fp_t**) malloc(256*sizeof(fp_t*));

        for(j = 0; j < 256; j++) {
            conv3_1_output[j] = (fp_t*) malloc(56*56*sizeof(fp_t));
        }

        fp_t* conv3_1_intermediate = (fp_t*) malloc(56*56*sizeof(fp_t));

        fp_t** conv3_1_kernels = kernels[4];
        fp_t* conv3_1_bias = biasses[4];


        for(j = 0; j < 256; j++) {
            convolution2d_naive(pool2_output[0], 56, 56, conv3_1_output[j], conv3_1_kernels[j*128], 3, 1, 1, 0.0);

            for(k = 1; k < 127; k++) {
                convolution2d_naive(pool2_output[k], 56, 56, conv3_1_intermediate, conv3_1_kernels[j*128+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv3_1_output[j], conv3_1_intermediate, 56, 56);
            }
            convolution2d_naive(pool2_output[127], 56, 56, conv3_1_intermediate, conv3_1_kernels[j*128+127], 3, 1, 1, conv3_1_bias[j]);
            add_image2d_naive(conv3_1_output[j], conv3_1_intermediate, 56, 56);
        }

        // make pgm of conv3_1 image
        #ifdef DEBUG
        fp_t* conv3_1_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&conv3_1_file_content[j*56*56], conv3_1_output[j], 56*56*sizeof(fp_t));
        }
        
        write_pgm(conv3_1_file_content, 256*56, 56, "conv3_1_output.pgm");
        write_float(conv3_1_file_content, 256*56, 56, "conv3_1_output.float");
        free(conv3_1_file_content);
        #endif

        // free conv3_1_intermediate
        free(conv3_1_intermediate);

        // free pool2 output
        for(j = 0; j < 128; j++) {
            free(pool2_output[j]);
        }
        free(pool2_output);

        
        // relu3_1
        for(j = 0; j < 256; j++) {
            relu_naive(conv3_1_output[j], 56, 56, conv3_1_output[j]);
        }

        // make pgm of relu2_2 output
        #ifdef DEBUG
        fp_t* relu3_1_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&relu3_1_file_content[j*56*56], conv3_1_output[j], 56*56*sizeof(fp_t));
        }
        write_pgm(relu3_1_file_content, 256*56, 56, "relu3_1_output.pgm");
        write_float(relu3_1_file_content, 256*56, 56, "relu3_1_output.float");
        free(relu3_1_file_content);
        #endif


        // conv3_2 input 56x56x256 -> output 56x56x256
        fp_t** conv3_2_output;
        conv3_2_output = (fp_t**) malloc(256*sizeof(fp_t*));

        for(j = 0; j < 256; j++) {
            conv3_2_output[j] = (fp_t*) malloc(56*56*sizeof(fp_t));
        }

        fp_t* conv3_2_intermediate = (fp_t*) malloc(56*56*sizeof(fp_t));

        fp_t** conv3_2_kernels = kernels[5];
        fp_t* conv3_2_bias = biasses[5];


        for(j = 0; j < 256; j++) {
            convolution2d_naive(conv3_1_output[0], 56, 56, conv3_2_output[j], conv3_2_kernels[j*256], 3, 1, 1, 0.0);

            for(k = 1; k < 255; k++) {
                convolution2d_naive(conv3_1_output[k], 56, 56, conv3_2_intermediate, conv3_2_kernels[j*256+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv3_2_output[j], conv3_2_intermediate, 56, 56);
            }
            convolution2d_naive(conv3_1_output[255], 56, 56, conv3_2_intermediate, conv3_2_kernels[j*256+255], 3, 1, 1, conv3_2_bias[j]);
            add_image2d_naive(conv3_2_output[j], conv3_2_intermediate, 56, 56);
        }

        // make pgm of conv3_2 image
        #ifdef DEBUG
        fp_t* conv3_2_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&conv3_2_file_content[j*56*56], conv3_2_output[j], 56*56*sizeof(fp_t));
        }
        
        write_pgm(conv3_2_file_content, 256*56, 56, "conv3_2_output.pgm");
        write_float(conv3_2_file_content, 256*56, 56, "conv3_2_output.float");
        free(conv3_2_file_content);
        #endif

        // free conv3_2_intermediate
        free(conv3_2_intermediate);


        // free conv3_1 output
        for(j = 0; j < 256; j++) {
            free(conv3_1_output[j]);
        }
        free(conv3_1_output);

        
        // relu3_2
        for(j = 0; j < 256; j++) {
            relu_naive(conv3_2_output[j], 56, 56, conv3_2_output[j]);
        }

        // make pgm of relu2_2 output
        #ifdef DEBUG
        fp_t* relu3_2_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&relu3_2_file_content[j*56*56], conv3_2_output[j], 56*56*sizeof(fp_t));
        }
        write_pgm(relu3_2_file_content, 256*56, 56, "relu3_2_output.pgm");
        write_float(relu3_2_file_content, 256*56, 56, "relu3_2_output.float");
        free(relu3_2_file_content);
        #endif


        // conv3_3 input 56x56x256 -> output 56x56x256
        fp_t** conv3_3_output;
        conv3_3_output = (fp_t**) malloc(256*sizeof(fp_t*));

        for(j = 0; j < 256; j++) {
            conv3_3_output[j] = (fp_t*) malloc(56*56*sizeof(fp_t));
        }

        fp_t* conv3_3_intermediate = (fp_t*) malloc(56*56*sizeof(fp_t));

        fp_t** conv3_3_kernels = kernels[6];
        fp_t* conv3_3_bias = biasses[6];


        for(j = 0; j < 256; j++) {
            convolution2d_naive(conv3_2_output[0], 56, 56, conv3_3_output[j], conv3_3_kernels[j*256], 3, 1, 1, 0.0);

            for(k = 1; k < 255; k++) {
                convolution2d_naive(conv3_2_output[k], 56, 56, conv3_3_intermediate, conv3_3_kernels[j*256+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv3_3_output[j], conv3_3_intermediate, 56, 56);
            }
            convolution2d_naive(conv3_2_output[255], 56, 56, conv3_3_intermediate, conv3_3_kernels[j*256+255], 3, 1, 1, conv3_3_bias[j]);
            add_image2d_naive(conv3_3_output[j], conv3_3_intermediate, 56, 56);
        }

        // make pgm of conv3_3 image
        #ifdef DEBUG
        fp_t* conv3_3_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&conv3_3_file_content[j*56*56], conv3_3_output[j], 56*56*sizeof(fp_t));
        }
        
        write_pgm(conv3_3_file_content, 256*56, 56, "conv3_3_output.pgm");
        write_float(conv3_3_file_content, 256*56, 56, "conv3_3_output.float");
        free(conv3_3_file_content);
        #endif

        // free conv3_3_intermediate
        free(conv3_3_intermediate);

        // free conv3_2 output
        for(j = 0; j < 256; j++) {
            free(conv3_2_output[j]);
        }
        free(conv3_2_output);


        // relu3_3
        for(j = 0; j < 256; j++) {
            relu_naive(conv3_3_output[j], 56, 56, conv3_3_output[j]);
        }

        // make pgm of relu3_3 output
        #ifdef DEBUG
        fp_t* relu3_3_file_content = (fp_t*) malloc(56*56*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&relu3_3_file_content[j*56*56], conv3_3_output[j], 56*56*sizeof(fp_t));
        }
        write_pgm(relu3_3_file_content, 256*56, 56, "relu3_3_output.pgm");
        write_float(relu3_3_file_content, 256*56, 56, "relu3_3_output.float");
        free(relu3_3_file_content);
        #endif

        
        // pool3 input 56x56x256 -> output 28x28x256
        fp_t** pool3_output;
        pool3_output = (fp_t**) malloc(256*sizeof(fp_t*));
        
        for(j = 0; j < 256; j++) {
            pool3_output[j] = (fp_t*) malloc(28*28*sizeof(fp_t));
        }

        for(j = 0; j < 256; j++) {
            max_pooling2d_naive(conv3_3_output[j], 56, 56, pool3_output[j], 2, 2);
        }

        // make pgm of pool3 output
        #ifdef DEBUG
        fp_t* pool3_file_content = (fp_t*) malloc(28*28*256*sizeof(fp_t));
        for(j = 0; j < 256; j++) {
            memcpy(&pool3_file_content[j*28*28], pool3_output[j], 28*28*sizeof(fp_t));
        }
        write_pgm(pool3_file_content, 256*28, 28, "pool3_output.pgm");
        write_float(pool3_file_content, 256*28, 28, "pool3_output.float");
        free(pool3_file_content);
        #endif

        // free conv3_3 output
        for(j = 0; j < 256; j++) {
            free(conv3_3_output[j]);
        }
        free(conv3_3_output);


        // conv4_1 input 28x28x256 -> output 28x28x512
        fp_t** conv4_1_output;
        conv4_1_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv4_1_output[j] = (fp_t*) malloc(28*28*sizeof(fp_t));
        }

        fp_t* conv4_1_intermediate = (fp_t*) malloc(28*28*sizeof(fp_t));

        fp_t** conv4_1_kernels = kernels[7];
        fp_t* conv4_1_bias = biasses[7];


        for(j = 0; j < 512; j++) {
            convolution2d_naive(pool3_output[0], 28, 28, conv4_1_output[j], conv4_1_kernels[j*256], 3, 1, 1, 0.0);

            for(k = 1; k < 255; k++) {
                convolution2d_naive(pool3_output[k], 28, 28, conv4_1_intermediate, conv4_1_kernels[j*256+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv4_1_output[j], conv4_1_intermediate, 28, 28);
            }
            convolution2d_naive(pool3_output[255], 28, 28, conv4_1_intermediate, conv4_1_kernels[j*256+255], 3, 1, 1, conv4_1_bias[j]);
            add_image2d_naive(conv4_1_output[j], conv4_1_intermediate, 28, 28);
        }

        // make pgm of conv4_1 image
        #ifdef DEBUG
        fp_t* conv4_1_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv4_1_file_content[j*28*28], conv4_1_output[j], 28*28*sizeof(fp_t));
        }
        write_pgm(conv4_1_file_content, 512*28, 28, "conv4_1_output.pgm");
        write_float(conv4_1_file_content, 512*28, 28, "conv4_1_output.float");
        free(conv4_1_file_content);
        #endif

        // free conv4_1_intermediate
        free(conv4_1_intermediate);

        // free pool3 output
        for(j = 0; j < 256; j++) {
            free(pool3_output[j]);
        }
        free(pool3_output);


        // relu4_1
        for(j = 0; j < 512; j++) {
            relu_naive(conv4_1_output[j], 28, 28, conv4_1_output[j]);
        }

        // make pgm of relu4_1 output
        #ifdef DEBUG
        fp_t* relu4_1_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&relu4_1_file_content[j*28*28], conv4_1_output[j], 28*28*sizeof(fp_t));
        }
        write_pgm(relu4_1_file_content, 512*28, 28, "relu4_1_output.pgm");
        write_float(relu4_1_file_content, 512*28, 28, "relu4_1_output.float");
        free(relu4_1_file_content);
        #endif


        // conv4_2 input 28x28x512 -> output 28x28x512
        fp_t** conv4_2_output;
        conv4_2_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv4_2_output[j] = (fp_t*) malloc(28*28*sizeof(fp_t));
        }

        fp_t* conv4_2_intermediate = (fp_t*) malloc(28*28*sizeof(fp_t));

        fp_t** conv4_2_kernels = kernels[8];
        fp_t* conv4_2_bias = biasses[8];


        for(j = 0; j < 512; j++) {
            convolution2d_naive(conv4_1_output[0], 28, 28, conv4_2_output[j], conv4_2_kernels[j*512], 3, 1, 1, 0.0);

            for(k = 1; k < 511; k++) {
                convolution2d_naive(conv4_1_output[k], 28, 28, conv4_2_intermediate, conv4_2_kernels[j*512+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv4_2_output[j], conv4_2_intermediate, 28, 28);
            }
            convolution2d_naive(conv4_1_output[511], 28, 28, conv4_2_intermediate, conv4_2_kernels[j*512+511], 3, 1, 1, conv4_2_bias[j]);
            add_image2d_naive(conv4_2_output[j], conv4_2_intermediate, 28, 28);
        }

        // make pgm of conv4_2 image
        #ifdef DEBUG
        fp_t* conv4_2_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv4_2_file_content[j*28*28], conv4_2_output[j], 28*28*sizeof(fp_t));
        }
        
        write_pgm(conv4_2_file_content, 512*28, 28, "conv4_2_output.pgm");
        write_float(conv4_2_file_content, 512*28, 28, "conv4_2_output.float");
        free(conv4_2_file_content);
        #endif

        // free conv4_2 intermediate
        free(conv4_2_intermediate);

        // free conv4_1 output
        for(j = 0; j < 512; j++) {
            free(conv4_1_output[j]);
        }
        free(conv4_1_output);


        // relu4_2
        for(j = 0; j < 512; j++) {
            relu_naive(conv4_2_output[j], 28, 28, conv4_2_output[j]);
        }

        // make pgm of relu4_2 output
        #ifdef DEBUG
        fp_t* relu4_2_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&relu4_2_file_content[j*28*28], conv4_2_output[j], 28*28*sizeof(fp_t));
        }
        write_pgm(relu4_2_file_content, 512*28, 28, "relu4_2_output.pgm");
        write_float(relu4_2_file_content, 512*28, 28, "relu4_2_output.float");
        free(relu4_2_file_content);
        #endif


        // conv4_3 input 28x28x512 -> output 28x28x512
        fp_t** conv4_3_output;
        conv4_3_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv4_3_output[j] = (fp_t*) malloc(28*28*sizeof(fp_t));
        }

        fp_t* conv4_3_intermediate = (fp_t*) malloc(28*28*sizeof(fp_t));

        fp_t** conv4_3_kernels = kernels[9];
        fp_t* conv4_3_bias = biasses[9];


        for(j = 0; j < 512; j++) {
            convolution2d_naive(conv4_2_output[0], 28, 28, conv4_3_output[j], conv4_3_kernels[j*512], 3, 1, 1, 0.0);

            for(k = 1; k < 511; k++) {
                convolution2d_naive(conv4_2_output[k], 28, 28, conv4_3_intermediate, conv4_3_kernels[j*512+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv4_3_output[j], conv4_3_intermediate, 28, 28);
            }
            convolution2d_naive(conv4_2_output[511], 28, 28, conv4_3_intermediate, conv4_3_kernels[j*512+511], 3, 1, 1, conv4_3_bias[j]);
            add_image2d_naive(conv4_3_output[j], conv4_3_intermediate, 28, 28);
        }

        // make pgm of conv4_3 image
        #ifdef DEBUG
        fp_t* conv4_3_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv4_3_file_content[j*28*28], conv4_3_output[j], 28*28*sizeof(fp_t));
        }
        
        write_pgm(conv4_3_file_content, 512*28, 28, "conv4_3_output.pgm");
        write_float(conv4_3_file_content, 512*28, 28, "conv4_3_output.float");
        free(conv4_3_file_content);
        #endif

        // free conv4_3 intermediate
        free(conv4_3_intermediate);

        // free conv4_2 output
        for(j = 0; j < 512; j++) {
            free(conv4_2_output[j]);
        }
        free(conv4_2_output);


        // relu4_3
        for(j = 0; j < 512; j++) {
            relu_naive(conv4_3_output[j], 28, 28, conv4_3_output[j]);
        }

        // make pgm of relu4_3 output
        #ifdef DEBUG
        fp_t* relu4_3_file_content = (fp_t*) malloc(28*28*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&relu4_3_file_content[j*28*28], conv4_3_output[j], 28*28*sizeof(fp_t));
        }
        write_pgm(relu4_3_file_content, 512*28, 28, "relu4_3_output.pgm");
        write_float(relu4_3_file_content, 512*28, 28, "relu4_3_output.float");
        free(relu4_3_file_content);
        #endif

        
        // pool4 input 28x28x512 -> output 14x14x512
        fp_t** pool4_output;
        pool4_output = (fp_t**) malloc(512*sizeof(fp_t*));
        
        for(j = 0; j < 512; j++) {
            pool4_output[j] = (fp_t*) malloc(14*14*sizeof(fp_t));
        }

        for(j = 0; j < 512; j++) {
            max_pooling2d_naive(conv4_3_output[j], 28, 28, pool4_output[j], 2, 2);
        }

        // make pgm of pool4 output
        #ifdef DEBUG
        fp_t* pool4_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&pool4_file_content[j*14*14], pool4_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(pool4_file_content, 512*14, 14, "pool4_output.pgm");
        write_float(pool4_file_content, 512*14, 14, "pool4_output.float");
        free(pool4_file_content);
        #endif

        // free conv4_3 output
        for(j = 0; j < 512; j++) {
            free(conv4_3_output[j]);
        }
        free(conv4_3_output);


        // conv5_1 input 14x14x512 -> output 14x14x512
        fp_t** conv5_1_output;
        conv5_1_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv5_1_output[j] = (fp_t*) malloc(14*14*sizeof(fp_t));
        }

        fp_t* conv5_1_intermediate = (fp_t*) malloc(14*14*sizeof(fp_t));

        fp_t** conv5_1_kernels = kernels[10];
        fp_t* conv5_1_bias = biasses[10];

        for(j = 0; j < 512; j++) {
            convolution2d_naive(pool4_output[0], 14, 14, conv5_1_output[j], conv5_1_kernels[j*512], 3, 1, 1, 0.0);

            for(k = 1; k < 511; k++) {
                convolution2d_naive(pool4_output[k], 14, 14, conv5_1_intermediate, conv5_1_kernels[j*512+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv5_1_output[j], conv5_1_intermediate, 14, 14);
            }
            convolution2d_naive(pool4_output[511], 14, 14, conv5_1_intermediate, conv5_1_kernels[j*512+511], 3, 1, 1, conv5_1_bias[j]);
            add_image2d_naive(conv5_1_output[j], conv5_1_intermediate, 14, 14);
        }

        // make pgm of conv5_1 image
        #ifdef DEBUG
        fp_t* conv5_1_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv5_1_file_content[j*14*14], conv5_1_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(conv5_1_file_content, 512*14, 14, "conv5_1_output.pgm");
        write_float(conv5_1_file_content, 512*14, 14, "conv5_1_output.float");
        free(conv5_1_file_content);
        #endif

        // free conv5_1 intermediate
        free(conv5_1_intermediate);

        // free pool4 output
        for(j = 0; j < 512; j++) {
            free(pool4_output[j]);
        }
        free(pool4_output);


        // relu5_1
        for(j = 0; j < 512; j++) {
            relu_naive(conv5_1_output[j], 14, 14, conv5_1_output[j]);
        }

        // make pgm of relu5_1 output
        #ifdef DEBUG
        fp_t* relu5_1_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&relu5_1_file_content[j*14*14], conv5_1_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(relu5_1_file_content, 512*14, 14, "relu5_1_output.pgm");
        write_float(relu5_1_file_content, 512*14, 14, "relu5_1_output.float");
        free(relu5_1_file_content);
        #endif

       
        // conv5_2 input 14x14x512 -> output 14x14x512
        fp_t** conv5_2_output;
        conv5_2_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv5_2_output[j] = (fp_t*) malloc(14*14*sizeof(fp_t));
        }

        fp_t* conv5_2_intermediate = (fp_t*) malloc(14*14*sizeof(fp_t));

        fp_t** conv5_2_kernels = kernels[11];
        fp_t* conv5_2_bias = biasses[11];

        for(j = 0; j < 512; j++) {
            convolution2d_naive(conv5_1_output[0], 14, 14, conv5_2_output[j], conv5_2_kernels[j*512], 3, 1, 1, 0.0);

            for(k = 1; k < 511; k++) {
                convolution2d_naive(conv5_1_output[k], 14, 14, conv5_2_intermediate, conv5_2_kernels[j*512+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv5_2_output[j], conv5_2_intermediate, 14, 14);
            }
            convolution2d_naive(conv5_1_output[511], 14, 14, conv5_2_intermediate, conv5_2_kernels[j*512+511], 3, 1, 1, conv5_2_bias[j]);
            add_image2d_naive(conv5_2_output[j], conv5_2_intermediate, 14, 14);
        }

        // make pgm of conv5_2 image
        #ifdef DEBUG
        fp_t* conv5_2_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv5_2_file_content[j*14*14], conv5_2_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(conv5_2_file_content, 512*14, 14, "conv5_2_output.pgm");
        write_float(conv5_2_file_content, 512*14, 14, "conv5_2_output.float");
        free(conv5_2_file_content);
        #endif

        // free conv5_2 intermediate
        free(conv5_2_intermediate);

        // free conv5_1 output
        for(j = 0; j < 512; j++) {
            free(conv5_1_output[j]);
        }
        free(conv5_1_output);


        // relu5_2
        for(j = 0; j < 512; j++) {
            relu_naive(conv5_2_output[j], 14, 14, conv5_2_output[j]);
        }

        // make pgm of relu5_2 output
        #ifdef DEBUG
        fp_t* relu5_2_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&relu5_2_file_content[j*14*14], conv5_2_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(relu5_2_file_content, 512*14, 14, "relu5_2_output.pgm");
        write_float(relu5_2_file_content, 512*14, 14, "relu5_2_output.float");
        free(relu5_2_file_content);
        #endif


        // conv5_3 input 14x14x512 -> output 14x14x512
        fp_t** conv5_3_output;
        conv5_3_output = (fp_t**) malloc(512*sizeof(fp_t*));

        for(j = 0; j < 512; j++) {
            conv5_3_output[j] = (fp_t*) malloc(14*14*sizeof(fp_t));
        }

        fp_t* conv5_3_intermediate = (fp_t*) malloc(14*14*sizeof(fp_t));

        fp_t** conv5_3_kernels = kernels[12];
        fp_t* conv5_3_bias = biasses[12];

        for(j = 0; j < 512; j++) {
            convolution2d_naive(conv5_2_output[0], 14, 14, conv5_3_output[j], conv5_3_kernels[j*512], 3, 1, 1, 0.0);

            for(k = 1; k < 511; k++) {
                convolution2d_naive(conv5_2_output[k], 14, 14, conv5_3_intermediate, conv5_3_kernels[j*512+k], 3, 1, 1, 0.0);
                add_image2d_naive(conv5_3_output[j], conv5_3_intermediate, 14, 14);
            }
            convolution2d_naive(conv5_2_output[511], 14, 14, conv5_3_intermediate, conv5_3_kernels[j*512+511], 3, 1, 1, conv5_3_bias[j]);
            add_image2d_naive(conv5_3_output[j], conv5_3_intermediate, 14, 14);
        }

        // make pgm of conv5_3 image
        #ifdef DEBUG
        fp_t* conv5_3_file_content = (fp_t*) malloc(14*14*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&conv5_3_file_content[j*14*14], conv5_3_output[j], 14*14*sizeof(fp_t));
        }
        write_pgm(conv5_3_file_content, 512*14, 14, "conv5_3_output.pgm");
        write_float(conv5_3_file_content, 512*14, 14, "conv5_3_output.float");
        free(conv5_3_file_content);
        #endif

        // free conv5_3 intermediate
        free(conv5_3_intermediate);


        for(j = 0; j < 512; j++) {
            free(conv5_2_output[j]);
        }
        free(conv5_2_output);


        // pool5 input 14x14x512 -> output 7x7x512
        fp_t** pool5_output;
        pool5_output = (fp_t**) malloc(512*sizeof(fp_t*));
        
        for(j = 0; j < 512; j++) {
            pool5_output[j] = (fp_t*) malloc(7*7*sizeof(fp_t));
        }

        for(j = 0; j < 512; j++) {
            max_pooling2d_naive(conv5_3_output[j], 14, 14, pool5_output[j], 2, 2);
        }

        // make pgm of pool5 output
        #ifdef DEBUG
        fp_t* pool5_file_content = (fp_t*) malloc(7*7*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&pool5_file_content[j*7*7], pool5_output[j], 7*7*sizeof(fp_t));
        }
        write_pgm(pool5_file_content, 512*7, 7, "pool5_output.pgm");
        write_float(pool5_file_content, 512*7, 7, "pool5_output.float");
        free(pool5_file_content);
        #endif

        for(j = 0; j < 512; j++) {
            free(conv5_3_output[j]);
        }
        free(conv5_3_output);


        // fc6 7x7x512 = 1x25088 -> 1x4096
        // merge pool5 output
        fp_t* pool5_output_merged = (fp_t*) malloc(7*7*512*sizeof(fp_t));
        for(j = 0; j < 512; j++) {
            memcpy(&pool5_output_merged[j*7*7], pool5_output[j], 7*7*sizeof(fp_t));
        }

        // free pool5 output
        for(j = 0; j < 512; j++) {
            free(pool5_output[j]);
        }
        free(pool5_output);

        fp_t* fc6_output;
        fc6_output = (fp_t*) malloc(4096*sizeof(fp_t));

        fp_t* fc6_kernel = kernels[13][0];
        fp_t* fc6_bias = biasses[13];
        
        fully_connected_naive(pool5_output_merged, 25088, fc6_output, 4096, fc6_kernel, fc6_bias);

        // make pgm fc6 output
        #ifdef DEBUG
        write_pgm(fc6_output, 1, 4096, "fc6_output.pgm");
        write_float(fc6_output, 1, 4096, "fc6_output.float");
        #endif

        free(pool5_output_merged);

        // relu6
        relu_naive(fc6_output, 1, 4096, fc6_output);

        // make pgm of relu6 output
        #ifdef DEBUG
        write_pgm(fc6_output, 1, 4096, "relu6_output.pgm");
        write_float(fc6_output, 1, 4096, "relu6_output.float");
        #endif


        // drop6
        // do nothing
        

        // fc7 1x4096 -> 1x4096
        fp_t* fc7_output;
        fc7_output = (fp_t*) malloc(4096*sizeof(fp_t));

        fp_t* fc7_kernel = kernels[14][0];
        fp_t* fc7_bias = biasses[14];
        
        fully_connected_naive(fc6_output, 4096, fc7_output, 4096, fc7_kernel, fc7_bias);

        // make pgm fc7 output
        #ifdef DEBUG
        write_pgm(fc7_output, 1, 4096, "fc7_output.pgm");
        write_float(fc7_output, 1, 4096, "fc7_output.float");
        #endif

        // free fc6 output
        free(fc6_output);


        // relu7
        relu_naive(fc7_output, 1, 4096, fc7_output);

        // make pgm of relu7 output
        #ifdef DEBUG
        write_pgm(fc7_output, 1, 4096, "relu7_output.pgm");
        write_float(fc7_output, 1, 4096, "relu7_output.float");
        #endif
        
        
        // drop7
        // do nothing


        // fc8 1x4096 -> 1x1000
        fp_t* fc8_output;
        fc8_output = (fp_t*) malloc(1000*sizeof(fp_t));

        fp_t* fc8_kernel = kernels[15][0];
        fp_t* fc8_bias = biasses[15];
        
        fully_connected_naive(fc7_output, 4096, fc8_output, 1000, fc8_kernel, fc8_bias);

        // make pgm fc8 output
        #ifdef DEBUG
        write_pgm(fc8_output, 1, 1000, "fc8_output.pgm");
        write_float(fc8_output, 1, 1000, "fc8_output.float");
        #endif

        // free fc7 output
        free(fc7_output);


        // loss
        softmax_naive(fc8_output, 1, 1000, fc8_output);

        // make pgm prob
        #ifdef DEBUG
        write_pgm(fc8_output, 1, 1000, "loss_output.pgm");
        write_float(fc8_output, 1, 1000, "loss_output.float");
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
    for(i = 0; i < 1000; i++) {
        free(labels[i]);
    }
    free(labels);

    for(i = 0; i < 192; i++) {
        free(kernels[0][i]);
    }
    free(kernels[0]);

    for(i = 0; i < 4096; i++) {
        free(kernels[1][i]);
    }
    free(kernels[1]);

    for(i = 0; i < 8192; i++) {
        free(kernels[2][i]);
    }
    free(kernels[2]);

    for(i = 0; i < 16384; i++) {
        free(kernels[3][i]);
    }
    free(kernels[3]);

    for(i = 0; i < 32768; i++) {
        free(kernels[4][i]);
    }
    free(kernels[4]);

    for(i = 0; i < 65536; i++) {
        free(kernels[5][i]);
    }
    free(kernels[5]);

    for(i = 0; i < 65536; i++) {
        free(kernels[6][i]);
    }
    free(kernels[6]);

    for(i = 0; i < 131072; i++) {
        free(kernels[7][i]);
    }
    free(kernels[7]);

    for(i = 0; i < 262144; i++) {
        free(kernels[8][i]);
    }
    free(kernels[8]);

    for(i = 0; i < 262144; i++) {
        free(kernels[9][i]);
    }
    free(kernels[9]);

    for(i = 0; i < 262144; i++) {
        free(kernels[10][i]);
    }
    free(kernels[10]);

    for(i = 0; i < 262144; i++) {
        free(kernels[11][i]);
    }
    free(kernels[11]);

    for(i = 0; i < 262144; i++) {
        free(kernels[12][i]);
    }
    free(kernels[12]);

    free(kernels[13][0]);
    free(kernels[13]);

    free(kernels[14][0]);
    free(kernels[14]);

    free(kernels[15][0]);
    free(kernels[15]);

    free(kernels);


    for(i = 0; i < 16; i++) {
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



