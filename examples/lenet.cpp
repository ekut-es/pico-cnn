#define MNIST
#define NUM 1000
#define INDEX 0

#include <cstdlib>
#include <iomanip>

#include "pico-cnn/pico-cnn.h"
#include "network.h"

void usage() {
    printf("./lenet PATH_TO_BINARY_WEIGHTS_FILE PATH_TO_DATASET\n");
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
        PRINT_ERROR("No path to weights and dataset provided!")
        usage();
        return 1;
    }

    if(argc == 2) {
        PRINT_ERROR("No path to dataset provided!")
        usage();
        return 1;
    }

    char weights_path[1024];
    char mnist_path[1024];

    strcpy(weights_path, argv[1]);
    strcpy(mnist_path, argv[2]);

    uint32_t i, j;

    // read mnist t10k images
    char t10k_images_path[strlen(mnist_path) + 20];
    t10k_images_path[0] = '\0';
    strcat(t10k_images_path, mnist_path);
    strcat(t10k_images_path, "/t10k-images.idx3-ubyte");

    fp_t** t10k_images;
    uint32_t num_t10k_images;

    PRINT_INFO("Reading images from " << t10k_images_path)

    num_t10k_images = read_mnist_images(t10k_images_path, &t10k_images, NUM, 0, 0.0, 1.0);

    if(num_t10k_images < 1) {
        PRINT_ERROR("Could not read mnist images from " << t10k_images_path)
        return 1;
    }

    // read t10k labels
    char t10k_labels_path[strlen(mnist_path) + 20];
    t10k_labels_path[0] = '\0';
    strcat(t10k_labels_path, mnist_path);
    strcat(t10k_labels_path, "/t10k-labels.idx1-ubyte");

    uint8_t* t10k_labels;
    uint32_t num_t10k_labels;

    PRINT_INFO("Reading labels from " << t10k_labels_path)

    num_t10k_labels = read_mnist_labels(t10k_labels_path, &t10k_labels, NUM);

    if(num_t10k_images != num_t10k_labels) {
        PRINT_ERROR(num_t10k_images << " images != " << num_t10k_labels << " labels")
        return 1;
    }

    // make pgm of original image
    #ifdef DEBUG
    write_pgm(t10k_images[INDEX], 28, 28, "input.pgm");
    write_float(t10k_images[INDEX], 28, 28, "input.float");
    #endif

    Network *net = new Network();

    PRINT_INFO("Reading weights from " << weights_path)

    if(read_binary_weights(weights_path, &net->kernels, &net->biases) != 0){
        PRINT_ERROR("Could not read weights from " << weights_path)
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

    pico_cnn::naive::Tensor *input_tensor = new pico_cnn::naive::Tensor(1, 1, 28, 28);
    pico_cnn::naive::Tensor *output_tensor = new pico_cnn::naive::Tensor(1, 10);

    PRINT_INFO("Starting CNN");

    for(i = 0; i < NUM; i++) {

        std::memcpy(input_tensor->data_, t10k_images[i], 1*1*28*28*sizeof(fp_t));
        net->run(input_tensor, output_tensor);

        fp_t max = output_tensor->data_[0];
        uint32_t label = 0;
        for(uint32_t idx = 1; idx < 10; idx++) {
            if(output_tensor->data_[idx] > max) {
                max = output_tensor->data_[idx];
                label = idx;
            }
        }

        if(t10k_labels[i] == label) {
            correct_predictions++;
        }

        #ifdef DEBUG
        if(i == 0) {

            std::cout << std::setprecision(1);
            PRINT_DEBUG(*output_tensor)
            PRINT_DEBUG(*input_tensor)
            std::cout << std::setprecision(0);
            PRINT_DEBUG("Prediction: " << label);
        }
        #endif

        confusion_matrix[label][t10k_labels[i]]++;

    }

    delete net;

    for(i = 0; i < NUM; i++) {
        free(t10k_images[i]);
    }

    free(t10k_images);
    free(t10k_labels);

    delete input_tensor;
    delete output_tensor;

    // calculate and print results
    fp_t error_rate = 1.0-((fp_t) correct_predictions/((fp_t) NUM));

    #ifdef INFO

    std::cout << "Error rate: " << error_rate << " (" << correct_predictions << "/" << num_t10k_images << ")" << std::endl;

    std::cout << "Columns: actual label" << std::endl;
    std::cout << "Rows: predicted label" << std::endl;
    std::cout << "*\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9" << std::endl;

    for(i = 0; i < 10; i++) {
        std::cout << i << "\t";
        for(j = 0; j < 10; j++) {
            std::cout <<confusion_matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    #endif

    return 0;
}
