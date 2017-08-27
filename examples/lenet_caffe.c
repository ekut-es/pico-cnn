/** 
 * @brief LeNet implementation as provided by the Caffe LeNet MNIST example:
 * http://caffe.berkeleyvision.org/gathered/examples/mnist.html
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include "pico-cnn/pico-cnn.h"

void usage() {
    printf("./lenet_caffe PATH_TO_MNIST_DATASET PATH_TO_WEIGHTS_FILE\n");
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

    return 0;
}
