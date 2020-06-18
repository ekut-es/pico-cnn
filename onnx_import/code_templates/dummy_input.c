#define LOWER_BOUND 0.0
#define UPPER_BOUND 1.0

#include <cstdlib>
#include <ctime>

#include "pico-cnn-cpp/pico-cnn.h"
#include "network.h"

void usage() {
    printf("./dummy_input PATH_TO_BINARY_WEIGHTS_FILE RUNS GENERATE_ONCE\n");
}

static inline fp_t urand(fp_t min, fp_t max) {
    return (fp_t) ((((fp_t)std::rand()/(fp_t)(RAND_MAX)) * 1.0f) * (max - min) + min);
}

int32_t main(int32_t argc, char** argv) {

    if(argc != 4) {
        usage();
        return 1;
    }

    std::srand(std::time(0));

    char weights_path[1024];
    strcpy(weights_path, argv[1]);

    int32_t RUNS = atoi(argv[2]);

    int32_t GENERATE_ONCE = atoi(argv[3]);

    {% if num_input_dims == 4 %}
    auto input_tensor = new pico_cnn::naive::Tensor({{num_input_batches}}, {{num_input_channels}}, {{input_channel_height}}, {{input_channel_width}});
    {% elif num_input_dims == 3 %}
    auto input_tensor = new pico_cnn::naive::Tensor({{num_input_batches}}, {{num_input_channels}}, {{input_channel_width}});
    {% elif num_input_dims == 2 %}
    auto input_tensor = new pico_cnn::naive::Tensor({{input_channel_height}}, {{input_channel_width}});
    {% endif %}

    {% if num_output_dims == 4 %}
    auto output_tensor = new pico_cnn::naive::Tensor({{num_output_batches}}, {{num_output_channels}}, {{output_channel_height}}, {{output_channel_width}});
    {% elif num_output_dims == 3 %}
    PRINT_ERROR_AND_DIE("3D output not supported.")
    {% elif num_output_dims == 2 %}
    auto output_tensor = new pico_cnn::naive::Tensor({{num_output_batches}}, {{num_output_channels}});
    {% endif %}


    if(GENERATE_ONCE) {
        PRINT_INFO("Random input will be generated once for all inference runs.")

        srand(time(NULL));

        for(uint32_t element = 0; element < input_tensor->num_elements(); element++) {
            input_tensor->access_blob(element) = urand(LOWER_BOUND, UPPER_BOUND);
        }

    } else {
        PRINT_INFO("Random input will be generated with new seed for each inference run.")
    }

    Network *net = new Network();

    PRINT_INFO("Reading weights from " << weights_path)

    if(read_binary_weights(weights_path, &net->kernels, &net->biases) != 0){
        PRINT_ERROR("could not read weights from " << weights_path)
        return 1;
    }

    PRINT_INFO("Starting CNN for " << RUNS << " runs...")

    for(uint32_t run = 0; run < RUNS; run++) {

        PRINT_DEBUG("Run " << run+1 << " of " << RUNS)

        if(!GENERATE_ONCE) {
            srand(time(NULL));

            for(uint32_t element = 0; element < input_tensor->num_elements(); element++) {
                input_tensor->access_blob(element) = urand(LOWER_BOUND, UPPER_BOUND);
            }
        }
        net->run(input_tensor, output_tensor);
    }

    PRINT_INFO("After CNN")

    delete net;

    delete input_tensor;
    delete output_tensor;

    return 0;

}