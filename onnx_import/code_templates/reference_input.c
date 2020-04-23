#define ALMOST 0.001

#include <cstdlib>

#include "pico-cnn-cpp/pico-cnn.h"
#include "network.h"

void usage() {
    printf("./reference_input PATH_TO_BINARY_WEIGHTS_FILE PATH_TO_REFERENCE_INPUT PATH_TO_REFERENCE_OUTPUT\n");
}

int32_t almost_equal(float a, float b, float epsilon){
    return (fabs(a-b) <= epsilon);
}

int32_t main(int32_t argc, char** argv) {

    if(argc != 4) {
        usage();
        return 1;
    }

    char weights_path[1024];
    strcpy(weights_path, argv[1]);

    char sample_input_path[1024];
    strcpy(sample_input_path, argv[2]);

    char sample_output_path[1024];
    strcpy(sample_output_path, argv[3]);

    {% if num_input_dims == 4 %}
    auto input_shape = new pico_cnn::naive::TensorShape({{num_input_batches}}, {{num_input_channels}}, {{input_channel_height}}, {{input_channel_width}});
    {% elif num_input_dims == 3 %}
    auto input_shape = new pico_cnn::naive::TensorShape({{num_input_batches}}, {{num_input_channels}}, {{input_channel_width}});
    {% elif num_input_dims == 2 %}
    auto input_shape = new pico_cnn::naive::TensorShape({{input_channel_height}}, {{input_channel_width}});
    {% endif %}
    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);

    {% if num_output_dims == 4 %}
    auto output_shape = new pico_cnn::naive::TensorShape({{num_output_batches}}, {{num_output_channels}}, {{output_channel_height}}, {{output_channel_width}});
    auto ref_output_shape = new pico_cnn::naive::TensorShape({{num_output_batches}}, {{num_output_channels}}, {{output_channel_height}}, {{output_channel_width}});
    {% elif num_output_dims == 3 %}
    //auto output_shape = new pico_cnn::naive::TensorShape({{num_output_batches}}, {{num_output_channels}}, {{output_channel_height}}, {{output_channel_width}});
    {% elif num_output_dims == 2 %}
    auto output_shape = new pico_cnn::naive::TensorShape({{num_output_batches}}, {{num_output_channels}});
    auto ref_output_shape = new pico_cnn::naive::TensorShape({{num_output_batches}}, {{num_output_channels}});
    {% endif %}
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto ref_output_tensor = new pico_cnn::naive::Tensor(output_shape);


    if(read_binary_reference_input_data(sample_input_path, input_tensor) != 0)
        return -1;

    if(read_binary_reference_output_data(sample_output_path, ref_output_tensor) != 0)
        return -1;

    Network *net = new Network();

    PRINT_INFO("Reading weights from: " << weights_path)

    if(read_binary_weights(weights_path, &net->kernels, &net->biases) != 0){
        PRINT_ERROR("Could not read weights from: " << weights_path)
        return 1;
    }

    PRINT_INFO("Starting CNN...")

    net->run(input_tensor, output_tensor);

    PRINT_INFO("After CNN")

    delete net;

    int32_t all_equal = 1;

    {% if num_output_dims == 4 or num_output_dims == 3 %}
    for(uint32_t batch = 0; batch < output_tensor->num_batches(); batch++) {
        for(uint32_t channel = 0; channel < output_tensor->num_channels(); channel++) {
            for(uint32_t row = 0; row < output_tensor->height(); row++) {
                for(uint32_t col = 0; col < output_tensor->width(); col++) {

                    PRINT_DEBUG("Batch: " << batch << "\tchannel: " << channel << "\trow: " << row << "\tcol: " << col << "\toutput: " << output_tensor->access(batch, channel, row, col) << "\tref_output: " << ref_output_tensor->access(batch, channel, row, col))

                    if(!almost_equal(output_tensor->access(batch, channel, row, col), ref_output_tensor->access(batch, channel, row, col), ALMOST)) {
                        all_equal = 0;
                        PRINT_ERROR("Not equal at batch: " << batch << "\tchannel: " << channel << "\trow: " << row << "\tcol: " << col << "\toutput " << output_tensor->access(batch, channel, row, col) << "\tref_output: " << ref_output_tensor->access(batch, channel, row, col));
                    }
                }
            }
        }
    }
    {% elif num_output_dims == 2 %}
    for(uint32_t row = 0; row < output_tensor->height(); row++) {
        for(uint32_t col = 0; col < output_tensor->width(); col++) {

            PRINT_DEBUG("Row: " << row << "\tcol: " << col << "\toutput: " << output_tensor->access(row, col) << "\tref_output: " << ref_output_tensor->access(row, col))

            if(!almost_equal(output_tensor->access(row, col), ref_output_tensor->access(row, col), ALMOST)) {
                all_equal = 0;
                PRINT_ERROR("Not equal at row: " << row << " col: " << col << ", output: " << output_tensor->access(row, col) << ", ref_output: " << ref_output_tensor->access(row, col))
            }
        }
    }
    {% endif %}

    if(all_equal) {
        PRINT_INFO("Output is almost equal to reference output! epsilon=" << ALMOST);
    } else {
        PRINT_ERROR("WARNING: Output is not almost equal to reference output! epsilon=" << ALMOST);
    }

    delete input_tensor;
    delete input_shape;

    delete output_tensor;
    delete output_shape;

    delete ref_output_tensor;
    delete ref_output_shape;

    return 0;

}