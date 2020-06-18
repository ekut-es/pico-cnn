#include "test_batch_normalization.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestBatchNormalization);

void TestBatchNormalization::setUp() {
    TestFixture::setUp();
}

void TestBatchNormalization::tearDown() {
    TestFixture::tearDown();
}

void TestBatchNormalization::runTestBatchNormalization_1() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 3);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 3);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 3);

    auto gamma_tensor = new pico_cnn::naive::Tensor(1);
    auto beta_tensor = new pico_cnn::naive::Tensor(1);
    auto mean_tensor = new pico_cnn::naive::Tensor(1);
    auto variance_tensor = new pico_cnn::naive::Tensor(1);

    fp_t gammas[1] = {1.0};
    fp_t beta[1] = {0};
    fp_t mean[1] = {0};
    fp_t variance[1] = {1.0};
    fp_t epsilon = 1e-5;

    fp_t input[1*3] = {-1, 0, 1};
    fp_t expected_output[1*3] = {-0.999995, 0.0, 0.999995};

    for (uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    for (uint32_t i = 0; i < gamma_tensor->num_elements(); i++) {
        gamma_tensor->access_blob(i) = gammas[i];
        beta_tensor->access_blob(i) = beta[i];
        mean_tensor->access_blob(i) = mean[i];
        variance_tensor->access_blob(i) = variance[i];
    }

    auto layer = new pico_cnn::naive::BatchNormalization("bn", 0, pico_cnn::op_type::BatchNormalization,
                                                         gamma_tensor, beta_tensor, mean_tensor, variance_tensor, epsilon);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestBatchNormalization::runTestBatchNormalization_2() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 3);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 3);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 3);

    auto gamma_tensor = new pico_cnn::naive::Tensor(1);
    auto beta_tensor = new pico_cnn::naive::Tensor(1);
    auto mean_tensor = new pico_cnn::naive::Tensor(1);
    auto variance_tensor = new pico_cnn::naive::Tensor(1);

    fp_t gammas[1] = {1.5};
    fp_t beta[1] = {1.1};
    fp_t mean[1] = {3};
    fp_t variance[1] = {0.9};
    fp_t epsilon = 1e-5;

    fp_t input[3*3] = {1, 2, 3,
                       4, 5, 6,
                       7, 8, 9};
    fp_t expected_output[3*3] = {-2.0622602, -0.48113, 1.1,
                                 2.68113, 4.26226, 5.84339,
                                 7.42452, 9.0056505, 10.586781};

    for (uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    for (uint32_t i = 0; i < gamma_tensor->num_elements(); i++) {
        gamma_tensor->access_blob(i) = gammas[i];
        beta_tensor->access_blob(i) = beta[i];
        mean_tensor->access_blob(i) = mean[i];
        variance_tensor->access_blob(i) = variance[i];
    }

    auto layer = new pico_cnn::naive::BatchNormalization("bn", 0, pico_cnn::op_type::BatchNormalization,
                                                         gamma_tensor, beta_tensor, mean_tensor, variance_tensor, epsilon);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
    delete expected_output_tensor;
}
