#include "test_tensor.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestTensor);

void TestTensor::setUp(){
    TestFixture::setUp();

    shape1 = new pico_cnn::naive::TensorShape(4, 5, 5);
    shape2 = new pico_cnn::naive::TensorShape(1, 3, 20, 20);
    shape3 = new pico_cnn::naive::TensorShape(1, 3, 20, 20);

    shape4 = new pico_cnn::naive::TensorShape();
    shape4->set_num_dimensions(6);
    for(int i = 0; i < 6; i++) {
        shape4->set_shape_idx(i, i+1);
    }

    shape5 = new pico_cnn::naive::TensorShape(3, 5);
    shape6 = new pico_cnn::naive::TensorShape(10);

    tensor1 = new pico_cnn::naive::Tensor(shape1);
    tensor2 = new pico_cnn::naive::Tensor(shape2);
    tensor3 = new pico_cnn::naive::Tensor(shape3);
    tensor4 = new pico_cnn::naive::Tensor(shape4);
    tensor5 = new pico_cnn::naive::Tensor(shape5);
    tensor6 = new pico_cnn::naive::Tensor(shape6);
}

void TestTensor::tearDown(){
    delete tensor1;
    delete tensor2;
    delete tensor3;
    delete tensor4;
    delete tensor5;
    delete tensor6;

    delete shape1;
    delete shape2;
    delete shape3;
    delete shape4;
    delete shape5;
    delete shape6;

    TestFixture::tearDown();
}

void TestTensor::runTestTensorShape() {
    //PRINT_INFO("Test TensorShapes...")

    CPPUNIT_ASSERT(*tensor1->shape() == *shape1);
    CPPUNIT_ASSERT(*tensor2->shape() == *tensor3->shape());
    CPPUNIT_ASSERT(*tensor2->shape() == *shape3);

    CPPUNIT_ASSERT(*tensor1->shape() == *shape1);
}

void TestTensor::runTestTensorAccess() {
    //PRINT_INFO("Test Tensor access...")

    // tensor1
    fp_t cnt = 0.0;
    for(int channel = 0; channel < 4; channel++) {
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                tensor1->access(channel, row, col) = cnt;
                cnt+=1.0;
            }
        }
    }
    cnt = 0.0;
    for(int channel = 0; channel < 4; channel++) {
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                CPPUNIT_ASSERT(abs(tensor1->access(channel, row, col) - cnt) < 0.0001);
                cnt+=1.0;
            }
        }
    }

    fp_t tmp = tensor1->access(0,0,0);
    tmp += 10.0;
    CPPUNIT_ASSERT(abs(tmp - tensor1->access(0,0,0)) == 10);

    // tensor2
    cnt = 0.0;
    for(int batch = 0; batch < 1; batch++) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = 0; row < 20; row++) {
                for (int col = 0; col < 20; col++) {
                    tensor2->access(batch, channel, row, col) = cnt;
                    cnt += 1.0;
                }
            }
        }
    }
    cnt = 0.0;
    for(int batch = 0; batch < 1; batch++) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = 0; row < 20; ++row) {
                for (int col = 0; col < 20; ++col) {
                    CPPUNIT_ASSERT(abs(tensor2->access(batch, channel, row, col) - cnt) < 0.0001);
                    cnt += 1.0;
                }
            }
        }
    }

    // tensor2 and tensor3
    cnt = 0.0;
    for(int batch = 0; batch < 1; batch++) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = 0; row < 20; row++) {
                for (int col = 0; col < 20; col++) {
                    tensor3->access(batch, channel, row, col) = cnt;
                    cnt += 1.0;
                }
            }
        }
    }
    CPPUNIT_ASSERT(*tensor2 == *tensor3);

    // tensor4
    cnt = 0.0;
    for(int x1 = 0; x1 < 1; x1++) {
        for (int x2 = 0; x2 < 2; x2++) {
            for (int x3 = 0; x3 < 3; ++x3) {
                for (int x4 = 0; x4 < 4; ++x4) {
                    for (int x5 = 0; x5 < 5; ++x5) {
                        for (int x6 = 0; x6 < 6; ++x6) {
                            tensor4->access(x1, x2, x3, x4, x5, x6) = cnt;
                            cnt += 1.0;
                        }
                    }
                }
            }
        }
    }
    cnt = 0.0;
    for(int x1 = 0; x1 < 1; x1++) {
        for (int x2 = 0; x2 < 2; x2++) {
            for (int x3 = 0; x3 < 3; ++x3) {
                for (int x4 = 0; x4 < 4; ++x4) {
                    for (int x5 = 0; x5 < 5; ++x5) {
                        for (int x6 = 0; x6 < 6; ++x6) {
                            CPPUNIT_ASSERT(abs(tensor4->access(x1, x2, x3, x4, x5, x6) - cnt) < 0.0001);
                            cnt += 1.0;
                        }
                    }
                }
            }
        }
    }

    // tensor5
    cnt = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 5; ++col) {
            tensor5->access(row, col) = cnt;
            cnt += 1.0;
        }
    }
    cnt = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 5; ++col) {
            CPPUNIT_ASSERT(abs(tensor5->access(row, col) - cnt) < 0.0001);
            cnt += 1.0;
        }
    }

    // tensor6
    cnt = 0.0;
    for (int col = 0; col < 10; ++col) {
        tensor6->access(col) = cnt;
        cnt += 1.0;
    }
    cnt = 0.0;
    for (int col = 0; col < 10; ++col) {
        CPPUNIT_ASSERT(abs(tensor6->access(col) - cnt) < 0.0001);
        cnt += 1.0;
    }
}

void TestTensor::runTestTensorAddition() {
    //PRINT_INFO("Test Tensor addition...")

    auto *input_shape1 = new pico_cnn::naive::TensorShape(3, 4);
    auto *input_shape2 = new pico_cnn::naive::TensorShape(3, 4);
    auto *expected_output_shape = new pico_cnn::naive::TensorShape(3, 4);
    auto *input_tensor1 = new pico_cnn::naive::Tensor(input_shape1);
    auto *input_tensor2 = new pico_cnn::naive::Tensor(input_shape2);
    auto *expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);

    fp_t input1[12] = {-6,-4, 3, -5,
                        3, 4, 2,  8,
                       -5, 6, 1, -4};
    fp_t input2[12] = {1, -3, -1,  5,
                      -7,  7,  6, -5,
                       7, -5,  4, -6};
    fp_t expected_output[12] = {-5, -7, 2,  0,
                                -4, 11, 8,  3,
                                 2,  1, 5,-10};

    for (uint32_t i = 0; i < input_tensor1->num_elements(); i++) {
        input_tensor1->access_blob(i) = input1[i];
        input_tensor2->access_blob(i) = input2[i];
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    input_tensor1->add_tensor(input_tensor2);

    CPPUNIT_ASSERT(*input_tensor1 == *expected_output_tensor);

    delete input_tensor1;
    delete input_tensor2;
    delete expected_output_tensor;
    delete input_shape1;
    delete input_shape2;
    delete expected_output_shape;

}

void TestTensor::runTestTensorGetPtr() {
    for(int batch = 0; batch < 1; batch++) {
        for (int channel = 0; channel < 3; channel++) {
            fp_t tmp1, tmp2;
            fp_t *tmp_ptr = tensor2->get_ptr_to_channel(batch, channel);
            for (int row = 0; row < 20; row++) {
                for (int col = 0; col < 20; col++) {
                    tmp1 = tensor2->access(batch, channel, row, col);
                    tmp2 = tmp_ptr[row*20+col];
                    CPPUNIT_ASSERT(tmp1 == tmp2);
                }
            }
        }
    }
}

void TestTensor::runTestTensorExpandPadding() {

    /// 4D test
    uint32_t padding[4] = {1, 1, 1, 1};

    auto *orig_shape = new pico_cnn::naive::TensorShape(1, 3, 2, 2);
    auto *expected_extended_shape = new pico_cnn::naive::TensorShape(1, 3, 4, 4);
    auto *extended_shape = orig_shape->expand_with_padding(padding);

    CPPUNIT_ASSERT(*extended_shape == *expected_extended_shape);

    auto *orig_tensor = new pico_cnn::naive::Tensor(orig_shape);
    auto *expected_extended_tensor = new pico_cnn::naive::Tensor(expected_extended_shape);

    fp_t input[12] = {1, 2,
                      3, 4,

                      5, 6,
                      7, 8,

                      9, 10,
                      11, 12};

    fp_t expected[48]= {0, 0, 0, 0,
                        0, 1, 2, 0,
                        0, 3, 4, 0,
                        0, 0, 0, 0,

                        0, 0, 0, 0,
                        0, 5, 6, 0,
                        0, 7, 8, 0,
                        0, 0, 0, 0,

                        0, 0, 0, 0,
                        0, 9, 10, 0,
                        0, 11, 12, 0,
                        0, 0, 0, 0};

    for (uint32_t i = 0; i < orig_tensor->num_elements(); i++) {
        orig_tensor->access_blob(i) = input[i];
    }
    for (uint32_t i = 0; i < expected_extended_tensor->num_elements(); i++) {
        expected_extended_tensor->access_blob(i) = expected[i];
    }

    auto *extended_tensor = orig_tensor->expand_with_padding(padding);

    CPPUNIT_ASSERT(*extended_tensor == *expected_extended_tensor);

    delete orig_tensor;
    delete expected_extended_tensor;

    delete extended_tensor->shape();
    delete extended_tensor;

    delete extended_shape;
    delete expected_extended_shape;
    delete orig_shape;

    /// 3D (1D data) test
    uint32_t padding_1d[4] = {2, 2};

    auto *orig_shape_1d = new pico_cnn::naive::TensorShape(1, 3,4);
    auto *expected_extended_shape_1d = new pico_cnn::naive::TensorShape(1,3,8);
    auto *extended_shape_1d = orig_shape_1d->expand_with_padding(padding_1d);

    CPPUNIT_ASSERT(*extended_shape_1d == *expected_extended_shape_1d);

    auto *orig_tensor_1d = new pico_cnn::naive::Tensor(orig_shape_1d);
    auto *expected_extended_tensor_1d = new pico_cnn::naive::Tensor(expected_extended_shape_1d);

    fp_t input_1d[12] = {1, 2, 3, 4,

                        5, 6, 7, 8,

                        9, 10, 11, 12};

    fp_t expected_1d[24]= {0, 0, 1, 2, 3, 4, 0, 0,

                           0, 0, 5, 6, 7, 8, 0, 0,

                           0, 0, 9, 10, 11, 12,  0, 0};

    for (uint32_t i = 0; i < orig_tensor_1d->num_elements(); i++) {
        orig_tensor_1d->access_blob(i) = input_1d[i];
    }
    for (uint32_t i = 0; i < expected_extended_tensor_1d->num_elements(); i++) {
        expected_extended_tensor_1d->access_blob(i) = expected_1d[i];
    }

    auto *extended_tensor_1d = orig_tensor_1d->expand_with_padding(padding_1d);

    CPPUNIT_ASSERT(*extended_tensor_1d == *expected_extended_tensor_1d);

    delete orig_tensor_1d;
    delete expected_extended_tensor_1d;

    delete extended_tensor_1d->shape();
    delete extended_tensor_1d;

    delete extended_shape_1d;
    delete expected_extended_shape_1d;
    delete orig_shape_1d;

    /// 2D test
    uint32_t padding_2d[4] = {1, 2, 4, 0};

    fp_t input_2d[35] = {4, 13,  13,  -2,   6,
                         -7, 15, -11,  -9, -15,
                         -8, -3,   2,   6,  -9,
                         1, -8,  13,  -6,  -7,
                         -6, -6,  13,  13,   8,
                         -15, 14, -12,   8,   1,
                         3,  6,   7, -12,   2};

    fp_t expected_2d[84]= {0, 0,  0,  0,   0,    0,   0,
                           0, 0,  4,  13,  13,  -2,   6,
                           0, 0,  -7, 15, -11,  -9, -15,
                           0, 0,  -8, -3,   2,   6,  -9,
                           0, 0,   1, -8,  13,  -6,  -7,
                           0, 0,  -6, -6,  13,  13,   8,
                           0, 0, -15, 14, -12,   8,   1,
                           0, 0,  3,  6,   7, -12,   2,
                           0, 0,  0,  0,   0,   0,   0,
                           0, 0,  0,  0,   0,   0,   0,
                           0, 0,  0,  0,   0,   0,   0,
                           0, 0,  0,  0,   0,   0,   0};

    auto *orig_shape_2d = new pico_cnn::naive::TensorShape(7, 5);
    auto *expected_extended_shape_2d = new pico_cnn::naive::TensorShape(12, 7);
    auto *extended_shape_2d = orig_shape_2d->expand_with_padding(padding_2d);

    CPPUNIT_ASSERT(*extended_shape_2d == *expected_extended_shape_2d);

    auto *orig_tensor_2d = new pico_cnn::naive::Tensor(orig_shape_2d);
    auto *expected_extended_tensor_2d = new pico_cnn::naive::Tensor(expected_extended_shape_2d);

    for (uint32_t i = 0; i < orig_tensor_2d->num_elements(); i++) {
        orig_tensor_2d->access_blob(i) = input_2d[i];
    }
    for (uint32_t i = 0; i < expected_extended_tensor_2d->num_elements(); i++) {
        expected_extended_tensor_2d->access_blob(i) = expected_2d[i];
    }

    auto *extended_tensor_2d = orig_tensor_2d->expand_with_padding(padding_2d);

    CPPUNIT_ASSERT(*extended_tensor_2d == *expected_extended_tensor_2d);

    delete orig_tensor_2d;
    delete expected_extended_tensor_2d;
    delete extended_tensor_2d->shape();
    delete extended_tensor_2d;

    delete extended_shape_2d;
    delete expected_extended_shape_2d;
    delete orig_shape_2d;
}

void TestTensor::runTestTensorConcatDim0() {
    auto *input_shape1 = new pico_cnn::naive::TensorShape(1, 2, 3, 3);
    auto *input_shape2 = new pico_cnn::naive::TensorShape(1, 3, 3, 3);
    auto *expected_output_shape = new pico_cnn::naive::TensorShape(1, 5, 3, 3);
    auto *output_shape = new pico_cnn::naive::TensorShape(1, 5, 3, 3);
    auto *input_tensor1 = new pico_cnn::naive::Tensor(input_shape1);
    auto *input_tensor2 = new pico_cnn::naive::Tensor(input_shape2);
    auto *expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto *output_tensor = new pico_cnn::naive::Tensor(output_shape);


    fp_t input_1[18] = {1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,

                        10, 11, 12,
                        13, 14, 15,
                        16, 17, 18};

    fp_t input_2[27] = {19, 20, 21,
                        22, 23, 24,
                        25, 26, 27,

                        28, 29, 30,
                        31, 32, 33,
                        34, 35, 36,

                        37, 38, 39,
                        40, 41, 42,
                        43, 44, 45};

    fp_t expected_output1[45] = {1, 2, 3,
                                 4, 5, 6,
                                 7, 8, 9,

                                 10, 11, 12,
                                 13, 14, 15,
                                 16, 17, 18,

                                 19, 20, 21,
                                 22, 23, 24,
                                 25, 26, 27,

                                 28, 29, 30,
                                 31, 32, 33,
                                 34, 35, 36,

                                 37, 38, 39,
                                 40, 41, 42,
                                 43, 44, 45};

    for (uint32_t i = 0; i < input_tensor1->num_elements(); i++) {
        input_tensor1->access_blob(i) = input_1[i];
    }
    for (uint32_t i = 0; i < input_tensor2->num_elements(); i++) {
        input_tensor2->access_blob(i) = input_2[i];
    }
    for (uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output1[i];
    }

    pico_cnn::naive::Tensor* inputs[2] = {input_tensor1, input_tensor2};

    output_tensor->concatenate_from(2, inputs, 1);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete input_tensor1;
    delete input_tensor2;
    delete expected_output_tensor;
    delete output_tensor;

    delete input_shape1;
    delete input_shape2;
    delete expected_output_shape;
    delete output_shape;
}
