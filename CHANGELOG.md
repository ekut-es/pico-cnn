# Changelog
## Version 2.0

 tag `v2.0`
 
 * Pico-CNN is now implemented in C++
   * The set of implemented operations and test coverage is equivalent to `v1.0.2`.
   * Data and its shape is now encapsulated in the `pico_cnn::naive::Tensor` class making the interface of all operations more similar and less cluttered. Moreover operations like Reshape or Flatten are now working more straightforward.
   * Right now the `pico_cnn::naive::Tensor` class supports up to four shape dimensions.

## Version 1.0.2

 tag `v1.0.2`

 * Bugfixes
   * Fixed bug in AveragePooling: When using asymmetric padding with `count_include_pad == 0` the center/edge case computation assumed symmetric padding.
   * Fixed typo in activation function test.
   * Fixed bug using 'height' twice when generating random 4D input reference data.
 * Additional tests
   * Added three more tests for the convolution layer covering multiple output channels.
 * Miscellaneous
   * Added optimization flags to the library's and the generated Makefile.
   * Added seeding option for the generation of reference data when using random input.
   * Removed targets from CMakeLists.txt which have to be generated first (an example target for LeNet can be found commented). CMake should only be used for debugging. For all other uses please use the (generated) Makefiles.

## Version 1.0.1

 tag `v1.0.1`

 * Fixed typo in GlobalMaxPooling 

## Version 1.0

 tag `v1.0`

 * Restructured C code (split in .c and .h)
 * Import of ONNX models and generation of C code
 * Tested neural networks:
    * LeNet
    * MNIST Multi-Layer-Perceptron (MLP)
    * MNIST Perceptron
    * AlexNet
    * VGG-16
    * VGG-19
    * MobileNet-V2
    * Inception-V3
    * Inception-Resnet-V2
    * TC-ResNet-8
 * Removed experimental hand-crafted ARM-NEON Code
 * Removed experimental OpenMP support

## Version 0.1 

 tag `v0.1`

 * Tested neural networks:
    * LeNet-5
    * AlexNet
    * VGG-16
 * Experimental support of ARM-NEON SIMD instructions for several layers
 * Experimental support of OpenMP on layer level
