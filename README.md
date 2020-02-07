# Pico-CNN

## TL;DR
Please read the whole document carefully!

### Ubuntu
```bash
sudo apt install libjpeg-dev
```

### Scientific Linux
```bash
sudo yum install libjpeg-devel
```

### C-Standard
Depending on the (system-) compiler you are using you might have to add a specific C-Standard to the `CFLAGS` variable in the generated Makefile. Assuming a modern operating system like Ubuntu 18.04 the default C-Standard is `-std=gnu11`. If you are using an older version of GCC it should suffice to chose `-std=c99` or `-std=gnu99` as the C-Standard.

### All distributions
Install `python3.6` for example with [pyenv](https://github.com/pyenv/pyenv) (probably also works with other Python versions). Then install the required Python packages with the requirements.txt file located in `pico-cnn/onnx_import`:
```bash
pip install -r requirements.txt
```
## Supported Neural Networks
The pico-cnn framework currently supports the following neural networks:
* LeNet
* AlexNet
* VGG-16
* VGG-19
* MobileNet-V2
* Inception-V3
* EKUT-Raw
 * CNN-3-Relu
 * CNN-3-Relu-2
 * CNN-6-Relu
 * CNN-6-Relu-Simple
* MNIST Multi-Layer-Perceptron (MLP)
* MNIST Perceptron


### All networks
#### Dummy Input
For every imported onnx model a `dummy_input.c` will be generated, which uses random numbers as input and calls the network, so that no network specific input data has to be downloaded to run inferences.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/model.onnx
cd generated_code/model
make dummy_input
./dummy_input network.weights.bin NUM_RUNS GENERATE_ONCE
```
`GENERATE_ONCE = 0` will lead to new random input for each inference run.  
`GENERATE_ONCE = 1` will lead to the same random input for each inference run.

If you might want to monitor overall progress by uncommenting `#define PRINT` in `dummy_input.c`.

#### Reference Input
There will also be generated a `reference_input.c` which can be used to validate the imported network against the reference input/output that is provided for onnx models from the official [onnx model-zoo](https://github.com/onnx/models). The data has to be preprocessed:
```bash
python util/parse_onnx_sample_files.py --input PATH_TO_SAMPLE_DATA
```
The script will generate an `input_X.data` and `output_X.data` file which can then be used like this:
```bash
cd onnx_import/generated_code/model
make reference_input
./reference_input network.weights.bin PATH_TO_SAMPLE_DATA/input_X.data PATH_TO_SAMPLE_DATA/output_X.data
```

If the model was acquired in some other way (self-trained or converted) you can create sample data with the following script:
```bash
cd util
python3.6 create_sample_data.py --model model.onnx --file PATH_TO_INPUT_DATA.[jpeg/wav] --shape 1 NUM_CHANNELS HEIGHT WIDTH
```

## MNIST Dataset
### LeNet-5
LeNet-5 implementation as proposed by Yann LeCun et. al <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>

**Note: MNIST dataset required to run the LeNet specific code.**
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/lenet.onnx
```
Copy `examples/lenet.c` from examples folder to `onnx_import/generated_code/lenet`.
```bash
cd generated_code/lenet
make lenet
./lenet PATH_TO_MNIST network.weights.bin
```

### MNIST Multi-Layer-Perceptron (MLP)
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mnist_mlp.onnx
```
Copy `examples/mnist_mlp.c` to `onnx_import/generated_code/mnist_mlp`.
```bash
cd generated_code/mnist_mlp
make mnist_mlp
./mnist_mlp PATH_TO_MNIST network.weights.bin
```

### MNIST Perceptron
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mnist_simple_perceptron.onnx
```
Copy `examples/mnist_simple_perceptron.c` to `onnx_import/generated_code/mnist_simple_perceptron`.
```bash
cd generated_code/mnist_simple_perceptron
make mnist_simple_perceptron
./mnist_simple_perceptron PATH_TO_MNIST network.weights.bin
```

## ImageNet Dataset
### AlexNet
AlexNet implementation as proposed by Alex Krizhevsky et. al <a id="cit_Krizhevsky2017">[[Krizhevsky2017]](#Krizhevsky2017)</a>

**Note: ImageNet dataset required to run the AlexNet specific code.**
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/alexnet.onnx
```
Copy `examples/alexnet.c` to `onnx_import/generated_code/alexnet`.
```bash
cd generated_code/alexnet
```
Add `-ljpeg` to `LDFLAGS` in `Makefile`
```bash
make alexnet
./alexnet network.weights.bin PATH_TO_IMAGE_MEANS PATH_TO_LABELS PATH_TO_IMAGE
```

### VGG-16
VGG-16 model retrieved from https://github.com/onnx/models/tree/master/vision/classification/vgg

**Note: ImageNet dataset required to run the VGG-16 specific code.**
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/vgg16.onnx
```
Copy `examples/vgg.c` to `onnx_import/generated_code/vgg16.c`.
```bash
cd generated_code/vgg16
```
Add `-ljpeg` to `LDFLAGS` in `Makefile`
```bash
make vgg16
./vgg16 network.weights.bin PATH_TO_IMAGE_MEANS PATH_TO_LABELS PATH_TO_IMAGE
```

### VGG-19
VGG-19 model retrieved from https://github.com/onnx/models/tree/master/vision/classification/vgg

**Note: ImageNet dataset required to run the VGG-19 specific code.**
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/vgg19.onnx
```
Copy `examples/vgg.c` to `onnx_import/generated_code/vgg19.c`.
```bash
cd generated_code/vgg19
```
Add `-ljpeg` to `LDFLAGS` in `Makefile`
```bash
make vgg19
./vgg19 network.weights.bin PATH_TO_IMAGE_MEANS PATH_TO_LABELS PATH_TO_IMAGE
```

### MobileNet-V2
MobileNet-V2 model retrieved from https://github.com/onnx/models/tree/master/vision/classification/mobilenet

See [Reference Input](#user-content-reference-input) section for details on input and output data generation.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mobilenetv2-1.0.onnx
cd generated_code/mobilenetv2-1
make reference_input
./reference_input network.weights.bin input.data output.data
```

### Inception-V3
Inception-V3 model retrieved from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz and converted using [tensorflow slim](https://github.com/tensorflow/models/tree/master/research/slim).

See [Reference Input](#user-content-reference-input) section for details on input and output data generation.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/inceptionv3.onnx
cd generated_code/inceptionv3
make reference_input
./reference_input network.weights.bin input.data output.data
```

## Speech Recognition
### EKUT-Raw
For those models the reference data has been generated with the script `util/create_sample_data.py` by running an inference using `caffe2` as backend:
```bash
python3.6 util/create_sample_data.py --model PATH_TO_ONNX/model.onnx --file PATH_TO_WAV/soundfile.wav --shape 1 1 16000
```
#### CNN-3-Relu
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/ekut-raw-cnn3-relu.onnx
make reference_input
./reference_input network.weights.bin PATH_TO_WAV/soundfile_model_input.data PATH_TO_WAV/soundfile_model_output.data
```

#### CNN-3-Relu-2
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/ekut-raw-cnn3-relu-2.onnx
make reference_input
./reference_input network.weights.bin PATH_TO_WAV/soundfile_model_input.data PATH_TO_WAV/soundfile_model_output.data
```

#### CNN-6-Relu
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/ekut-raw-cnn6-relu.onnx
make reference_input
./reference_input network.weights.bin PATH_TO_WAV/soundfile_model_input.data PATH_TO_WAV/soundfile_model_output.data
```

#### CNN-6-Relu-Simple
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/ekut-raw-cnn6-relu-simple.onnx
make reference_input
./reference_input network.weights.bin PATH_TO_WAV/soundfile_model_input.data PATH_TO_WAV/soundfile_model_output.data
```

## References
<b id="LeCun1998">[LeCun1998]</b> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. [↩](#cit_LeCun1998)

<b id="Krizhevsky2017">[Krizhevsky2017]</b>  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2017. [↩](#cit_Krizhevsky2017)
