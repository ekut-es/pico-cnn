# Pico-CNN
Pico-CNN is a (almost) library free and lightweight neural network inference framework for embedded systems (Linux and bare-metal) implemented in C. Neural networks can be trained with any training framework which supports export of ONNX (open neural network exchange, [onnx.ai](https://onnx.ai)) and afterwards deployed using Pico-CNN's ONNX import. 

## Setup and Use 
Please read the whole document carefully!

### Ubuntu
```bash
sudo apt install libjpeg-dev
```
If you want to use `cppunit`:
```bash
sudo apt install libcppunit-dev
```

### Scientific Linux
```bash
sudo yum install libjpeg-devel
```
If you want to use `cppunit`:
```bash
git clone --branch cppunit-1.14.0 git://anongit.freedesktop.org/git/libreoffice/cppunit/
cd cppunit
./autogen.sh
./configure
make
sudo make install
sudo ln -s /usr/local/include/cppunit /usr/local/include/CppUnit
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### macOS
```bash
brew install jpeg
```
If you want to use `cppunit`:
```bash
brew install cppunit
```

### C-Standard
Depending on the (system-) compiler you are using you might have to add a specific C-Standard to the `CFLAGS` variable in the generated Makefile. Assuming a modern operating system like Ubuntu 18.04 the default C-Standard is `-std=gnu11`. If you are using an older version of GCC it should suffice to chose `-std=c99` or `-std=gnu99` as the C-Standard.

### All Operating Systems
Install Python in version 3.6.5 for example with [pyenv](https://github.com/pyenv/pyenv) (probably also works with other versions of Python 3.6). Then install the required Python packages with the requirements.txt file located in `pico-cnn/onnx_import`:
```bash
cd onnx_import
pip install -r requirements.txt
```
Of course you can always install the requirements into a virtual environment like this:
```bash
cd onnx_import
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
In the future you always have to activate the environment:
```bash
cd onnx_import
source venv/bin/activate
```

### Testing with `cppunit`
```bash
git clone --branch cppunit-1.14.0 git://anongit.freedesktop.org/git/libreoffice/cppunit/
cd cppunit
./autogen.sh
./configure
make
sudo make install
sudo ln -s /usr/local/include/cppunit /usr/local/include/CppUnit
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Utilities
If you want to use scripts from the `pico-cnn/util` folder you should create a virtual environment for it and install the respective python packages:
```bash
cd utils
python -m venv venv
source venv/bin/activate
pip install -r util_requirements.txt
```
In the future you always have to activate the environment:
```bash
cd utils
source venv/bin/activate
```

## Tested Neural Networks
Pico-CNN was tested with the following neural networks:

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

You can monitor overall progress of the current `RUN` by adding `-DDEBUG` in the generated Makefile.

#### Reference Input
There will also be generated a `reference_input.c` which can be used to validate the imported network against the reference input/output that is provided for onnx models from the official [onnx model-zoo](https://github.com/onnx/models). The data has to be preprocessed:
```bash
python util/parse_onnx_reference_files.py --input PATH_TO_REFERENCE_DATA
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
python3.6 generate_reference_data.py --model model.onnx --file PATH_TO_INPUT_DATA --shape 1 NUM_CHANNELS HEIGHT WIDTH
```
If `--file` is not given the script will use random values instead. Supported file types are `audio/x-wav`, `image/jpeg` and `image/x-portable-greymap`.

## MNIST Dataset
### LeNet-5
LeNet-5 implementation as proposed by Yann LeCun et. al <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a> ONNX model at: [./data/lenet/lenet.onnx](./data/lenet/lenet.onnx)

**Note: MNIST dataset required to run the LeNet specific code.**
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input ../data/lenet/lenet.onnx
```
Copy `examples/lenet.c` from examples folder to `onnx_import/generated_code/lenet`.
```bash
cd generated_code/lenet
make lenet
./lenet PATH_TO_MNIST network.weights.bin
```

### MNIST Multi-Layer-Perceptron (MLP)
ONNX model at: [./data/mnist_mlp/mnist_mlp.onnx](./data/mnist_mlp/mnist_mlp.onnx)

```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input ../data/mnist_mlp/mnist_mlp.onnx
```
Copy `examples/mnist_mlp.c` to `onnx_import/generated_code/mnist_mlp`.
```bash
cd generated_code/mnist_mlp
make mnist_mlp
./mnist_mlp PATH_TO_MNIST network.weights.bin
```

### MNIST Perceptron
Single fully-connected layer. ONNX model at: [./data/mnist_simple_perceptron/mnist_simple_perceptron.onnx](./data/mnist_simple_perceptron/mnist_simple_perceptron.onnx)
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input ../data/mnist_simple_perceptron/mnist_simple_perceptron.onnx
```
Copy `examples/mnist_simple_perceptron.c` to `onnx_import/generated_code/mnist_simple_perceptron`.
```bash
cd generated_code/mnist_simple_perceptron
make mnist_simple_perceptron
./mnist_simple_perceptron PATH_TO_MNIST network.weights.bin
```

## ImageNet Dataset
### AlexNet
AlexNet <a id="cit_Krizhevsky2017">[[Krizhevsky2017]](#Krizhevsky2017)</a> ONNX model retrieved from: [https://github.com/onnx/models/tree/master/vision/classification/alexnet](https://github.com/onnx/models/tree/master/vision/classification/alexnet) 

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
VGG-16 <a id="cit_Simonyan2014">[[Simonyan2014]](#Simonyan2014)</a> ONNX model retrieved from: [https://github.com/onnx/models/tree/master/vision/classification/vgg](https://github.com/onnx/models/tree/master/vision/classification/vgg)

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
VGG-19 <a id="cit_Simonyan2014">[[Simonyan2014]](#Simonyan2014)</a> ONNX model retrieved from: [https://github.com/onnx/models/tree/master/vision/classification/vgg](https://github.com/onnx/models/tree/master/vision/classification/vgg)

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
MobileNet-V2 <a id="cit_Sandler2019">[[Sandler2019]](#Sandler2019)</a> ONNX model retrieved from: [https://github.com/onnx/models/tree/master/vision/classification/mobilenet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)

See [Reference Input](#user-content-reference-input) section for details on input and output data generation.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mobilenetv2-1.0.onnx
cd generated_code/mobilenetv2-1
make reference_input
./reference_input network.weights.bin input.data output.data
```

### Inception-V3
Inception-V3 <a id="cit_Szegedy2014">[[Szegedy2014]](#Szegedy2014)</a> ONNX model retrieved from: [http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) and converted using [TensorFlow slim](https://github.com/tensorflow/models/tree/master/research/slim).

See [Reference Input](#user-content-reference-input) section for details on input and output data generation.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/inceptionv3.onnx
cd generated_code/inceptionv3
make reference_input
./reference_input network.weights.bin input.data output.data
```

### Inception-ResNet-V2
Inception-ResNet-V2 <a id="cit_Szegedy2016">[[Szegedy2016]](#Szegedy2016)</a> ONNX model retrieved from: [http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) and converted using [TensorFlow slim](https://github.com/tensorflow/models/tree/master/research/slim).

See [Reference Input](#user-content-reference-input) section for details on input and output data generation.
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/inception_resnet_v2.onnx
cd generated_code/inception_resnet_v2
make reference_input
./reference_input network.weights.bin input.data output.data
```

## Speech Recognition
### TC-ResNet-8
TC-ResNet-8 <a id="cit_Choi2019">[[Choi2019]](#Choi2019)</a> trained on the [Google Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
```bash
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/tc-res8-update.onnx
cd generated_code/tc-res8-update
make reference_input
./reference_input network.weights.bin input.data output.data
```

## Contributors
* Alexander Jung (Maintainer) [@alexjung](https://github.com/alexjung)
* Konstantin Lübeck (Maintainer) [@k0nze](https://github.com/k0nze)
* Nils Weinhardt (Developer) 


## Citing Pico-CNN
Please cite Pico-CNN in your publications if it helps your research:

```
@inproceedings{luebeck2019picocnn,
    author = {Lübeck, Konstantin and Bringmann, Oliver},
    title = {A Heterogeneous and Reconfigurable Embedded Architecture for Energy-Efficient Execution of Convolutional Neural Networks},
    booktitle = {Architecture of Computing Systems (ARCS 2019)},
    year = {2019},
    month = {May},
    pages = {267--280},
    address = {Copenhagen, Denmark},
    isbn = {978-3-030-18656-2}
}
```

## References
<b id="LeCun1998">[LeCun1998]</b> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. [↩](#cit_LeCun1998)

<b id="Krizhevsky2017">[Krizhevsky2017]</b>  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2017. [↩](#cit_Krizhevsky2017)

<b id="Simonyan2014">[Simonyan2014]</b>  K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv:1409.1556 [cs], Sep. 2014. [↩](#cit_Simonyan2014)

<b id="Sandler2019">[Sandler2019]</b> M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” arXiv:1801.04381 [cs], Mar. 2019. [↩](#cit_Sandler2019)

<b id="Szegedy2014">[Szegedy2014]</b> C. Szegedy et al., “Going Deeper with Convolutions,” arXiv:1409.4842 [cs], Sep. 2014. [↩](#cit_Szegedy2014)

<b id="Szegedy2016">[Szegedy2016]</b> C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,” arXiv:1602.07261 [cs], Aug. 2016. [↩](#cit_Szegedy2016)

<b id="Choi2019">[Choi2019]</b> S. Choi et al., “Temporal Convolution for Real-time Keyword Spotting on Mobile Devices,” arXiv:1904.03814 [cs, eess], Nov. 2019. [↩](#cit_Choi2019)

