# Pico-CNN

## TL;DR
### Ubuntu
```{bash}
sudo apt install libjpeg-dev
```

### Scientific Linux
```{bash}
sudo yum install libjpeg-devel
```

### All distributions
Install `python3.6` for example with [pyenv](https://github.com/pyenv/pyenv) (probably also works with other Python versions). Then install the required Python packages with the requirements.txt file located in `pico-cnn/onnx_import`:
```{bash}
pip install -r requirements.txt
```

### All networks
For every imported onnx model a `dummy_input.c` will be generated, which uses random numbers as input and calls the network, so that no network specific input data has to be downloaded to run inferences.
```{bash}
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/model.onnx
cd generated_code/model
make dummy_input
./dummy_input network.weights.bin NUM_RUNS
```
If you might want to monitor overall progress by uncommenting `#define PRINT` in `dummy_input.c`.

### LeNet-5
LeNet-5 implementation as proposed by Yann LeCun et. al <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>

**Note: MNIST dataset required to run the LeNet specific code.**
```{bash}
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/lenet.onnx
```
Copy `lenet.c` from examples folder to `generated_code/lenet`.
```{bash}
cd generated_code/lenet
make lenet
./lenet PATH_TO_MNIST network.weights.bin
```

### AlexNet
AlexNet implementation as proposed by Alex Krizhevsky et. al <a id="cit_Krizhevsky2017">[[Krizhevsky2017]](#Krizhevsky2017)</a>

**Note: ImageNet dataset required to run the AlexNet specific code.**
```{bash}
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/alexnet.onnx
```
Copy `alexnet.c` from examples folder to `generated_code/alexnet`.
```{bash}
cd generated_code/alexnet
```
Add `-ljpeg` to `LDFLAGS` in `Makefile`
```{bash}
make alexnet
./alexnet network.weights.bin PATH_TO_IMAGE_MEANS PATH_TO_LABELS PATH_TO_IMAGE
```

### MNIST Multi-Layer-Perceptron (MLP)
```{bash}
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mnist_mlp.onnx
```
Copy `mnist_mlp.c` from examples folder to `generated_code/mnist_mlp`.
```{bash}
cd generated_code/mnist_mlp
make mnist_mlp
./mnist_mlp PATH_TO_MNIST network.weights.bin
```

### MNIST Perceptron
```{bash}
cd onnx_import
python3.6 onnx_to_pico_cnn.py --input PATH_TO_ONNX/mnist_simple_perceptron.onnx
```
Copy `mnist_simple_perceptron.c` from examples folder to `generated_code/mnist_simple_perceptron`.
```{bash}
cd generated_code/mnist_simple_perceptron
make mnist_simple_perceptron
./mnist_simple_perceptron PATH_TO_MNIST network.weights.bin
```

## References
<b id="LeCun1998">[LeCun1998]</b> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. [↩](#cit_LeCun1998)

<b id="Krizhevsky2017">[Krizhevsky2017]</b>  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2017. [↩](#cit_Krizhevsky2017)
